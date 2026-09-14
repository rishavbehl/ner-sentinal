"""
NER Logistics Sentinel — API server.

Built on FastAPI + uvicorn. Full async, native WebSockets, interactive Swagger
UI at /docs, ReDoc at /redoc, and OpenAPI 3.1 specification at /openapi.json.
Every endpoint is fully documented and served alongside the frontend dashboard.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (Body, FastAPI, HTTPException, Path, Query, Request,
                     Response, WebSocket, WebSocketDisconnect, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import alerts as alerts_mod
from . import config, db, inference, routing
from .providers import gps as gps_mod
from .providers import weather_openmeteo as wx_live
from .fleet import FLEET
from .features import FEATURE_COLUMNS, FEATURE_LABELS
from .geography import CORRIDORS, NODES, STATES
from .explain import explain_delay, explain_segment_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE_DIR, "frontend")


# --------------------------------------------------------------------------
# PYDANTIC SCHEMAS FOR SWAGGER UI DOCUMENTATION
# --------------------------------------------------------------------------
class IncidentReportSchema(BaseModel):
    road_id: Optional[str] = Field(None, description="Associated road segment ID (e.g. AS_NH27_001). If omitted, automatically snapped to nearest segment.")
    lat: float = Field(..., description="Latitude coordinate of incident")
    lon: float = Field(..., description="Longitude coordinate of incident")
    issue_type: str = Field("road_damage", description="Type of incident: landslide | flood | road_damage | blockage | weather_hazard | accident")
    severity: str = Field("medium", description="Severity level: low | medium | high")
    description: Optional[str] = Field(None, description="Field notes or incident description")
    reporter_id: Optional[str] = Field("FIELD-APP", description="Identifier of reporter or vehicle unit")
    client_id: Optional[str] = Field(None, description="Client UUID for idempotent offline replay deduplication")
    reported_at: Optional[str] = Field(None, description="ISO timestamp of observation (defaults to current system time)")
    source: Optional[str] = Field("field_app_pwa", description="Origin source tag")


class GPSTrackingPayload(BaseModel):
    vehicle_id: str = Field(..., description="Unique vehicle identifier (e.g. TRUCK-01, AIS140-987654)")
    lat: float = Field(..., description="Current latitude coordinate")
    lon: float = Field(..., description="Current longitude coordinate")
    speed_kmh: Optional[float] = Field(0.0, description="Current vehicle speed in km/h")
    heading: Optional[float] = Field(0.0, description="Vehicle heading direction in degrees (0-360)")
    altitude_m: Optional[float] = Field(None, description="Altitude in meters")
    accuracy_m: Optional[float] = Field(None, description="GPS fix horizontal accuracy in meters")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of fix")
    battery_pct: Optional[float] = Field(None, description="Device battery percentage (0-100)")


class ForgetVehiclePayload(BaseModel):
    vehicle_id: str = Field(..., description="Vehicle ID to drop from active tracking")


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def ok(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code)


def err(msg: str, status_code: int = 400, detail: Optional[str] = None) -> JSONResponse:
    body = {"error": msg}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status_code)


def qp(req: Request, key: str, default=None):
    v = req.query_params.get(key)
    return default if v is None or v == "" else v


def qf(req: Request, key: str, default=None):
    v = req.query_params.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def overrides_from_query(req: Request) -> Optional[dict]:
    """What-if scenario parameters, accepted on any state-reading endpoint."""
    ov: Dict[str, Any] = {}
    for k, cast in (("rain_multiplier", float), ("rain_24h_set", float),
                    ("api_multiplier", float), ("api_set", float),
                    ("temperature_delta", float)):
        v = req.query_params.get(k)
        if v not in (None, ""):
            try:
                ov[k] = cast(v)
            except ValueError:
                pass
    cond = req.query_params.get("condition_set")
    if cond:
        ov["condition_set"] = cond
    st = req.query_params.get("states")
    if st:
        ov["states"] = [s.strip().upper() for s in st.split(",") if s.strip()]
    return ov or None


def _point_seg_km(plat, plon, alat, alon, blat, blon) -> float:
    """Great-circle-ish distance from a point to a segment (planar approx)."""
    from .geography import haversine_km
    ax, ay = alon, alat
    bx, by = blon, blat
    px, py = plon, plat
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return haversine_km(plat, plon, alat, alon)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return haversine_km(plat, plon, cy, cx)


# ==========================================================================
# OPENAPI TAGS & FASTAPI INSTANCE
# ==========================================================================
tags_metadata = [
    {
        "name": "Meta & Health",
        "description": "API discovery manifest, system health check, ML model fingerprints, training metrics, and language manifests.",
    },
    {
        "name": "Network & Geography",
        "description": "Scored road networks, national corridors, routable hub nodes, district weather saturation, and segment risk attribution.",
    },
    {
        "name": "Routing & Resilience",
        "description": "Multi-criteria route planning (fastest, balanced, safest, emergency), profile comparisons, departure optimization, choke-point criticality, district accessibility, and climate what-if scenario simulations.",
    },
    {
        "name": "Alerts & Incidents",
        "description": "Multilingual control-room alerting and crowd-sourced / driver field incident reporting.",
    },
    {
        "name": "Live Weather",
        "description": "Open-Meteo weather station ingestion, cache control, and live rainfall provenance.",
    },
    {
        "name": "Tracking & Fleet",
        "description": "Real GPS tracker ingestion (AIS-140/phones), simulated fleet telemetry, SSE/WebSocket streams, and fleet control.",
    },
]


# ==========================================================================
# STARTUP LIFESPAN — live weather refresh loop
# ==========================================================================
async def _weather_loop() -> None:
    """Refresh live weather on boot, then on an interval. Failures are logged
    and swallowed: a dead network must never take the dashboard down."""
    first = True
    while True:
        try:
            res = await asyncio.to_thread(wx_live.refresh, True)
            if res.get("ok"):
                inference.invalidate_cache()
            elif first:
                print("[weather] falling back to the built-in simulated "
                      "climatology — the app is fully functional, just not live.")
        except Exception as e:
            print("[weather] refresh loop error:", e)
        first = False
        await asyncio.sleep(config.WEATHER_INTERVAL_MIN * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "LIVE (Open-Meteo)" if config.LIVE_WEATHER else "SIMULATED"
    print(f"[sentinel] weather source: {mode}")
    print(f"[sentinel] GPS tracking:   "
          f"{'enabled' if config.LIVE_GPS else 'disabled'}"
          f"  -> open /track on a phone to stream real positions")
    print(f"[sentinel] Swagger UI:     http://localhost:8000/docs")
    print(f"[sentinel] ReDoc:          http://localhost:8000/redoc")
    task = asyncio.create_task(_weather_loop()) if config.LIVE_WEATHER else None
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(
    title="NER Logistics Sentinel API",
    description="""
# NER Logistics Sentinel API 🛰️🚛

**AI route-risk and accessibility intelligence for the North Eastern Region of India.**

- **Interactive Swagger UI**: Explore and execute all endpoints below.
- **Alternative ReDoc**: Available at [`/redoc`](/redoc).
- **OpenAPI Schema**: Available at [`/openapi.json`](/openapi.json).
- **Web Dashboard**: Available at [`/`](/).
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# META & HEALTH
# ==========================================================================
@app.get("/api", tags=["Meta & Health"], summary="API Capability Manifest & Index")
async def api_index(req: Request) -> JSONResponse:
    """Return an overview of all Sentinel endpoints, routing profiles, and what-if parameters."""
    return ok({
        "name": "NER Logistics Sentinel",
        "tagline": "AI route-risk and accessibility intelligence for the North Eastern Region",
        "now": inference.now_ts(),
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "endpoints": {
            "GET  /docs": "Interactive Swagger UI documentation",
            "GET  /redoc": "Alternative ReDoc API documentation",
            "GET  /openapi.json": "Full OpenAPI 3.1 schema",
            "GET  /api/health": "liveness + model fingerprint",
            "GET  /api/metrics": "full training + evaluation report",
            "GET  /api/network": "all road segments scored at a timestamp",
            "GET  /api/network/summary": "headline counters",
            "GET  /api/nodes": "routable nodes (towns, hubs, ports, passes)",
            "GET  /api/corridors": "national highway corridors",
            "GET  /api/weather": "weather / soil-saturation grid at a timestamp",
            "GET  /api/segment/{road_id}": "one segment + EXPLAINED prediction",
            "GET  /api/route": "plan routes (origin, dest, profile, k)",
            "GET  /api/route/compare": "side-by-side route comparison table",
            "GET  /api/departure": "48-72 h departure-window optimiser",
            "GET  /api/criticality": "single-point-of-failure ranking",
            "GET  /api/accessibility": "district accessibility index",
            "GET  /api/heatmap": "district-level risk choropleth data",
            "GET  /api/alerts": "multilingual control-room alert feed",
            "GET  /api/languages": "supported alert languages",
            "GET  /api/incidents": "incident log (filterable)",
            "POST /api/incidents": "submit a geo-tagged field report",
            "GET  /api/cargo": "critical-cargo viability check on a route",
            "GET  /api/whatif": "counterfactual scenario vs baseline",
            "GET  /api/weather/status": "live-weather provenance + config",
            "POST /api/weather/refresh": "pull real weather from Open-Meteo now",
            "GET  /api/track": "REAL GPS-tracked vehicles",
            "POST /api/track": "ingest a GPS fix (phone, VTS or AIS-140 unit)",
            "POST /api/track/forget": "drop a tracked vehicle",
            "GET  /api/fleet": "current fleet snapshot",
            "GET  /api/fleet/stream": "LIVE fleet telemetry (Server-Sent Events)",
            "POST /api/fleet/reset": "re-plan and restart the simulation",
            "WS   /ws/fleet": "live fleet position + alert stream",
        },
        "whatif_parameters": ["rain_multiplier", "rain_24h_set", "api_multiplier",
                             "api_set", "temperature_delta", "condition_set",
                             "states"],
        "profiles": {k: {"label": v.label, "description": v.description}
                     for k, v in routing.PROFILES.items()},
    })


@app.get("/api/health", tags=["Meta & Health"], summary="Health Check & Model Fingerprint")
async def health(req: Request) -> JSONResponse:
    """Check API server liveness, loaded machine learning models, and accuracy benchmarks."""
    try:
        m = inference.metrics()
        st = inference.network_state()
        return ok({
            "status": "ok",
            "now": st["ts"],
            "segments": st["summary"]["segments_scored"],
            "models": {
                "risk_classifier": "RandomForestClassifier",
                "delay_regressor": "RandomForestRegressor",
                "route_delay_regressor": "RandomForestRegressor (stage 2)",
            },
            "risk_accuracy_temporal": m["risk_model_temporal"]["accuracy"],
            "blocked_recall_temporal": m["risk_model_temporal"]["blocked_recall"],
            "explainability": "exact additive decision-path attribution",
        })
    except Exception as e:
        return err("not ready — run scripts/pipeline.py first", 503, str(e))


@app.get("/api/metrics", tags=["Meta & Health"], summary="Model Training & Evaluation Metrics")
async def metrics(req: Request) -> JSONResponse:
    """Return detailed cross-validation and temporal hold-out metrics for all Sentinel models."""
    return ok(inference.metrics())


@app.get("/api/languages", tags=["Meta & Health"], summary="Supported Alert Languages")
async def languages(req: Request) -> JSONResponse:
    """Return dictionary of supported Northeast Indian regional languages and state defaults."""
    return ok({
        "languages": alerts_mod.LANGUAGES,
        "state_defaults": {k: {"state": STATES[k], "language": v}
                           for k, v in alerts_mod.STATE_LANGUAGE.items()},
        "note": ("Locales flagged review_pending are structurally correct "
                 "placeholders written without a native speaker and must be "
                 "reviewed before any real deployment."),
    })


# ==========================================================================
# NETWORK & GEOGRAPHY
# ==========================================================================
@app.get("/api/network", tags=["Network & Geography"], summary="Scored Network Road Segments")
async def network(
    req: Request,
    ts: Optional[str] = Query(None, description="ISO timestamp (defaults to current time)"),
    state: Optional[str] = Query(None, description="Comma-separated state filter, e.g. AS,ML"),
    slim: Optional[str] = Query(None, description="If true, omits heavy feature vector dicts"),
    rain_multiplier: Optional[float] = Query(None, description="Scenario rainfall multiplier (e.g. 1.5)"),
    rain_24h_set: Optional[float] = Query(None, description="Scenario 24h rainfall override in mm"),
    api_multiplier: Optional[float] = Query(None, description="Scenario 7-day API multiplier"),
    api_set: Optional[float] = Query(None, description="Scenario 7-day API override"),
    temperature_delta: Optional[float] = Query(None, description="Scenario temperature delta in °C"),
    condition_set: Optional[str] = Query(None, description="Scenario weather condition override"),
    states: Optional[str] = Query(None, description="Scenario states target filter"),
) -> JSONResponse:
    """Retrieve all road segments scored by the ML model with real-time risk, delay, and severity."""
    ov = overrides_from_query(req)
    st = inference.network_state(ts, ov)
    segs = list(st["segments"].values())
    if state:
        want = {s.strip().upper() for s in state.split(",")}
        segs = [s for s in segs if s["state"] in want]
    is_slim = (slim or "").lower() in ("1", "true", "yes")
    if is_slim:
        segs = [{k: v for k, v in s.items() if k != "features"} for s in segs]
    return ok({"ts": st["ts"], "summary": st["summary"], "segments": segs})


@app.get("/api/network/summary", tags=["Network & Geography"], summary="Network Summary Counters")
async def network_summary(
    req: Request,
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Headline network metrics: blocked segments, risky segments, total km blocked, mean severity."""
    st = inference.network_state(qp(req, "ts"), overrides_from_query(req))
    return ok(st["summary"])


@app.get("/api/nodes", tags=["Network & Geography"], summary="Routable Hub Nodes")
async def nodes(req: Request) -> JSONResponse:
    """Return all routable nodes in the NER graph (district headquarters, hubs, border crossings)."""
    return ok({"nodes": [
        {"id": n.id, "name": n.name, "state": n.state,
         "state_name": STATES[n.state], "lat": n.lat, "lon": n.lon,
         "elev_m": n.elev_m, "kind": n.kind, "population_k": n.population_k}
        for n in sorted(NODES.values(), key=lambda x: (x.state, x.name))]})


@app.get("/api/corridors", tags=["Network & Geography"], summary="National Highway Corridors")
async def corridors(req: Request) -> JSONResponse:
    """List major strategic corridors and national highways across the Northeast."""
    return ok({"corridors": [
        {"code": c.code, "name": c.name, "road_class": c.road_class,
         "lanes": c.lanes, "strategic": c.strategic, "notes": c.notes,
         "path": c.path,
         "path_names": [NODES[p].name for p in c.path]}
        for c in CORRIDORS]})


@app.get("/api/weather", tags=["Network & Geography"], summary="District Weather & Soil Saturation")
async def weather(
    req: Request,
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """District-level weather readings, 24h rainfall, and 7-day antecedent precipitation index."""
    timestamp = qp(req, "ts") or inference.now_ts()
    wx = inference.apply_overrides(inference.weather_at(timestamp),
                                   overrides_from_query(req))
    d = inference.districts()
    out = []
    for name, row in wx.items():
        di = d.get(name, {})
        out.append({**row, "lat": di.get("lat"), "lon": di.get("lon"),
                    "state": di.get("state"), "elev_m": di.get("elev_m")})
    return ok({"ts": timestamp, "districts": out})


@app.get("/api/segment/{road_id:path}", tags=["Network & Geography"], summary="Segment Details & AI Explanation")
async def segment_detail(
    req: Request,
    road_id: str = Path(..., description="Road segment ID (e.g. AS_NH27_001)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Return single segment features, recent incident history, and exact decision-path risk attribution."""
    rid = road_id
    st = inference.network_state(qp(req, "ts"), overrides_from_query(req))
    seg = st["segments"].get(rid)
    if not seg:
        return err(f"unknown road_id {rid}", 404)

    import numpy as np
    x = np.array([seg["features"][c] for c in FEATURE_COLUMNS], dtype=np.float64)
    M = inference.models()
    risk_x = explain_segment_risk(M["risk"], x, top_k=7)
    delay_x = explain_delay(M["delay"], x, top_k=5)

    recent = db.query(
        "SELECT incident_id, issue_type, severity, reported_at, description, "
        "reporter_id, verified FROM incidents WHERE road_id = ? "
        "ORDER BY reported_at DESC LIMIT 12", (rid,))
    meta = inference.roads_by_id()[rid]
    return ok({
        "segment": {k: v for k, v in seg.items() if k != "features"},
        "static_attributes": meta,
        "features": seg["features"],
        "feature_labels": FEATURE_LABELS,
        "risk_explanation": risk_x,
        "delay_explanation": delay_x,
        "recent_incidents": recent,
        "alert": alerts_mod.segment_alert(seg),
    })


# ==========================================================================
# ROUTING & RESILIENCE
# ==========================================================================
@app.get("/api/route", tags=["Routing & Resilience"], summary="Plan Multimodal / Multi-criteria Routes")
async def route(
    req: Request,
    origin: str = Query("GUWAHATI", description="Origin node code"),
    dest: str = Query("SHILLONG", description="Destination node code"),
    profile: str = Query("balanced", description="Routing profile: fastest | balanced | safest | emergency"),
    k: int = Query(3, ge=1, le=5, description="Number of candidate alternative routes"),
    cargo: Optional[str] = Query(None, description="Cargo type (blood, oxygen, vaccine, perishable, relief, medicine, general)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Plan K risk-aware routes between origin and destination with delay estimates and cargo check."""
    o = (origin or "").upper()
    d = (dest or "").upper()
    prof = profile or "balanced"
    try:
        res = routing.plan(o, d, prof, ts=ts or qp(req, "ts"), k=max(1, min(k, 5)),
                           overrides=overrides_from_query(req))
    except ValueError as e:
        return err(str(e), 400)
    except Exception:
        return err("routing failed", 500, traceback.format_exc(limit=3))

    cargo_val = cargo or qp(req, "cargo")
    if cargo_val and res.get("routes"):
        res["cargo_check"] = routing.cargo_viability(res["routes"][0], cargo_val)
    return ok(res)


@app.get("/api/route/compare", tags=["Routing & Resilience"], summary="Compare All Routing Profiles Side-by-Side")
async def route_compare(
    req: Request,
    origin: str = Query("GUWAHATI", description="Origin node code"),
    dest: str = Query("SHILLONG", description="Destination node code"),
    profile: str = Query("balanced", description="Primary profile"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Evaluate fastest, balanced, safest, and emergency trade-offs for a given origin-destination pair."""
    o = (origin or "").upper()
    d = (dest or "").upper()
    timestamp = ts or qp(req, "ts")
    ov = overrides_from_query(req)
    out: Dict[str, Any] = {"origin": o, "dest": d, "by_profile": [], "routes": []}
    try:
        for key in ("fastest", "balanced", "safest", "emergency"):
            r = routing.plan(o, d, key, ts=timestamp, k=1, overrides=ov)
            if r.get("routes"):
                b = r["routes"][0]
                out["by_profile"].append({
                    "profile": key,
                    "label": routing.PROFILES[key].label,
                    "distance_km": b["distance_km"],
                    "eta_hours": b["eta_hours"],
                    "eta_text": b["eta_text"],
                    "predicted_delay_hours": b["predicted_delay_hours"],
                    "avg_risk": b["avg_risk"], "max_risk": b["max_risk"],
                    "n_risky": b["n_risky"], "n_blocked": b["n_blocked"],
                    "composite_score": b["composite_score"],
                    "status": b["status"],
                    "path_names": b["path_names"],
                    "worst_segment": b["worst_segment"],
                    "geometry": b["geometry"],
                })
        main = routing.plan(o, d, profile or "balanced", ts=timestamp, k=3,
                            overrides=ov)
        out["routes"] = main.get("routes", [])
        out["comparison_note"] = main.get("comparison_note")
        out["ts"] = main.get("ts")

        distinct = {tuple(r["path_names"]) for r in out["by_profile"]}
        out["distinct_routes_across_profiles"] = len(distinct)
        if len(distinct) == 1 and len(out["by_profile"]) > 1:
            out["no_alternative_note"] = (
                f"All four routing profiles return the SAME road. There is no "
                f"alternative corridor to {NODES[d].name if d in NODES else d} — "
                f"safety cannot be traded for time because there is nothing to "
                f"trade with. This is a structural accessibility failure, not a "
                f"routing result: see /api/criticality and /api/accessibility.")
        elif len(out["by_profile"]) > 1:
            out["no_alternative_note"] = None
    except ValueError as e:
        return err(str(e), 400)
    except Exception:
        return err("comparison failed", 500, traceback.format_exc(limit=3))
    return ok(out)


@app.get("/api/departure", tags=["Routing & Resilience"], summary="Optimal Departure Window Sweeper")
async def departure(
    req: Request,
    origin: str = Query("GUWAHATI", description="Origin node"),
    dest: str = Query("SHILLONG", description="Destination node"),
    profile: str = Query("balanced", description="Profile"),
    horizon: int = Query(72, ge=6, le=78, description="Horizon sweep hours (6 to 78)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
) -> JSONResponse:
    """Simulate route risk every 3 hours over a 48-72h horizon to identify optimal departure timing."""
    o = (origin or "").upper()
    d = (dest or "").upper()
    if o not in NODES or d not in NODES:
        return err("unknown origin or dest", 400)
    hz = horizon or 72
    try:
        return ok(routing.departure_windows(
            o, d, profile or "balanced",
            horizon_hours=max(6, min(hz, 78)), ts=ts or qp(req, "ts")))
    except Exception:
        return err("departure sweep failed", 500, traceback.format_exc(limit=3))


@app.get("/api/criticality", tags=["Routing & Resilience"], summary="Single-Point-of-Failure Ranking")
async def criticality(
    req: Request,
    hub: Optional[str] = Query(None, description="Central supply hub node ID (default GUWAHATI)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Rank road segments whose disruption isolates the highest number of districts or population."""
    try:
        selected_hub = (hub or qp(req, "hub", routing.SUPPLY_HUB) or routing.SUPPLY_HUB).upper()
        return ok(routing.criticality(ts or qp(req, "ts"),
                                      hub=selected_hub,
                                      overrides=overrides_from_query(req)))
    except Exception:
        return err("criticality failed", 500, traceback.format_exc(limit=3))


@app.get("/api/accessibility", tags=["Routing & Resilience"], summary="District Accessibility & Isolation Index")
async def accessibility(
    req: Request,
    hub: Optional[str] = Query(None, description="Central supply hub node ID (default GUWAHATI)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Measure connectivity, travel time, and vulnerability for every district capital in the Northeast."""
    try:
        selected_hub = (hub or qp(req, "hub", routing.SUPPLY_HUB) or routing.SUPPLY_HUB).upper()
        return ok(routing.accessibility(ts or qp(req, "ts"),
                                        hub=selected_hub,
                                        overrides=overrides_from_query(req)))
    except Exception:
        return err("accessibility failed", 500, traceback.format_exc(limit=3))


@app.get("/api/heatmap", tags=["Routing & Resilience"], summary="District-Level Risk Choropleth Data")
async def heatmap(
    req: Request,
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """District-level aggregations of road length, blocked km, and mean severity for GIS choropleth maps."""
    st = inference.network_state(ts or qp(req, "ts"), overrides_from_query(req))
    dist = inference.districts()
    agg: Dict[str, dict] = {}
    for s in st["segments"].values():
        a = agg.setdefault(s["district"], {
            "district": s["district"], "state": s["state"],
            "state_name": STATES.get(s["state"], s["state"]),
            "lat": dist.get(s["district"], {}).get("lat"),
            "lon": dist.get(s["district"], {}).get("lon"),
            "segments": 0, "km": 0.0, "km_blocked": 0.0, "km_risky": 0.0,
            "sev_sum": 0.0, "blocked": 0, "risky": 0, "safe": 0,
            "max_severity": 0.0, "rainfall_24h_mm": s["weather"]["rainfall_24h_mm"],
            "api_7d": s["weather"]["api_7d"],
            "condition": s["weather"]["condition"],
        })
        a["segments"] += 1
        a["km"] += s["length_km"]
        a["sev_sum"] += s["severity"] * s["length_km"]
        a["max_severity"] = max(a["max_severity"], s["severity"])
        if s["risk_label"] == 2:
            a["blocked"] += 1
            a["km_blocked"] += s["length_km"]
        elif s["risk_label"] == 1:
            a["risky"] += 1
            a["km_risky"] += s["length_km"]
        else:
            a["safe"] += 1

    out = []
    for a in agg.values():
        a["mean_severity"] = round(a["sev_sum"] / max(a["km"], 0.01), 4)
        a["km"] = round(a["km"], 1)
        a["km_blocked"] = round(a["km_blocked"], 1)
        a["km_risky"] = round(a["km_risky"], 1)
        a.pop("sev_sum")
        a["grade"] = ("critical" if a["mean_severity"] >= 0.55 else
                      "high" if a["mean_severity"] >= 0.38 else
                      "moderate" if a["mean_severity"] >= 0.22 else "low")
        out.append(a)
    out.sort(key=lambda x: -x["mean_severity"])
    return ok({"ts": st["ts"], "districts": out, "hotspots": out[:12]})


@app.get("/api/cargo", tags=["Routing & Resilience"], summary="Critical Cargo Viability & Spoilage Check")
async def cargo(
    req: Request,
    origin: str = Query("GUWAHATI", description="Origin node"),
    dest: str = Query("SHILLONG", description="Destination node"),
    cargo_type: str = Query("vaccine", description="blood | oxygen | vaccine | perishable | relief | medicine | general"),
    profile: str = Query("emergency", description="Routing profile"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Check whether time-sensitive cargo will spoil given predicted delays along the emergency route."""
    o = (origin or "").upper()
    d = (dest or "").upper()
    ctype = cargo_type or "vaccine"
    if o not in NODES or d not in NODES:
        return err("unknown origin or dest", 400)
    res = routing.plan(o, d, profile or "emergency", ts=ts or qp(req, "ts"), k=2,
                       overrides=overrides_from_query(req))
    if not res.get("routes"):
        return err("no route", 404)
    best = res["routes"][0]
    check = routing.cargo_viability(best, ctype)
    al = None
    if check["severity"] == "critical":
        al = alerts_mod.cargo_alert(
            check["cargo_label"], NODES[d].name, best["eta_text"],
            int(check["max_transit_hours"]), NODES[d].state, check["advice"])
    return ok({
        "origin": o, "dest": d, "cargo": check, "route": best,
        "cargo_types": routing.CARGO, "alert": al,
    })


@app.get("/api/whatif", tags=["Routing & Resilience"], summary="Climate What-If Counterfactual Simulator")
async def whatif(
    req: Request,
    rain_multiplier: Optional[float] = Query(None, description="Precipitation multiplier (e.g. 2.0 for 2x rain)"),
    rain_24h_set: Optional[float] = Query(None, description="Fixed 24h rainfall in mm"),
    api_multiplier: Optional[float] = Query(None, description="API 7-day saturation multiplier"),
    api_set: Optional[float] = Query(None, description="Fixed API 7-day value"),
    temperature_delta: Optional[float] = Query(None, description="Temperature delta in Celsius"),
    condition_set: Optional[str] = Query(None, description="Weather condition (e.g. monsoon_downpour)"),
    states: Optional[str] = Query(None, description="Target states (e.g. ML,AS)"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
) -> JSONResponse:
    """Compare baseline network conditions against simulated extreme weather or disruption events."""
    timestamp = ts or qp(req, "ts")
    ov = overrides_from_query(req)
    if not ov:
        return err("supply at least one scenario parameter "
                   "(rain_24h_set, rain_multiplier, api_set, api_multiplier, "
                   "temperature_delta, condition_set; optional states=ML,AS)", 400)
    base = inference.network_state(timestamp)
    cf = inference.network_state(timestamp, ov)

    changed = []
    for rid, b in base["segments"].items():
        c = cf["segments"][rid]
        if c["risk_label"] != b["risk_label"]:
            changed.append({
                "road_id": rid, "name": b["name"], "corridor": b["corridor"],
                "state": b["state"], "coords": b["coords"],
                "from": b["risk_status"], "to": c["risk_status"],
                "severity_before": b["severity"], "severity_after": c["severity"],
                "delay_before": b["delay_hours"], "delay_after": c["delay_hours"],
                "direction": "worse" if c["risk_label"] > b["risk_label"] else "better",
            })
    changed.sort(key=lambda x: -(x["severity_after"] - x["severity_before"]))

    newly_isolated: List[dict] = []
    try:
        ab = routing.accessibility(timestamp)
        ac = routing.accessibility(timestamp, overrides=ov)
        bidx = {d["node_id"]: d for d in ab["districts"]}
        for d in ac["districts"]:
            b = bidx.get(d["node_id"])
            if not b:
                continue
            drop = (b["index"] or 0) - (d["index"] or 0)
            if drop >= 8:
                newly_isolated.append({
                    "district": d["district"], "state": d["state"],
                    "index_before": b["index"], "index_after": d["index"],
                    "drop": round(drop, 1),
                    "grade_before": b["grade"], "grade_after": d["grade"],
                    "hours_before": b["travel_hours"], "hours_after": d["travel_hours"],
                })
        newly_isolated.sort(key=lambda x: -x["drop"])
    except Exception:
        pass

    return ok({
        "ts": base["ts"],
        "scenario": ov,
        "baseline": base["summary"],
        "counterfactual": cf["summary"],
        "delta": {
            "blocked": cf["summary"]["blocked"] - base["summary"]["blocked"],
            "risky": cf["summary"]["risky"] - base["summary"]["risky"],
            "safe": cf["summary"]["safe"] - base["summary"]["safe"],
            "km_blocked": round(cf["summary"]["km_blocked"] -
                                base["summary"]["km_blocked"], 1),
            "mean_severity": round(cf["summary"]["mean_severity"] -
                                   base["summary"]["mean_severity"], 4),
        },
        "segments_changed": changed[:60],
        "n_segments_changed": len(changed),
        "accessibility_impact": newly_isolated[:15],
    })


# ==========================================================================
# ALERTS & INCIDENTS
# ==========================================================================
@app.get("/api/alerts", tags=["Alerts & Incidents"], summary="Multilingual Control Room Alert Feed")
async def get_alerts(
    req: Request,
    limit: int = Query(30, ge=1, le=80, description="Max alerts to return"),
    lang: Optional[str] = Query(None, description="ISO language filter (e.g. as, bn, mni, hi, en)"),
    persist: bool = Query(False, description="Persist generated alerts to SQLite log"),
    ts: Optional[str] = Query(None, description="ISO timestamp"),
    rain_multiplier: Optional[float] = Query(None),
    rain_24h_set: Optional[float] = Query(None),
    api_multiplier: Optional[float] = Query(None),
    api_set: Optional[float] = Query(None),
    temperature_delta: Optional[float] = Query(None),
    condition_set: Optional[str] = Query(None),
    states: Optional[str] = Query(None),
) -> JSONResponse:
    """Retrieve prioritized control room alerts with native multilingual translations."""
    st = inference.network_state(ts or qp(req, "ts"), overrides_from_query(req))
    lim = limit or 30
    feed = alerts_mod.build_alert_feed(st, limit=max(1, min(lim, 80)))
    lang_val = lang or qp(req, "lang")
    if lang_val:
        for a in feed:
            if lang_val in a.get("text", {}):
                a["text"] = {lang_val: a["text"][lang_val], "en": a["text"].get("en")}
    if persist or qp(req, "persist") in ("1", "true"):
        try:
            alerts_mod.persist(feed)
        except Exception:
            pass
    return ok({"ts": st["ts"], "count": len(feed), "alerts": feed})


@app.get("/api/incidents", tags=["Alerts & Incidents"], summary="Incident Log & Filter")
async def get_incidents(
    req: Request,
    road_id: Optional[str] = Query(None, description="Filter by road ID"),
    state: Optional[str] = Query(None, description="Filter by state (e.g. AS, ML)"),
    issue_type: Optional[str] = Query(None, description="Filter by issue type (landslide, flood, etc.)"),
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high)"),
    since: Optional[str] = Query(None, description="Filter incidents reported after ISO timestamp"),
    limit: int = Query(200, ge=1, le=2000, description="Max records to return"),
) -> JSONResponse:
    """Query field-reported incidents from SQLite with optional filtering by location or type."""
    where, params = [], []
    r_id = road_id or qp(req, "road_id")
    if r_id:
        where.append("road_id = ?")
        params.append(r_id)
    st = state or qp(req, "state")
    if st:
        where.append("state = ?")
        params.append(st.upper())
    iss = issue_type or qp(req, "issue_type")
    if iss:
        where.append("issue_type = ?")
        params.append(iss)
    sev = severity or qp(req, "severity")
    if sev:
        where.append("severity = ?")
        params.append(sev)
    sc = since or qp(req, "since")
    if sc:
        where.append("reported_at >= ?")
        params.append(sc)
    sql = "SELECT * FROM incidents"
    if where:
        sql += " WHERE " + " AND ".join(where)
    lim = limit or 200
    sql += f" ORDER BY reported_at DESC LIMIT {max(1, min(lim, 2000))}"
    rows = db.query(sql, tuple(params))
    stats = db.query("SELECT issue_type, COUNT(*) n FROM incidents "
                     "GROUP BY issue_type ORDER BY n DESC")
    return ok({"count": len(rows), "incidents": rows, "by_type": stats})


@app.post("/api/incidents", tags=["Alerts & Incidents"], summary="Submit Geo-Tagged Incident Report", status_code=201)
async def post_incident(
    req: Request,
    payload: Optional[IncidentReportSchema] = Body(None, description="JSON incident report body"),
) -> JSONResponse:
    """
    Field-reporter ingestion. Accepts JSON body or multipart form data (with optional photo attachment).
    Supports client_id de-duplication for offline PWA sync.
    """
    ct = req.headers.get("content-type", "")
    photo_url = None
    data: Dict[str, Any] = {}

    try:
        if "multipart/form-data" in ct:
            form = await req.form()
            data = {k: form[k] for k in form if k != "photo"}
            up = form.get("photo")
            if up is not None and hasattr(up, "filename") and up.filename:
                ext = os.path.splitext(up.filename)[1][:6] or ".jpg"
                fn = f"{uuid.uuid4().hex[:12]}{ext}"
                path = os.path.join(db.UPLOAD_DIR, fn)
                with open(path, "wb") as f:
                    content = await up.read()
                    f.write(content)
                photo_url = f"/uploads/{fn}"
        elif payload is not None:
            data = payload.model_dump(exclude_none=True)
        else:
            data = await req.json()
    except Exception as e:
        return err("could not parse request body", 400, str(e))

    def g(k, default=None):
        v = data.get(k, default)
        return v if v not in ("", None) else default

    try:
        lat = float(g("lat"))
        lon = float(g("lon"))
    except (TypeError, ValueError):
        return err("lat and lon are required and must be numeric", 400)

    issue = g("issue_type", "road_damage")
    sev = g("severity", "medium")
    if sev not in ("low", "medium", "high"):
        return err("severity must be low | medium | high", 400)

    # De-duplicate offline replays
    client_id = g("client_id")
    if client_id:
        existing = db.query_one(
            "SELECT incident_id FROM incidents WHERE incident_id = ?",
            (f"FLD-{client_id[:14]}",))
        if existing:
            return ok({"status": "duplicate_ignored",
                       "incident_id": existing["incident_id"]}, 200)

    # Snap to the nearest road segment
    rid = g("road_id")
    snapped = None
    if not rid:
        best, bestd = None, 1e9
        for r in inference.roads():
            d = _point_seg_km(lat, lon, r["start_lat"], r["start_lon"],
                              r["end_lat"], r["end_lon"])
            if d < bestd:
                best, bestd = r, d
        if best is not None:
            rid = best["road_id"]
            snapped = {"road_id": rid, "name": best["name"],
                       "corridor": best["corridor"],
                       "distance_km": round(bestd, 3)}
    meta = inference.roads_by_id().get(rid)
    if meta is None:
        return err("could not associate the report with a road segment", 400)

    iid = f"FLD-{client_id[:14]}" if client_id else f"FLD-{uuid.uuid4().hex[:10]}"
    reported_at = g("reported_at") or inference.now_ts()
    if reported_at > inference.now_ts():
        reported_at = inference.now_ts()
    row = (iid, rid, meta["district"], meta["state"], lat, lon, issue, sev,
           g("description", f"{issue.replace('_', ' ').title()} reported from field"),
           reported_at, g("reporter_id", "FIELD-APP"),
           g("source", "field_app_pwa"), photo_url, 0, None)
    db.execute("INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               row)
    routing.reset_caches()
    inference.invalidate_cache()

    return ok({
        "status": "accepted", "incident_id": iid, "road_id": rid,
        "snapped_to": snapped, "photo_url": photo_url,
        "district": meta["district"], "state": meta["state"],
        "note": ("Report stored. It feeds the incident-memory features "
                 "(hist_incidents_90d, days_since_last_incident) on the next "
                 "risk inference for this segment."),
    }, 201)


# ==========================================================================
# LIVE WEATHER
# ==========================================================================
@app.get("/api/weather/status", tags=["Live Weather"], summary="Live Weather Provenance & Status")
async def weather_status(req: Request) -> JSONResponse:
    """Return status of the Open-Meteo weather station pipeline and refresh timestamp."""
    last = wx_live.last_refresh()
    return ok({
        "config": config.describe(),
        "last_live_refresh": last,
        "source": "open-meteo" if last else "simulated (built-in climatology)",
        "provider_note": (
            "Open-Meteo needs no API key. The adapter computes rainfall_24h, "
            "rainfall_72h and the 7-day antecedent index with exactly the same "
            "definitions the models were trained on — see "
            "backend/providers/weather_openmeteo.py."),
    })


@app.post("/api/weather/refresh", tags=["Live Weather"], summary="Force Refresh Live Weather")
async def weather_refresh(req: Request) -> JSONResponse:
    """Fetch latest weather observation directly from Open-Meteo and update model cache."""
    res = await asyncio.to_thread(wx_live.refresh, False)
    inference.invalidate_cache()
    if res.get("ok"):
        res["summary"] = inference.network_state()["summary"]
    return ok(res, 200 if res.get("ok") else 503)


# ==========================================================================
# REAL GPS TRACKING & FLEET
# ==========================================================================
@app.get("/api/track", tags=["Tracking & Fleet"], summary="List Active Real-Time Tracked Vehicles")
async def track_get(req: Request) -> JSONResponse:
    """List real-time GPS coordinates and snapped road locations of connected drivers."""
    return ok(gps_mod.live_fleet())


@app.post("/api/track", tags=["Tracking & Fleet"], summary="Ingest Real-Time GPS Fix", status_code=201)
async def track_post(
    req: Request,
    payload: Optional[GPSTrackingPayload] = Body(None, description="GPS tracking fix JSON payload"),
) -> JSONResponse:
    """Ingest a GPS fix from a phone browser, VTS device, or AIS-140 in-vehicle tracker."""
    data: Dict[str, Any] = {}
    if payload is not None:
        data = payload.model_dump(exclude_none=True)
    else:
        try:
            data = await req.json()
        except Exception:
            try:
                form = await req.form()
                data = {k: form[k] for k in form}
            except Exception:
                return err("could not parse body — send JSON with lat/lon", 400)
    try:
        res = await asyncio.to_thread(gps_mod.ingest, data)
    except ValueError as e:
        return err(str(e), 400)
    except Exception:
        return err("tracking failed", 500, traceback.format_exc(limit=3))
    return ok(res, 201)


@app.post("/api/track/forget", tags=["Tracking & Fleet"], summary="Drop Tracked Vehicle")
async def track_delete(
    req: Request,
    payload: Optional[ForgetVehiclePayload] = Body(None, description="Vehicle ID payload to drop"),
    vehicle_id: Optional[str] = Query(None, description="Vehicle ID as query parameter"),
) -> JSONResponse:
    """Remove a vehicle from the active tracking map."""
    vid = (payload.vehicle_id if payload else None) or vehicle_id or qp(req, "vehicle_id")
    if not vid:
        return err("vehicle_id is required", 400)
    return ok({"forgotten": gps_mod.forget(vid), "vehicle_id": vid})


@app.get("/api/fleet", tags=["Tracking & Fleet"], summary="Simulated Fleet Snapshot")
async def fleet_snapshot(req: Request) -> JSONResponse:
    """Current positions, assigned routes, and delay alerts for simulated fleet trucks."""
    return ok(FLEET.snapshot())


@app.get("/api/fleet/stream", tags=["Tracking & Fleet"], summary="Live Fleet Telemetry (SSE)")
async def fleet_stream(req: Request) -> Response:
    """Live Server-Sent Events (SSE) telemetry stream pushing truck positions every second."""
    async def gen():
        q = await FLEET.subscribe()
        try:
            snap = FLEET.snapshot()
            yield f"event: snapshot\ndata: {json.dumps(snap)}\n\n".encode()
            while True:
                if await req.is_disconnected():
                    break
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                yield f"event: tick\ndata: {json.dumps(frame)}\n\n".encode()
        finally:
            FLEET.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/fleet/reset", tags=["Tracking & Fleet"], summary="Reset Fleet Simulation")
async def fleet_reset(req: Request) -> JSONResponse:
    """Re-plan all simulated delivery routes and restart vehicle positions from hubs."""
    FLEET.reset()
    return ok({"status": "reset", "vehicles": len(FLEET.vehicles)})


@app.websocket("/ws/fleet")
async def ws_fleet(ws: WebSocket) -> None:
    """Bidirectional WebSocket connection for live fleet positions and alerts."""
    await ws.accept()
    q = await FLEET.subscribe()
    try:
        await ws.send_json(FLEET.snapshot())
        while True:
            frame = await q.get()
            await ws.send_json(frame)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        FLEET.unsubscribe(q)


# ==========================================================================
# STATIC / PAGES / FALLBACK
# ==========================================================================
@app.get("/", include_in_schema=False)
async def page_index(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "index.html"))


@app.get("/field", include_in_schema=False)
async def page_field(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "field.html"))


@app.get("/track", include_in_schema=False)
async def page_track(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "track.html"))


@app.get("/sw.js", include_in_schema=False)
async def sw_js(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "sw.js"),
                        media_type="application/javascript")


@app.exception_handler(404)
async def not_found(req: Request, exc) -> Response:
    if req.url.path.startswith("/api"):
        return err("no such endpoint — see GET /api or /docs", 404)
    return FileResponse(os.path.join(FRONTEND, "index.html"))


@app.exception_handler(500)
async def server_error(req: Request, exc) -> Response:
    return err("internal error", 500, str(exc))


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
app.mount("/uploads", StaticFiles(directory=db.UPLOAD_DIR), name="uploads")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, log_level="info")
