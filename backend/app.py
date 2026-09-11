"""
NER Logistics Sentinel — API server.

Built on Starlette (the ASGI core FastAPI itself is built on) + uvicorn. Full
async, native WebSockets, zero additional dependencies. Every endpoint is
documented at GET /api and the whole thing is served alongside the dashboard so
`./run.sh` is the only command needed.
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

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (FileResponse, JSONResponse, Response,
                                 StreamingResponse)
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

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
def ok(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


def err(msg: str, status: int = 400, detail: Optional[str] = None) -> JSONResponse:
    body = {"error": msg}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status)


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


# ==========================================================================
# META
# ==========================================================================
async def api_index(req: Request) -> JSONResponse:
    return ok({
        "name": "NER Logistics Sentinel",
        "tagline": "AI route-risk and accessibility intelligence for the "
                   "North Eastern Region",
        "now": inference.now_ts(),
        "endpoints": {
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


async def health(req: Request) -> JSONResponse:
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


async def metrics(req: Request) -> JSONResponse:
    return ok(inference.metrics())


async def languages(req: Request) -> JSONResponse:
    return ok({
        "languages": alerts_mod.LANGUAGES,
        "state_defaults": {k: {"state": STATES[k], "language": v}
                           for k, v in alerts_mod.STATE_LANGUAGE.items()},
        "note": ("Locales flagged review_pending are structurally correct "
                 "placeholders written without a native speaker and must be "
                 "reviewed before any real deployment."),
    })


# ==========================================================================
# NETWORK
# ==========================================================================
async def network(req: Request) -> JSONResponse:
    ts = qp(req, "ts")
    ov = overrides_from_query(req)
    st = inference.network_state(ts, ov)
    state_filter = qp(req, "state")
    segs = list(st["segments"].values())
    if state_filter:
        want = {s.strip().upper() for s in state_filter.split(",")}
        segs = [s for s in segs if s["state"] in want]
    slim = qp(req, "slim", "0") in ("1", "true", "yes")
    if slim:
        segs = [{k: v for k, v in s.items() if k != "features"} for s in segs]
    return ok({"ts": st["ts"], "summary": st["summary"], "segments": segs})


async def network_summary(req: Request) -> JSONResponse:
    st = inference.network_state(qp(req, "ts"), overrides_from_query(req))
    return ok(st["summary"])


async def nodes(req: Request) -> JSONResponse:
    return ok({"nodes": [
        {"id": n.id, "name": n.name, "state": n.state,
         "state_name": STATES[n.state], "lat": n.lat, "lon": n.lon,
         "elev_m": n.elev_m, "kind": n.kind, "population_k": n.population_k}
        for n in sorted(NODES.values(), key=lambda x: (x.state, x.name))]})


async def corridors(req: Request) -> JSONResponse:
    return ok({"corridors": [
        {"code": c.code, "name": c.name, "road_class": c.road_class,
         "lanes": c.lanes, "strategic": c.strategic, "notes": c.notes,
         "path": c.path,
         "path_names": [NODES[p].name for p in c.path]}
        for c in CORRIDORS]})


async def weather(req: Request) -> JSONResponse:
    ts = qp(req, "ts") or inference.now_ts()
    wx = inference.apply_overrides(inference.weather_at(ts),
                                   overrides_from_query(req))
    d = inference.districts()
    out = []
    for name, row in wx.items():
        di = d.get(name, {})
        out.append({**row, "lat": di.get("lat"), "lon": di.get("lon"),
                    "state": di.get("state"), "elev_m": di.get("elev_m")})
    return ok({"ts": ts, "districts": out})


async def segment_detail(req: Request) -> JSONResponse:
    rid = req.path_params["road_id"]
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
# ROUTING
# ==========================================================================
async def route(req: Request) -> JSONResponse:
    o = (qp(req, "origin") or "").upper()
    d = (qp(req, "dest") or "").upper()
    prof = qp(req, "profile", "balanced")
    k = int(qf(req, "k", 3) or 3)
    try:
        res = routing.plan(o, d, prof, ts=qp(req, "ts"), k=max(1, min(k, 5)),
                           overrides=overrides_from_query(req))
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err("routing failed", 500, traceback.format_exc(limit=3))

    cargo = qp(req, "cargo")
    if cargo and res.get("routes"):
        res["cargo_check"] = routing.cargo_viability(res["routes"][0], cargo)
    return ok(res)


async def route_compare(req: Request) -> JSONResponse:
    """One row per candidate route — the side-by-side table the UI renders."""
    o = (qp(req, "origin") or "").upper()
    d = (qp(req, "dest") or "").upper()
    ts = qp(req, "ts")
    ov = overrides_from_query(req)
    out: Dict[str, Any] = {"origin": o, "dest": d, "by_profile": [], "routes": []}
    try:
        # the same OD under every profile — shows the trade-off explicitly
        for key in ("fastest", "balanced", "safest", "emergency"):
            r = routing.plan(o, d, key, ts=ts, k=1, overrides=ov)
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
        main = routing.plan(o, d, qp(req, "profile", "balanced"), ts=ts, k=3,
                            overrides=ov)
        out["routes"] = main.get("routes", [])
        out["comparison_note"] = main.get("comparison_note")
        out["ts"] = main.get("ts")

        # When every profile returns the same road, that is a FINDING, not a
        # bug: the destination has no route choice at all. Say so — it is the
        # single most important fact about NER logistics.
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


async def departure(req: Request) -> JSONResponse:
    o = (qp(req, "origin") or "").upper()
    d = (qp(req, "dest") or "").upper()
    if o not in NODES or d not in NODES:
        return err("unknown origin or dest", 400)
    hz = int(qf(req, "horizon", 72) or 72)
    try:
        return ok(routing.departure_windows(
            o, d, qp(req, "profile", "balanced"),
            horizon_hours=max(6, min(hz, 78)), ts=qp(req, "ts")))
    except Exception:
        return err("departure sweep failed", 500, traceback.format_exc(limit=3))


async def criticality(req: Request) -> JSONResponse:
    try:
        return ok(routing.criticality(qp(req, "ts"),
                                      hub=(qp(req, "hub", routing.SUPPLY_HUB) or
                                           routing.SUPPLY_HUB).upper(),
                                      overrides=overrides_from_query(req)))
    except Exception:
        return err("criticality failed", 500, traceback.format_exc(limit=3))


async def accessibility(req: Request) -> JSONResponse:
    try:
        return ok(routing.accessibility(qp(req, "ts"),
                                        hub=(qp(req, "hub", routing.SUPPLY_HUB) or
                                             routing.SUPPLY_HUB).upper(),
                                        overrides=overrides_from_query(req)))
    except Exception:
        return err("accessibility failed", 500, traceback.format_exc(limit=3))


async def heatmap(req: Request) -> JSONResponse:
    """District-level aggregation for the choropleth layer."""
    st = inference.network_state(qp(req, "ts"), overrides_from_query(req))
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
    return ok({"ts": st["ts"], "districts": out,
               "hotspots": out[:12]})


# ==========================================================================
# ALERTS + INCIDENTS
# ==========================================================================
async def get_alerts(req: Request) -> JSONResponse:
    st = inference.network_state(qp(req, "ts"), overrides_from_query(req))
    lim = int(qf(req, "limit", 30) or 30)
    feed = alerts_mod.build_alert_feed(st, limit=max(1, min(lim, 80)))
    lang = qp(req, "lang")
    if lang:
        for a in feed:
            if lang in a.get("text", {}):
                a["text"] = {lang: a["text"][lang], "en": a["text"].get("en")}
    if qp(req, "persist") in ("1", "true"):
        try:
            alerts_mod.persist(feed)
        except Exception:
            pass
    return ok({"ts": st["ts"], "count": len(feed), "alerts": feed})


async def get_incidents(req: Request) -> JSONResponse:
    where, params = [], []
    if qp(req, "road_id"):
        where.append("road_id = ?")
        params.append(qp(req, "road_id"))
    if qp(req, "state"):
        where.append("state = ?")
        params.append(qp(req, "state").upper())
    if qp(req, "issue_type"):
        where.append("issue_type = ?")
        params.append(qp(req, "issue_type"))
    if qp(req, "severity"):
        where.append("severity = ?")
        params.append(qp(req, "severity"))
    if qp(req, "since"):
        where.append("reported_at >= ?")
        params.append(qp(req, "since"))
    sql = "SELECT * FROM incidents"
    if where:
        sql += " WHERE " + " AND ".join(where)
    lim = int(qf(req, "limit", 200) or 200)
    sql += f" ORDER BY reported_at DESC LIMIT {max(1, min(lim, 2000))}"
    rows = db.query(sql, tuple(params))
    stats = db.query("SELECT issue_type, COUNT(*) n FROM incidents "
                     "GROUP BY issue_type ORDER BY n DESC")
    return ok({"count": len(rows), "incidents": rows, "by_type": stats})


async def post_incident(req: Request) -> JSONResponse:
    """
    Field-reporter ingestion. Accepts JSON or multipart (with a photo).
    Designed for the offline PWA: the client queues reports locally and replays
    them here when a signal returns, so `client_id` de-duplicates retries.
    """
    ct = req.headers.get("content-type", "")
    photo_url = None
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
                    f.write(await up.read())
                photo_url = f"/uploads/{fn}"
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
    # Stamp against the SYSTEM clock the rest of the app runs on, not wall clock.
    # The incident-memory features select `reported_at <= ts`, so a report
    # stamped in the future would be silently invisible to the very model it is
    # supposed to inform — the feedback loop has to actually close.
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
# CARGO + WHAT-IF
# ==========================================================================
async def cargo(req: Request) -> JSONResponse:
    o = (qp(req, "origin") or "").upper()
    d = (qp(req, "dest") or "").upper()
    ctype = qp(req, "cargo_type", "vaccine")
    if o not in NODES or d not in NODES:
        return err("unknown origin or dest", 400)
    res = routing.plan(o, d, qp(req, "profile", "emergency"), ts=qp(req, "ts"), k=2,
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


async def whatif(req: Request) -> JSONResponse:
    """Baseline vs counterfactual, with the delta that makes the point."""
    ts = qp(req, "ts")
    ov = overrides_from_query(req)
    if not ov:
        return err("supply at least one scenario parameter "
                   "(rain_24h_set, rain_multiplier, api_set, api_multiplier, "
                   "temperature_delta, condition_set; optional states=ML,AS)", 400)
    base = inference.network_state(ts)
    cf = inference.network_state(ts, ov)

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
        ab = routing.accessibility(ts)
        ac = routing.accessibility(ts, overrides=ov)
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
# FLEET
# ==========================================================================
async def fleet_snapshot(req: Request) -> JSONResponse:
    return ok(FLEET.snapshot())


async def fleet_reset(req: Request) -> JSONResponse:
    FLEET.reset()
    return ok({"status": "reset", "vehicles": len(FLEET.vehicles)})


async def fleet_stream(req: Request) -> Response:
    """
    Live fleet telemetry over Server-Sent Events.

    SSE is the PRIMARY transport here, deliberately. Fleet telemetry is strictly
    one-way (server → control room), which is exactly what SSE is for: it runs
    over plain HTTP/1.1 with no extra dependency, reconnects automatically in
    the browser, and passes through corporate proxies that drop WebSocket
    upgrades. The /ws/fleet WebSocket below is kept for clients that want
    bidirectional control and activates automatically when a WebSocket library
    is installed.
    """
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


async def ws_fleet(ws: WebSocket) -> None:
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
# LIVE WEATHER
# ==========================================================================
async def weather_status(req: Request) -> JSONResponse:
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


async def weather_refresh(req: Request) -> JSONResponse:
    """Pull live weather now. Safe to call any time; never takes the app down."""
    res = await asyncio.to_thread(wx_live.refresh, False)
    inference.invalidate_cache()
    if res.get("ok"):
        res["summary"] = inference.network_state()["summary"]
    return ok(res, 200 if res.get("ok") else 503)


# ==========================================================================
# REAL GPS TRACKING
# ==========================================================================
async def track_post(req: Request) -> JSONResponse:
    try:
        payload = await req.json()
    except Exception:
        try:
            form = await req.form()
            payload = {k: form[k] for k in form}
        except Exception:
            return err("could not parse body — send JSON with lat/lon", 400)
    try:
        res = await asyncio.to_thread(gps_mod.ingest, payload)
    except ValueError as e:
        return err(str(e), 400)
    except Exception:
        return err("tracking failed", 500, traceback.format_exc(limit=3))
    return ok(res, 201)


async def track_get(req: Request) -> JSONResponse:
    return ok(gps_mod.live_fleet())


async def track_delete(req: Request) -> JSONResponse:
    vid = qp(req, "vehicle_id")
    if not vid:
        return err("vehicle_id is required", 400)
    return ok({"forgotten": gps_mod.forget(vid), "vehicle_id": vid})


async def page_track(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "track.html"))


# ==========================================================================
# STATIC / PAGES
# ==========================================================================
async def page_index(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "index.html"))


async def page_field(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "field.html"))


async def sw_js(req: Request) -> Response:
    return FileResponse(os.path.join(FRONTEND, "sw.js"),
                        media_type="application/javascript")


async def not_found(req: Request, exc) -> Response:
    if req.url.path.startswith("/api"):
        return err("no such endpoint — see GET /api", 404)
    return FileResponse(os.path.join(FRONTEND, "index.html"))


async def server_error(req: Request, exc) -> Response:
    return err("internal error", 500, str(exc))


routes = [
    Route("/", page_index),
    Route("/field", page_field),
    Route("/track", page_track),
    Route("/sw.js", sw_js),

    Route("/api", api_index),
    Route("/api/health", health),
    Route("/api/metrics", metrics),
    Route("/api/languages", languages),

    Route("/api/network", network),
    Route("/api/network/summary", network_summary),
    Route("/api/nodes", nodes),
    Route("/api/corridors", corridors),
    Route("/api/weather", weather),
    Route("/api/segment/{road_id:path}", segment_detail),

    Route("/api/route", route),
    Route("/api/route/compare", route_compare),
    Route("/api/departure", departure),
    Route("/api/criticality", criticality),
    Route("/api/accessibility", accessibility),
    Route("/api/heatmap", heatmap),

    Route("/api/alerts", get_alerts),
    Route("/api/incidents", get_incidents, methods=["GET"]),
    Route("/api/incidents", post_incident, methods=["POST"]),

    Route("/api/cargo", cargo),
    Route("/api/whatif", whatif),

    Route("/api/weather/status", weather_status),
    Route("/api/weather/refresh", weather_refresh, methods=["POST"]),

    Route("/api/track", track_get, methods=["GET"]),
    Route("/api/track", track_post, methods=["POST"]),
    Route("/api/track/forget", track_delete, methods=["POST"]),

    Route("/api/fleet", fleet_snapshot),
    Route("/api/fleet/stream", fleet_stream),
    Route("/api/fleet/reset", fleet_reset, methods=["POST"]),
    WebSocketRoute("/ws/fleet", ws_fleet),
]

# ==========================================================================
# STARTUP — live weather refresh loop
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
async def lifespan(app):
    """Starlette 1.x lifespan — replaces the removed on_startup hook."""
    mode = "LIVE (Open-Meteo)" if config.LIVE_WEATHER else "SIMULATED"
    print(f"[sentinel] weather source: {mode}")
    print(f"[sentinel] GPS tracking:   "
          f"{'enabled' if config.LIVE_GPS else 'disabled'}"
          f"  -> open /track on a phone to stream real positions")
    task = asyncio.create_task(_weather_loop()) if config.LIVE_WEATHER else None
    try:
        yield
    finally:
        if task:
            task.cancel()


app = Starlette(
    debug=False,
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])],
    exception_handlers={404: not_found, 500: server_error},
)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
app.mount("/uploads", StaticFiles(directory=db.UPLOAD_DIR), name="uploads")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, log_level="info")
