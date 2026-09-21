"""
REAL GPS TRACKING.

A phone (or any VTS/AIS unit that can POST JSON) streams positions to
`POST /api/track`. For each fix we:

  1. snap it to the nearest road segment in the network,
  2. read that segment's LIVE model prediction,
  3. raise a geofence-breach alert the moment the vehicle enters a segment the
     model calls risky or blocked,
  4. look ahead along the road it is on and warn about what is coming.

The point is that this is the same risk surface the control room is looking at —
a vehicle is not tracked on a separate map, it is tracked *against the model*.

Production note: this endpoint accepts any JSON with lat/lon, so a real VAHAN /
VTS feed or an AIS-140 device pushes into the same path with no code change.
The phone is just the cheapest possible device for a demo.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .. import db, inference
from ..geography import haversine_km

# A fix further than this from any road is treated as off-network rather than
# being force-snapped to something irrelevant.
MAX_SNAP_KM = 6.0
STALE_MINUTES = 15          # after this with no fix, a vehicle is "stale"

# A look-ahead warning must fire ONCE per hazard, not on every GPS fix. A driver
# who is warned every eight seconds stops reading warnings, which is worse than
# not warning at all.
_WARNED_AHEAD: Dict[str, str] = {}


# --------------------------------------------------------------------------
def point_to_segment_km(plat: float, plon: float,
                        alat: float, alon: float,
                        blat: float, blon: float) -> Tuple[float, float]:
    """
    Distance from a point to a segment, and the fractional position along it.
    Planar approximation in degrees with a cos(lat) correction — accurate to
    well under a metre at NER latitudes over segment-length distances.
    """
    latr = math.radians((alat + blat) / 2.0)
    kx = math.cos(latr)
    ax, ay = alon * kx, alat
    bx, by = blon * kx, blat
    px, py = plon * kx, plat
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return haversine_km(plat, plon, alat, alon), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    clat, clon = cy, cx / kx
    return haversine_km(plat, plon, clat, clon), t


def snap(lat: float, lon: float, state: dict) -> Optional[dict]:
    """Nearest road segment to a GPS fix, with its live prediction attached."""
    best, bestd, bestt = None, None, 0.0
    for rid, s in state["segments"].items():
        (a, b) = s["coords"]
        d, t = point_to_segment_km(lat, lon, a[0], a[1], b[0], b[1])
        if bestd is None or d < bestd:
            best, bestd, bestt = s, d, t
    if best is None or bestd > MAX_SNAP_KM:
        return None
    return {"segment": best, "distance_km": round(bestd, 3),
            "along": round(bestt, 3)}


# --------------------------------------------------------------------------
def look_ahead(state: dict, road_id: str, heading: Optional[float],
               max_km: float = 120.0) -> Optional[dict]:
    """
    Walk the corridor forward from the current segment and report the first
    risky/blocked segment within `max_km`.

    We follow the corridor rather than the shortest path because a driver on
    NH-306 is, physically, going to stay on NH-306 — that is the whole problem
    with NER roads.
    """
    cur = state["segments"].get(road_id)
    if not cur:
        return None
    corridor = cur["corridor"]
    peers = [s for s in state["segments"].values() if s["corridor"] == corridor]
    if not peers:
        return None

    # order the corridor's segments into a chain from the current one
    by_u: Dict[str, dict] = {s["u"]: s for s in peers}
    chain: List[dict] = []
    node = cur["v"]
    seen = {cur["road_id"]}
    dist = 0.0
    while node in by_u and dist < max_km:
        nxt = by_u[node]
        if nxt["road_id"] in seen:
            break
        seen.add(nxt["road_id"])
        dist += nxt["length_km"]
        chain.append({"seg": nxt, "km_ahead": round(dist, 1)})
        node = nxt["v"]

    for item in chain:
        s = item["seg"]
        if s["risk_label"] > 0:
            return {"road_id": s["road_id"], "name": s["name"],
                    "corridor": s["corridor"], "risk_status": s["risk_status"],
                    "severity": s["severity"], "km_ahead": item["km_ahead"],
                    "delay_hours": s["delay_hours"]}
    return None


# --------------------------------------------------------------------------
def ingest(payload: dict) -> dict:
    """Accept one GPS fix. Returns what the driver's device should display."""
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("lat and lon are required and must be numeric")
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("lat/lon out of range")

    vid = str(payload.get("vehicle_id") or "").strip()[:32]
    if not vid:
        vid = f"DEV-{uuid.uuid4().hex[:6].upper()}"

    now = datetime.now().isoformat(timespec="seconds")
    speed = payload.get("speed_kmph")
    speed = float(speed) if speed not in (None, "") else None
    heading = payload.get("heading")
    heading = float(heading) if heading not in (None, "") else None
    acc = payload.get("accuracy_m")
    acc = float(acc) if acc not in (None, "") else None

    state = inference.network_state()
    hit = snap(lat, lon, state)

    road_id = seg_name = corridor = risk_status = None
    severity = None
    risk_label = None
    snap_km = None
    ahead = None
    on_network = hit is not None

    if hit:
        s = hit["segment"]
        road_id = s["road_id"]
        seg_name = s["name"]
        corridor = s["corridor"]
        risk_status = s["risk_status"]
        risk_label = s["risk_label"]
        severity = s["severity"]
        snap_km = hit["distance_km"]
        ahead = look_ahead(state, road_id, heading)

    prev = db.query_one("SELECT * FROM live_vehicles WHERE vehicle_id = ?", (vid,))
    first_seen = prev["first_seen"] if prev else now
    points = (prev["points"] if prev else 0) + 1
    prev_road = prev["road_id"] if prev else None

    db.execute(
        "INSERT OR REPLACE INTO live_vehicles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (vid,
         payload.get("label") or (prev["label"] if prev else vid),
         payload.get("cargo_type") or (prev["cargo_type"] if prev else "general"),
         payload.get("driver") or (prev["driver"] if prev else ""),
         lat, lon, speed, heading, acc,
         road_id, snap_km, risk_label, risk_status, severity,
         first_seen, now, points,
         payload.get("dest") or (prev["dest"] if prev else None)))

    db.execute(
        "INSERT INTO track_points (vehicle_id, ts, lat, lon, speed_kmph, heading,"
        " accuracy_m, road_id, severity) VALUES (?,?,?,?,?,?,?,?,?)",
        (vid, now, lat, lon, speed, heading, acc, road_id, severity))

    # ---- alerts -----------------------------------------------------------
    alerts: List[dict] = []
    entered_new = road_id and road_id != prev_road
    if entered_new and risk_label and risk_label > 0:
        alerts.append({
            "alert_id": f"GPS-{uuid.uuid4().hex[:8]}",
            "kind": "geofence_breach",
            "severity": "critical" if risk_label == 2 else "warning",
            "vehicle": vid, "road_id": road_id,
            "corridor": corridor, "segment_name": seg_name,
            "risk_status": risk_status, "severity_score": severity,
            "position": [lat, lon], "at": now, "live_gps": True,
            "message": (f"{vid} has ENTERED a {str(risk_status).upper()} segment: "
                        f"{seg_name} ({corridor})."),
        })
    already = _WARNED_AHEAD.get(vid)
    if ahead and ahead["risk_status"] == "blocked" and already != ahead["road_id"]:
        _WARNED_AHEAD[vid] = ahead["road_id"]
        alerts.append({
            "alert_id": f"GPS-{uuid.uuid4().hex[:8]}",
            "kind": "look_ahead", "severity": "warning",
            "vehicle": vid, "road_id": ahead["road_id"],
            "corridor": ahead["corridor"], "segment_name": ahead["name"],
            "risk_status": ahead["risk_status"],
            "severity_score": ahead["severity"],
            "position": [lat, lon], "at": now, "live_gps": True,
            "message": (f"{vid}: BLOCKED segment {ahead['name']} "
                        f"({ahead['corridor']}) is ~{ahead['km_ahead']:.0f} km "
                        f"ahead. Re-route before the next junction."),
        })

    return {
        "status": "ok",
        "vehicle_id": vid,
        "received_at": now,
        "on_network": on_network,
        "position": [lat, lon],
        "snapped_to": None if not hit else {
            "road_id": road_id, "name": seg_name, "corridor": corridor,
            "distance_km": snap_km,
            "risk_status": risk_status, "severity": severity,
            "delay_hours": hit["segment"]["delay_hours"],
            "terrain": hit["segment"]["terrain"],
            "weather": hit["segment"]["weather"],
        },
        "look_ahead": ahead,
        "alerts": alerts,
        "advice": _advice(on_network, risk_status, ahead),
        "points_recorded": points,
    }


def _advice(on_network: bool, risk_status: Optional[str],
            ahead: Optional[dict]) -> str:
    if not on_network:
        return ("You are more than 6 km from any mapped highway — no road risk "
                "assessment available for this position.")
    if risk_status == "blocked":
        return ("You are on a segment the model predicts is BLOCKED. Stop at the "
                "next safe point and contact the control room before proceeding.")
    if ahead and ahead["risk_status"] == "blocked":
        return (f"Road ahead is predicted blocked in ~{ahead['km_ahead']:.0f} km. "
                f"Plan to divert at the next junction.")
    if risk_status == "risky":
        return ("Current segment is elevated risk. Reduce speed, avoid halting "
                "under cut slopes, and report any debris you see.")
    return "Current segment is clear. Continue and keep reporting."


# --------------------------------------------------------------------------
def live_fleet(include_stale: bool = True) -> dict:
    """Everything the control room needs about real, GPS-tracked vehicles."""
    rows = db.query("SELECT * FROM live_vehicles ORDER BY last_seen DESC")
    now = datetime.now()
    out: List[dict] = []
    for r in rows:
        try:
            age = (now - datetime.fromisoformat(r["last_seen"])).total_seconds() / 60.0
        except Exception:
            age = 9999
        stale = age > STALE_MINUTES
        if stale and not include_stale:
            continue
        trail = db.query(
            "SELECT lat, lon FROM track_points WHERE vehicle_id = ? "
            "ORDER BY ts DESC LIMIT 400", (r["vehicle_id"],))
        out.append({
            "id": r["vehicle_id"],
            "label": r["label"] or r["vehicle_id"],
            "cargo_type": r["cargo_type"],
            "driver": r["driver"],
            "position": [r["lat"], r["lon"]],
            "speed_kmph": r["speed_kmph"],
            "heading": r["heading"] or 0.0,
            "accuracy_m": r["accuracy_m"],
            "road_id": r["road_id"],
            "risk_status": r["risk_status"],
            "severity": r["severity"],
            "snap_distance_km": r["snap_distance_km"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "age_minutes": round(age, 1),
            "stale": stale,
            "points": r["points"],
            "in_risk_zone": bool(r["risk_label"] and r["risk_label"] > 0),
            "trail": [[p["lat"], p["lon"]] for p in reversed(trail)],
            "live_gps": True,
        })
    return {
        "vehicles": out,
        "count": len(out),
        "active": sum(1 for v in out if not v["stale"]),
        "in_risk_zone": sum(1 for v in out if v["in_risk_zone"] and not v["stale"]),
        "stale_after_minutes": STALE_MINUTES,
    }


def forget(vehicle_id: str) -> bool:
    _WARNED_AHEAD.pop(vehicle_id, None)
    if not db.query_one("SELECT 1 FROM live_vehicles WHERE vehicle_id = ?",
                        (vehicle_id,)):
        return False
    db.execute("DELETE FROM live_vehicles WHERE vehicle_id = ?", (vehicle_id,))
    db.execute("DELETE FROM track_points WHERE vehicle_id = ?", (vehicle_id,))
    return True
