"""
Live fleet simulation — trucks moving along REAL route geometry, streamed over
a WebSocket.

This is not decoration. It demonstrates the thing a logistics control room
actually does: watch vehicles against a risk surface and react before the
vehicle reaches the bad segment. Each tick we
  * advance every vehicle along its planned polyline at a speed modulated by the
    risk of the segment it is currently on,
  * detect when a vehicle ENTERS a segment predicted risky/blocked (geofence
    breach) and raise an alert,
  * look AHEAD on the remaining route and warn before it gets there,
  * recompute ETA from the live predicted delay,
  * flag critical-cargo vehicles whose viability window is about to be missed.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from . import config, inference, routing
from .providers import gps as gps_mod
from .geography import NODES, haversine_km

TICK_SECONDS = 2.0
SIM_SPEEDUP = 900.0          # 2 s of wall clock = 30 min of convoy movement


# --------------------------------------------------------------------------
@dataclass
class Vehicle:
    vid: str
    label: str
    cargo_type: str
    cargo_label: str
    profile: str
    origin: str
    dest: str
    geometry: List[List[float]]          # [[lat, lon], ...]
    cum_km: List[float]                  # cumulative distance at each vertex
    seg_road_ids: List[str]              # road_id for each leg
    total_km: float
    travelled_km: float = 0.0
    status: str = "en_route"             # en_route | halted | arrived
    breached: Set[str] = field(default_factory=set)
    warned_ahead: Set[str] = field(default_factory=set)
    started_at: str = ""
    eta_hours: float = 0.0
    max_transit_hours: float = 240.0
    elapsed_hours: float = 0.0

    # ---- geometry helpers ------------------------------------------------
    def position(self) -> List[float]:
        d = min(self.travelled_km, self.total_km)
        for i in range(len(self.cum_km) - 1):
            if d <= self.cum_km[i + 1]:
                span = self.cum_km[i + 1] - self.cum_km[i]
                f = 0.0 if span <= 0 else (d - self.cum_km[i]) / span
                a, b = self.geometry[i], self.geometry[i + 1]
                return [round(a[0] + (b[0] - a[0]) * f, 6),
                        round(a[1] + (b[1] - a[1]) * f, 6)]
        return list(self.geometry[-1])

    def current_leg(self) -> int:
        d = min(self.travelled_km, self.total_km)
        for i in range(len(self.cum_km) - 1):
            if d <= self.cum_km[i + 1]:
                return i
        return max(0, len(self.seg_road_ids) - 1)

    def heading(self) -> float:
        i = self.current_leg()
        a, b = self.geometry[i], self.geometry[min(i + 1, len(self.geometry) - 1)]
        dy, dx = b[0] - a[0], b[1] - a[1]
        return round((math.degrees(math.atan2(dx, dy)) + 360) % 360, 1)


# --------------------------------------------------------------------------
DEMO_SHIPMENTS = [
    ("NL-01-AC-4412", "GUWAHATI", "AIZAWL", "vaccine", "emergency"),
    ("AS-01-BC-7788", "SILIGURI", "GUWAHATI", "general", "fastest"),
    ("MN-01-DD-1199", "DIMAPUR", "IMPHAL", "oxygen", "emergency"),
    ("SK-01-JA-2031", "SILIGURI", "GANGTOK", "blood", "emergency"),
    ("TR-01-AB-5560", "GUWAHATI", "AGARTALA", "medicine", "balanced"),
    ("AR-01-KA-3344", "TEZPUR", "TAWANG", "relief", "safest"),
    ("AS-02-CC-9021", "GUWAHATI", "DIBRUGARH", "perishable", "fastest"),
    ("ML-01-EE-6677", "GUWAHATI", "TURA", "general", "balanced"),
]


class FleetSimulator:
    def __init__(self) -> None:
        self.vehicles: Dict[str, Vehicle] = {}
        self.alerts: List[dict] = []
        self.subscribers: Set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self.tick_count = 0
        self.sim_clock: Optional[datetime] = None
        self._gps_seen: Set[str] = set()

    # ---------------------------------------------------------------
    def bootstrap(self) -> None:
        """Plan each demo shipment once, on the live network."""
        if self.vehicles or not config.SIM_FLEET:
            return
        state = inference.network_state()
        self.sim_clock = datetime.fromisoformat(state["ts"])
        for plate, o, d, cargo, profile in DEMO_SHIPMENTS:
            try:
                res = routing.plan(o, d, profile, k=1)
                if not res.get("routes"):
                    continue
                r = res["routes"][0]
                geo = r["geometry"]
                cum = [0.0]
                for i in range(len(geo) - 1):
                    cum.append(cum[-1] + haversine_km(geo[i][0], geo[i][1],
                                                      geo[i + 1][0], geo[i + 1][1]))
                spec = routing.CARGO.get(cargo, routing.CARGO["general"])
                v = Vehicle(
                    vid=plate, label=f"{NODES[o].name} → {NODES[d].name}",
                    cargo_type=cargo, cargo_label=spec["label"],
                    profile=profile, origin=o, dest=d,
                    geometry=geo, cum_km=cum,
                    seg_road_ids=[s["road_id"] for s in r["segments"]],
                    total_km=max(cum[-1], 0.1),
                    travelled_km=random.uniform(0.0, 0.35) * max(cum[-1], 0.1),
                    started_at=state["ts"],
                    eta_hours=r["eta_hours"],
                    max_transit_hours=spec["max_hours"],
                )
                self.vehicles[plate] = v
            except Exception:
                continue

    # ---------------------------------------------------------------
    def _leg_speed(self, v: Vehicle, state: dict) -> float:
        """km/h on the current leg, degraded by its predicted risk."""
        i = v.current_leg()
        if i >= len(v.seg_road_ids):
            return 0.0
        rid = v.seg_road_ids[i]
        seg = state["segments"].get(rid)
        if not seg:
            return 30.0
        base = seg["length_km"] / max(seg["freeflow_hours"], 0.01)
        if seg["risk_label"] == 2:
            return 0.0                      # stopped at the blockage
        return base * (1.0 - 0.55 * seg["severity"])

    def step(self) -> dict:
        state = inference.network_state()
        self.tick_count += 1
        sim_dt_hours = (TICK_SECONDS * SIM_SPEEDUP) / 3600.0
        if self.sim_clock:
            self.sim_clock += timedelta(hours=sim_dt_hours)

        new_alerts: List[dict] = []
        payload_vehicles: List[dict] = []

        for v in self.vehicles.values():
            if v.status == "arrived":
                payload_vehicles.append(self._vehicle_payload(v, state, None))
                continue

            leg = v.current_leg()
            rid = v.seg_road_ids[leg] if leg < len(v.seg_road_ids) else None
            seg = state["segments"].get(rid) if rid else None

            speed = self._leg_speed(v, state)
            v.status = "halted" if speed <= 0.01 else "en_route"
            v.travelled_km += speed * sim_dt_hours
            v.elapsed_hours += sim_dt_hours

            if v.travelled_km >= v.total_km:
                v.travelled_km = v.total_km
                v.status = "arrived"

            # ---- geofence breach: entered a risky/blocked segment ---------
            if seg and seg["risk_label"] > 0 and rid not in v.breached:
                v.breached.add(rid)
                new_alerts.append({
                    "alert_id": f"GEO-{uuid.uuid4().hex[:8]}",
                    "kind": "geofence_breach",
                    "severity": "critical" if seg["risk_label"] == 2 else "warning",
                    "vehicle": v.vid, "cargo": v.cargo_label,
                    "road_id": rid, "corridor": seg["corridor"],
                    "segment_name": seg["name"],
                    "risk_status": seg["risk_status"],
                    "severity_score": seg["severity"],
                    "position": v.position(),
                    "at": (self.sim_clock or datetime.now()).isoformat(
                        timespec="seconds"),
                    "message": (
                        f"{v.vid} has ENTERED a {seg['risk_status'].upper()} "
                        f"segment: {seg['name']} ({seg['corridor']}). "
                        f"Carrying {v.cargo_label}."),
                })

            # ---- look-ahead warning: bad segment coming up ---------------
            ahead_km = 0.0
            for j in range(leg + 1, len(v.seg_road_ids)):
                s2 = state["segments"].get(v.seg_road_ids[j])
                if not s2:
                    continue
                ahead_km += s2["length_km"]
                if ahead_km > 160:
                    break
                if s2["risk_label"] == 2 and v.seg_road_ids[j] not in v.warned_ahead:
                    v.warned_ahead.add(v.seg_road_ids[j])
                    new_alerts.append({
                        "alert_id": f"AHD-{uuid.uuid4().hex[:8]}",
                        "kind": "look_ahead",
                        "severity": "warning",
                        "vehicle": v.vid, "cargo": v.cargo_label,
                        "road_id": v.seg_road_ids[j], "corridor": s2["corridor"],
                        "segment_name": s2["name"],
                        "risk_status": s2["risk_status"],
                        "severity_score": s2["severity"],
                        "position": v.position(),
                        "at": (self.sim_clock or datetime.now()).isoformat(
                            timespec="seconds"),
                        "message": (
                            f"{v.vid}: BLOCKED segment {s2['name']} "
                            f"({s2['corridor']}) is ~{ahead_km:.0f} km ahead. "
                            f"Re-route now, before the next junction."),
                    })
                    break

            # ---- cargo viability breach ---------------------------------
            remaining_frac = 1.0 - (v.travelled_km / v.total_km)
            projected = v.elapsed_hours + v.eta_hours * remaining_frac
            if (projected > v.max_transit_hours and v.status != "arrived"
                    and f"cargo:{v.vid}" not in v.warned_ahead):
                v.warned_ahead.add(f"cargo:{v.vid}")
                new_alerts.append({
                    "alert_id": f"CGO-{uuid.uuid4().hex[:8]}",
                    "kind": "cargo_viability",
                    "severity": "critical",
                    "vehicle": v.vid, "cargo": v.cargo_label,
                    "position": v.position(),
                    "at": (self.sim_clock or datetime.now()).isoformat(
                        timespec="seconds"),
                    "message": (
                        f"{v.vid}: projected transit {projected:.1f} h now exceeds "
                        f"the {v.max_transit_hours:.0f} h viability window for "
                        f"{v.cargo_label}. Escalate to air lift or divert to the "
                        f"nearest cold-chain node."),
                })

            payload_vehicles.append(self._vehicle_payload(v, state, seg))

        # ---- REAL GPS vehicles ------------------------------------------
        live = {"vehicles": [], "count": 0, "active": 0, "in_risk_zone": 0}
        if config.LIVE_GPS:
            try:
                live = gps_mod.live_fleet(include_stale=False)
                for v in live["vehicles"]:
                    aid = f"gpsbreach:{v['id']}:{v.get('road_id')}"
                    if v["in_risk_zone"] and aid not in self._gps_seen:
                        self._gps_seen.add(aid)
                        new_alerts.append({
                            "alert_id": f"GPS-{uuid.uuid4().hex[:8]}",
                            "kind": "geofence_breach", "live_gps": True,
                            "severity": "critical" if v["risk_status"] == "blocked"
                                        else "warning",
                            "vehicle": v["id"], "cargo": v.get("cargo_type", ""),
                            "road_id": v.get("road_id"),
                            "risk_status": v.get("risk_status"),
                            "severity_score": v.get("severity"),
                            "position": v["position"],
                            "at": v["last_seen"],
                            "message": (f"LIVE GPS — {v['id']} is on a "
                                        f"{str(v.get('risk_status')).upper()} "
                                        f"segment."),
                        })
            except Exception as e:
                print("[fleet] gps merge error:", e)

        self.alerts = (new_alerts + self.alerts)[:60]
        return {
            "type": "fleet_tick",
            "tick": self.tick_count,
            "sim_time": (self.sim_clock or datetime.now()).isoformat(
                timespec="seconds"),
            "network_ts": state["ts"],
            "vehicles": payload_vehicles,
            "live_gps": live["vehicles"],
            "new_alerts": new_alerts,
            "summary": {
                "live_gps_total": live["count"],
                "live_gps_active": live["active"],
                "live_gps_in_risk": live["in_risk_zone"],
                "total": len(self.vehicles),
                "en_route": sum(1 for v in self.vehicles.values()
                                if v.status == "en_route"),
                "halted": sum(1 for v in self.vehicles.values()
                              if v.status == "halted"),
                "arrived": sum(1 for v in self.vehicles.values()
                               if v.status == "arrived"),
                "in_risk_zone": sum(
                    1 for v in self.vehicles.values()
                    if v.status != "arrived"
                    and v.current_leg() < len(v.seg_road_ids)
                    and state["segments"].get(
                        v.seg_road_ids[v.current_leg()], {}).get("risk_label", 0) > 0),
            },
        }

    def _vehicle_payload(self, v: Vehicle, state: dict, seg: Optional[dict]) -> dict:
        leg = v.current_leg()
        rid = v.seg_road_ids[leg] if leg < len(v.seg_road_ids) else None
        seg = seg or (state["segments"].get(rid) if rid else None)
        remaining = max(0.0, v.total_km - v.travelled_km)
        remaining_frac = remaining / v.total_km if v.total_km else 0.0
        return {
            "id": v.vid, "label": v.label,
            "cargo_type": v.cargo_type, "cargo_label": v.cargo_label,
            "profile": v.profile,
            "position": v.position(),
            "heading": v.heading(),
            "status": v.status,
            "progress_pct": round(100.0 * v.travelled_km / v.total_km, 1),
            "travelled_km": round(v.travelled_km, 1),
            "remaining_km": round(remaining, 1),
            "total_km": round(v.total_km, 1),
            "elapsed_hours": round(v.elapsed_hours, 2),
            "eta_remaining_hours": round(v.eta_hours * remaining_frac, 2),
            "max_transit_hours": v.max_transit_hours,
            "current_segment": None if not seg else {
                "road_id": seg["road_id"], "name": seg["name"],
                "corridor": seg["corridor"], "risk_status": seg["risk_status"],
                "severity": seg["severity"],
            },
            "in_risk_zone": bool(seg and seg["risk_label"] > 0),
            "geometry": v.geometry,
            "breach_count": len(v.breached),
        }

    # ---------------------------------------------------------------
    async def _run(self) -> None:
        self.bootstrap()
        while True:
            try:
                frame = await asyncio.to_thread(self.step)
                dead = []
                for q in list(self.subscribers):
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        dead.append(q)
                for q in dead:
                    self.subscribers.discard(q)
            except Exception as e:
                print("[fleet] tick error:", e)
            await asyncio.sleep(TICK_SECONDS)

    def ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def subscribe(self) -> asyncio.Queue:
        self.ensure_running()
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def snapshot(self) -> dict:
        self.bootstrap()
        state = inference.network_state()
        return {
            "type": "fleet_snapshot",
            "sim_time": (self.sim_clock or datetime.now()).isoformat(
                timespec="seconds"),
            "vehicles": [self._vehicle_payload(v, state, None)
                         for v in self.vehicles.values()],
            "live_gps": (gps_mod.live_fleet(include_stale=False)["vehicles"]
                         if config.LIVE_GPS else []),
            "alerts": self.alerts[:20],
        }

    def reset(self) -> None:
        self.vehicles.clear()
        self.alerts.clear()
        self.tick_count = 0
        self.bootstrap()


FLEET = FleetSimulator()
