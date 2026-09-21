"""
Live inference: turn DB state + models into a scored network, fast.

The whole dashboard, every routing query and the what-if simulator all read
from ONE function -- `network_state()` -- which batch-scores all 95 segments in
a single vectorised predict call and caches the result. That keeps the UI
responsive and keeps one definition of "current risk" across the product.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from . import config, db
from .features import (FEATURE_COLUMNS, MONSOON_MONTHS, RISK_LABELS,
                       build_feature_dict)

_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}
_METRICS: Optional[dict] = None

# in-process cache: key -> (built_at, payload)
_CACHE: Dict[str, dict] = {}
_CACHE_ORDER: List[str] = []
_CACHE_MAX = 64

NOW_DEFAULT = "2026-09-11T15:00:00"

# Weight of the "risky" class when collapsing the 3-class distribution into a
# single graded severity. A risky segment costs real hours; a blocked one costs
# the trip. 0.40 is calibrated against the observed delay ratio between the two.
SEVERITY_RISKY = 0.40


# --------------------------------------------------------------------------
def models() -> Dict[str, object]:
    global _MODELS
    if _MODELS:
        return _MODELS
    with _LOCK:
        if _MODELS:
            return _MODELS

        use_nn = config.MODEL_BACKEND in ("nn", "neural", "deep")
        loaded_nn = False

        if use_nn:
            try:
                from .nn_wrappers import (NeuralRiskEnsemble, NeuralDelayEnsemble,
                                          NeuralRouteEnsemble)
                risk_ens = NeuralRiskEnsemble(db.ARTIFACT_DIR)
                delay_ens = NeuralDelayEnsemble(risk_ens, db.ARTIFACT_DIR)
                route_ens = NeuralRouteEnsemble(db.ARTIFACT_DIR)
                _MODELS = {
                    "risk": risk_ens,
                    "delay": delay_ens,
                    "route_delay": route_ens,
                    "backend": "neural_core_onnx",
                }
                loaded_nn = True
            except Exception as e:
                print(f"[inference] Neural core load failed ({e}), falling back to tree ensemble")

        if not loaded_nn:
            _MODELS = {
                "risk": joblib.load(os.path.join(db.ARTIFACT_DIR, "risk_clf.joblib")),
                "delay": joblib.load(os.path.join(db.ARTIFACT_DIR, "delay_reg.joblib")),
                "route_delay": joblib.load(
                    os.path.join(db.ARTIFACT_DIR, "route_delay_reg.joblib")),
                "backend": "tree_ensemble",
            }
    return _MODELS


def metrics() -> dict:
    global _METRICS
    if _METRICS is None:
        p = os.path.join(db.ARTIFACT_DIR, "metrics.json")
        with open(p) as f:
            _METRICS = json.load(f)
    return _METRICS


# --------------------------------------------------------------------------
# Static caches loaded once
# --------------------------------------------------------------------------
_ROADS: Optional[List[dict]] = None
_DISTRICTS: Optional[Dict[str, dict]] = None


def roads() -> List[dict]:
    global _ROADS
    if _ROADS is None:
        _ROADS = db.query("SELECT * FROM roads")
    return _ROADS


def roads_by_id() -> Dict[str, dict]:
    return {r["road_id"]: r for r in roads()}


def districts() -> Dict[str, dict]:
    global _DISTRICTS
    if _DISTRICTS is None:
        _DISTRICTS = {d["district"]: d for d in db.query("SELECT * FROM districts")}
    return _DISTRICTS


def now_ts() -> str:
    """
    The system clock.

    In LIVE-WEATHER mode this is the newest observed (non-forecast) weather
    row, so the whole app moves with real time. Otherwise it is the end of the
    simulated observation history. Everything downstream — routing, alerts,
    the departure sweep — reads this one function, so switching between live
    and simulated time never leaves half the app in the past.
    """
    if config.LIVE_WEATHER:
        row = db.query_one(
            "SELECT MAX(ts) t FROM weather WHERE is_forecast = 0")
        if row and row["t"]:
            return row["t"]
    row = db.query_one("SELECT MAX(ts) t FROM observations")
    return row["t"] if row and row["t"] else NOW_DEFAULT


def available_weather_ts(after: str, hours: int) -> List[str]:
    end = (datetime.fromisoformat(after) + timedelta(hours=hours)).isoformat(
        timespec="seconds")
    rows = db.query(
        "SELECT DISTINCT ts FROM weather WHERE ts > ? AND ts <= ? ORDER BY ts",
        (after, end))
    return [r["ts"] for r in rows]


# --------------------------------------------------------------------------
def _nearest_weather_ts(ts: str) -> str:
    """Snap a requested time to the nearest available weather grid point."""
    r = db.query_one(
        "SELECT ts FROM weather WHERE ts <= ? ORDER BY ts DESC LIMIT 1", (ts,))
    if r:
        return r["ts"]
    r = db.query_one("SELECT MIN(ts) ts FROM weather")
    return r["ts"]


def weather_at(ts: str) -> Dict[str, dict]:
    ts = _nearest_weather_ts(ts)
    rows = db.query("SELECT * FROM weather WHERE ts = ?", (ts,))
    return {r["district"]: r for r in rows}


def incident_context(ts: str) -> Dict[str, dict]:
    """Rolling 90-day incident memory per segment as of `ts`."""
    t = datetime.fromisoformat(ts)
    lo = (t - timedelta(days=90)).isoformat(timespec="seconds")
    counts = {r["road_id"]: r["n"] for r in db.query(
        "SELECT road_id, COUNT(*) n FROM incidents "
        "WHERE reported_at BETWEEN ? AND ? GROUP BY road_id", (lo, ts))}
    last = {r["road_id"]: r["m"] for r in db.query(
        "SELECT road_id, MAX(reported_at) m FROM incidents "
        "WHERE reported_at <= ? GROUP BY road_id", (ts,))}
    out: Dict[str, dict] = {}
    for rid in roads_by_id():
        ds = 999
        if rid in last and last[rid]:
            ds = max(0, (t - datetime.fromisoformat(last[rid])).days)
        out[rid] = {"hist_incidents_90d": counts.get(rid, 0),
                    "days_since_last_incident": min(ds, 999)}
    return out


def active_disruptions(ts: str) -> Dict[str, dict]:
    rows = db.query(
        "SELECT * FROM disruptions WHERE start_ts <= ? AND end_ts >= ?", (ts, ts))
    return {r["corridor"]: r for r in rows}


# --------------------------------------------------------------------------
# What-if overrides
# --------------------------------------------------------------------------
def apply_overrides(wx: Dict[str, dict], ov: Optional[dict]) -> Dict[str, dict]:
    """
    Counterfactual weather injection for the simulator.
      rain_multiplier   : scale 24h/72h rainfall everywhere (or in `states`)
      rain_24h_set      : hard-set 24h rainfall (mm)
      api_multiplier    : scale soil saturation
      api_set           : hard-set soil saturation (mm)
      temperature_delta : shift temperature (snow scenarios)
      states            : restrict the scenario to these state codes
      condition_set     : force a weather condition string
    """
    if not ov:
        return wx
    dist = districts()
    states = set(ov.get("states") or [])
    out: Dict[str, dict] = {}
    for name, row in wx.items():
        r = dict(row)
        d = dist.get(name)
        if states and (not d or d["state"] not in states):
            out[name] = r
            continue
        if ov.get("rain_24h_set") is not None:
            r["rainfall_24h_mm"] = float(ov["rain_24h_set"])
            r["rainfall_72h_mm"] = max(r["rainfall_72h_mm"],
                                       float(ov["rain_24h_set"]) * 1.6)
        if ov.get("rain_multiplier") is not None:
            m = float(ov["rain_multiplier"])
            r["rainfall_24h_mm"] *= m
            r["rainfall_72h_mm"] *= m
        if ov.get("api_set") is not None:
            r["api_7d"] = float(ov["api_set"])
        if ov.get("api_multiplier") is not None:
            r["api_7d"] *= float(ov["api_multiplier"])
        if ov.get("temperature_delta") is not None:
            r["temperature_c"] += float(ov["temperature_delta"])
        if ov.get("condition_set"):
            r["weather_condition"] = ov["condition_set"]
        else:
            # keep the condition physically consistent with the new rainfall
            rr = r["rainfall_24h_mm"]
            if r["temperature_c"] <= 1.5 and rr > 1:
                r["weather_condition"] = "snow"
            elif rr >= 65:
                r["weather_condition"] = "storm"
            elif rr >= 25:
                r["weather_condition"] = "heavy_rain"
            elif rr >= 7:
                r["weather_condition"] = "rain"
            elif rr >= 1:
                r["weather_condition"] = "light_rain"
        out[name] = r
    return out


def _cache_key(ts: str, ov: Optional[dict]) -> str:
    return ts + "|" + (json.dumps(ov, sort_keys=True) if ov else "-")


# --------------------------------------------------------------------------
def network_state(ts: Optional[str] = None, overrides: Optional[dict] = None,
                  use_cache: bool = True) -> dict:
    """
    Score every road segment at time `ts`. Returns
      {"ts", "segments": {road_id: {...}}, "order": [road_id...], "summary": {...}}
    """
    ts = ts or now_ts()
    key = _cache_key(ts, overrides)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    t = datetime.fromisoformat(ts)
    wx = apply_overrides(weather_at(ts), overrides)
    ictx = incident_context(ts)
    disr = active_disruptions(ts)
    rds = roads()

    feats: List[dict] = []
    order: List[str] = []
    for r in rds:
        w = wx.get(r["district"])
        if w is None:
            continue
        ctx = {
            "month": t.month, "hour": t.hour,
            "is_night": 1 if (t.hour >= 19 or t.hour < 5) else 0,
            "hist_incidents_90d": ictx[r["road_id"]]["hist_incidents_90d"],
            "days_since_last_incident":
                ictx[r["road_id"]]["days_since_last_incident"],
            "disruption_flag": 1 if r["corridor"] in disr else 0,
        }
        feats.append(build_feature_dict(r, w, ctx))
        order.append(r["road_id"])

    X = np.array([[f[c] for c in FEATURE_COLUMNS] for f in feats], dtype=np.float64)
    M = models()
    proba = M["risk"].predict_proba(X)
    labels = np.argmax(proba, axis=1)
    delays = M["delay"].predict(X)

    by_id = roads_by_id()
    segs: Dict[str, dict] = {}
    for i, rid in enumerate(order):
        r = by_id[rid]
        p = proba[i]
        fail = float(min(1.0, p[1] + p[2])) if len(p) >= 3 else float(p[-1])
        # Failure probability saturates near 1.0 for any hill segment in the
        # monsoon, so it cannot separate "slow going" from "impassable".
        # SEVERITY is the graded, expected-impact score used for routing cost
        # and for the map colour ramp.
        severity = float(min(1.0, SEVERITY_RISKY * p[1] + 1.0 * p[2])) \
            if len(p) >= 3 else fail
        freeflow = r["length_km"] / r["freeflow_kmph"]
        segs[rid] = {
            "road_id": rid,
            "corridor": r["corridor"],
            "name": r["name"],
            "road_class": r["road_class"],
            "state": r["state"],
            "district": r["district"],
            "u": r["u"], "v": r["v"],
            "terrain": r["terrain"],
            "length_km": r["length_km"],
            "lanes": r["lanes"],
            "strategic": bool(r["strategic"]),
            "coords": [[r["start_lat"], r["start_lon"]], [r["end_lat"], r["end_lon"]]],
            "risk_label": int(labels[i]),
            "risk_status": RISK_LABELS[int(labels[i])],
            "risk_score": round(fail, 4),
            "severity": round(severity, 4),
            "proba": [round(float(q), 4) for q in p],
            "delay_hours": round(float(max(0.0, delays[i])), 2),
            "freeflow_hours": round(freeflow, 3),
            "travel_hours": round(freeflow + float(max(0.0, delays[i])), 2),
            "disruption": r["corridor"] in disr,
            "disruption_kind": disr.get(r["corridor"], {}).get("kind"),
            "weather": {
                "condition": wx[r["district"]]["weather_condition"],
                "rainfall_24h_mm": round(wx[r["district"]]["rainfall_24h_mm"], 1),
                "api_7d": round(wx[r["district"]]["api_7d"], 1),
                "temperature_c": wx[r["district"]]["temperature_c"],
                "wind_kmph": wx[r["district"]]["wind_kmph"],
            },
            "features": feats[i],
            "notes": r["notes"],
        }

    n = len(segs) or 1
    summary = {
        "ts": ts,
        "segments_scored": len(segs),
        "safe": int((labels == 0).sum()),
        "risky": int((labels == 1).sum()),
        "blocked": int((labels == 2).sum()),
        "network_km": round(sum(s["length_km"] for s in segs.values()), 1),
        "km_blocked": round(sum(s["length_km"] for s in segs.values()
                                if s["risk_label"] == 2), 1),
        "mean_risk": round(float(np.mean([s["risk_score"] for s in segs.values()])), 4),
        "mean_severity": round(float(np.mean([s["severity"] for s in segs.values()])), 4),
        "active_disruptions": [
            {"corridor": k, "kind": v["kind"], "note": v["note"],
             "until": v["end_ts"]} for k, v in disr.items()],
        "is_monsoon": t.month in MONSOON_MONTHS,
        "counterfactual": bool(overrides),
    }
    payload = {"ts": ts, "segments": segs, "order": order, "summary": summary}

    if use_cache:
        _CACHE[key] = payload
        _CACHE_ORDER.append(key)
        while len(_CACHE_ORDER) > _CACHE_MAX:
            _CACHE.pop(_CACHE_ORDER.pop(0), None)
    return payload


def invalidate_cache() -> None:
    _CACHE.clear()
    _CACHE_ORDER.clear()
    global _ROADS
    _ROADS = None
