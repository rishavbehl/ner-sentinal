"""
Routing + network-intelligence engine.

Four things here that ordinary "shortest path on a map" demos do not do:

 1. RISK-AWARE COST with CONVEX TAIL AVERSION. A relief convoy does not want
    the route with the best *average* outcome, it wants the route whose *worst*
    outcome is survivable. We add a cubic term in segment risk, so the
    optimiser refuses to trade a small time saving for one terrible segment.

 2. GENUINELY DIVERSE ALTERNATES. `shortest_simple_paths` returns k routes that
    differ by one junction and are useless as alternates. We iteratively
    penalise edges already used, which produces structurally different
    corridors -- the thing a control room actually needs.

 3. CRITICALITY / SINGLE-POINT-OF-FAILURE ANALYSIS. For every segment we delete
    it and re-solve the network from the regional supply hub. The output is
    "if this 38 km fails, 6 districts lose their only road link" -- which is
    the real story of NER logistics (NH-10 to Sikkim, Sela Pass, the Siliguri
    Corridor) and is invisible to route-level tools.

 4. DISTRICT ACCESSIBILITY INDEX. The problem statement asks for accessibility
    intelligence, not just routing. We publish a composite, auditable index
    (reach + reliability + redundancy + terrain burden) per district so an
    official can rank where to spend money, not just where to send a truck.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from . import db, inference
from .explain import explain_route_delay
from .features import MONSOON_MONTHS, route_vector
from .geography import NODES, STATES

SUPPLY_HUB = "GUWAHATI"          # NER's de-facto distribution centre
GATEWAY = "SILIGURI"             # the corridor everything enters through


# --------------------------------------------------------------------------
# Cost profiles
# --------------------------------------------------------------------------
@dataclass
class Profile:
    key: str
    label: str
    description: str
    time_w: float
    risk_w: float
    tail_w: float
    blocked_penalty: float
    rural_penalty: float = 0.0
    night_hill_penalty: float = 0.0


PROFILES: Dict[str, Profile] = {
    "fastest": Profile(
        "fastest", "Fastest",
        "Minimises predicted arrival time. Accepts risk.",
        time_w=1.0, risk_w=0.20, tail_w=0.0, blocked_penalty=14.0),
    "balanced": Profile(
        "balanced", "Balanced",
        "Default commercial haulage: time with meaningful risk weighting.",
        time_w=1.0, risk_w=1.30, tail_w=1.2, blocked_penalty=45.0,
        rural_penalty=0.3),
    "safest": Profile(
        "safest", "Safest",
        "Minimises exposure. Will accept a long detour to avoid bad slopes.",
        time_w=0.55, risk_w=3.6, tail_w=4.5, blocked_penalty=260.0,
        rural_penalty=1.0, night_hill_penalty=0.6),
    "emergency": Profile(
        "emergency", "Emergency / Critical Supply",
        "Medicines, blood, oxygen, disaster relief. Hard-avoids blocked "
        "segments, strongly penalises tail risk, prefers national highways "
        "with faster clearance response, and avoids night hill movement.",
        time_w=1.0, risk_w=4.8, tail_w=9.0, blocked_penalty=1.0e6,
        rural_penalty=2.0, night_hill_penalty=1.2),
}


def edge_cost(seg: dict, p: Profile, is_night: bool) -> float:
    t = seg["travel_hours"]
    risk = seg["severity"]      # graded expected impact, not saturated P(fail)
    c = p.time_w * t
    # risk scaled by exposure time: a risky 8 h segment is worse than a risky 1 h one
    c += p.risk_w * risk * (t + 0.35)
    # convex tail aversion — refuses "one terrible segment to save 20 minutes"
    c += p.tail_w * (risk ** 3) * 6.0
    if seg["risk_label"] == 2:
        c += p.blocked_penalty
    if seg["road_class"] == "RR":
        c += p.rural_penalty
    if is_night and seg["terrain"] in ("hilly", "high_pass"):
        c += p.night_hill_penalty
    return max(c, 1e-4)


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def build_graph(state: dict, profile: Profile, hard_avoid_blocked: bool = False,
                risk_ceiling: float = 1.01) -> nx.Graph:
    """
    `hard_avoid_blocked` removes segments predicted BLOCKED outright.
    `risk_ceiling` additionally removes segments whose failure probability is
    above the ceiling — because a segment the model thinks is 95 % likely to
    fail is not a road you send blood down, even if argmax happened to land on
    'risky' rather than 'blocked'.
    """
    t = datetime.fromisoformat(state["ts"])
    is_night = t.hour >= 19 or t.hour < 5
    G = nx.Graph()
    for nid, n in NODES.items():
        G.add_node(nid, name=n.name, state=n.state, lat=n.lat, lon=n.lon,
                   elev=n.elev_m, kind=n.kind)
    for rid, s in state["segments"].items():
        if hard_avoid_blocked and s["risk_label"] == 2:
            continue
        if s["severity"] >= risk_ceiling:
            continue
        c = edge_cost(s, profile, is_night)
        G.add_edge(s["u"], s["v"], road_id=rid, cost=c,
                   km=s["length_km"], hours=s["travel_hours"],
                   risk=s["severity"], label=s["risk_label"])
    return G


def distance_graph(state: dict) -> nx.Graph:
    """Pure distance graph — used for criticality and accessibility topology."""
    G = nx.Graph()
    for nid in NODES:
        G.add_node(nid)
    for rid, s in state["segments"].items():
        G.add_edge(s["u"], s["v"], road_id=rid, km=s["length_km"],
                   hours=s["travel_hours"], risk=s["severity"],
                   label=s["risk_label"])
    return G


# --------------------------------------------------------------------------
# Diverse alternates
# --------------------------------------------------------------------------
def k_diverse_paths(G: nx.Graph, o: str, d: str, k: int = 3,
                    penalty: float = 1.6, max_tries: int = 14
                    ) -> List[List[str]]:
    paths: List[List[str]] = []
    used: Counter = Counter()
    for _ in range(max_tries):
        def w(u, v, data):
            key = tuple(sorted((u, v)))
            return data["cost"] * (1.0 + penalty * used[key])
        try:
            p = nx.shortest_path(G, o, d, weight=w)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        if p not in paths:
            paths.append(p)
            if len(paths) >= k:
                break
        for i in range(len(p) - 1):
            used[tuple(sorted((p[i], p[i + 1])))] += 1
    return paths


# --------------------------------------------------------------------------
# Route assembly + scoring
# --------------------------------------------------------------------------
_INC_COUNTS: Optional[Dict[str, int]] = None


def _incident_counts() -> Dict[str, int]:
    """Cached: this is read on every route scoring, including 25-slot sweeps."""
    global _INC_COUNTS
    if _INC_COUNTS is None:
        _INC_COUNTS = {r["road_id"]: r["n"] for r in db.query(
            "SELECT road_id, COUNT(*) n FROM incidents GROUP BY road_id")}
    return _INC_COUNTS


def reset_caches() -> None:
    global _INC_COUNTS
    _INC_COUNTS = None


def _route_features(segs: List[dict], path: List[str], ts: str) -> Dict[str, float]:
    t = datetime.fromisoformat(ts)
    length = sum(s["length_km"] for s in segs)
    freeflow = sum(s["freeflow_hours"] for s in segs)
    risks = [s["severity"] for s in segs]
    ascent = 0.0
    for i in range(len(path) - 1):
        ascent += max(0, NODES[path[i + 1]].elev_m - NODES[path[i]].elev_m)
    states = [NODES[p].state for p in path]
    crossings = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
    by_id = inference.roads_by_id()
    inc_counts = _incident_counts()
    return {
        "route_length_km": round(length, 2),
        "n_segments": float(len(segs)),
        "freeflow_hours": round(freeflow, 3),
        "avg_risk_score": round(float(np.mean(risks)), 4),
        "max_risk_score": round(float(np.max(risks)), 4),
        "n_risky_segments": float(sum(1 for s in segs if s["risk_label"] == 1)),
        "n_blocked_segments": float(sum(1 for s in segs if s["risk_label"] == 2)),
        "frac_hilly": round(sum(1 for s in segs
                                if s["terrain"] in ("hilly", "high_pass")) / len(segs), 4),
        "frac_nh": round(sum(1 for s in segs if s["road_class"] == "NH") / len(segs), 4),
        "total_ascent_m": float(ascent),
        "max_rainfall_24h": round(max(s["weather"]["rainfall_24h_mm"] for s in segs), 2),
        "mean_api_7d": round(float(np.mean([s["weather"]["api_7d"] for s in segs])), 2),
        "n_incidents_on_route": float(sum(inc_counts.get(s["road_id"], 0) for s in segs)),
        "n_bridges": float(sum(by_id[s["road_id"]]["bridge_count"] for s in segs)),
        "n_state_crossings": float(crossings),
        "departs_at_night": 1.0 if (t.hour >= 19 or t.hour < 5) else 0.0,
        "is_monsoon": 1.0 if t.month in MONSOON_MONTHS else 0.0,
        "disruption_segments": float(sum(1 for s in segs if s["disruption"])),
    }


def score_route(state: dict, path: List[str], profile: Profile,
                explain: bool = False) -> dict:
    G = distance_graph(state)
    segs: List[dict] = []
    for i in range(len(path) - 1):
        rid = G[path[i]][path[i + 1]]["road_id"]
        segs.append(state["segments"][rid])
    if not segs:
        return {}

    rf = _route_features(segs, path, state["ts"])
    M = inference.models()
    rx = route_vector(rf)
    ml_delay = float(max(0.0, M["route_delay"].predict(rx.reshape(1, -1))[0]))
    freeflow = rf["freeflow_hours"]
    eta_hours = freeflow + ml_delay

    # Composite route score (0 = perfect). Published so it is auditable.
    risk_component = 0.55 * rf["avg_risk_score"] + 0.45 * rf["max_risk_score"]
    norm_delay = min(ml_delay / 24.0, 1.0)
    norm_dist = min(rf["route_length_km"] / 900.0, 1.0)
    composite = round(100.0 * (0.45 * risk_component + 0.35 * norm_delay +
                               0.20 * norm_dist), 2)

    worst = max(segs, key=lambda s: s["risk_score"])
    blocked = [s for s in segs if s["risk_label"] == 2]
    status = "blocked" if blocked else ("risky" if rf["max_risk_score"] >= 0.5
                                        else "open")

    out = {
        "path_nodes": path,
        "path_names": [NODES[p].name for p in path],
        "states_traversed": sorted({NODES[p].state for p in path}),
        "segments": [{
            "road_id": s["road_id"], "corridor": s["corridor"], "name": s["name"],
            "length_km": s["length_km"], "risk_status": s["risk_status"],
            "risk_score": s["risk_score"], "delay_hours": s["delay_hours"],
            "travel_hours": s["travel_hours"], "terrain": s["terrain"],
            "coords": s["coords"], "disruption": s["disruption"],
        } for s in segs],
        "geometry": [s["coords"][0] for s in segs] + [segs[-1]["coords"][1]],
        "distance_km": rf["route_length_km"],
        "freeflow_hours": round(freeflow, 2),
        "predicted_delay_hours": round(ml_delay, 2),
        "eta_hours": round(eta_hours, 2),
        "eta_text": _hours_text(eta_hours),
        "avg_risk": rf["avg_risk_score"],
        "max_risk": rf["max_risk_score"],
        "n_risky": int(rf["n_risky_segments"]),
        "n_blocked": int(rf["n_blocked_segments"]),
        "n_state_crossings": int(rf["n_state_crossings"]),
        "frac_hilly": rf["frac_hilly"],
        "status": status,
        "composite_score": composite,
        "profile": profile.key,
        "worst_segment": {
            "road_id": worst["road_id"], "name": worst["name"],
            "corridor": worst["corridor"], "risk_score": worst["risk_score"],
            "risk_status": worst["risk_status"],
        },
        "route_features": rf,
        "sum_of_segment_delays": round(sum(s["delay_hours"] for s in segs), 2),
    }
    if explain:
        out["delay_explanation"] = explain_route_delay(M["route_delay"], rx)
    return out


def _hours_text(h: float) -> str:
    if h < 0:
        h = 0
    d = int(h // 24)
    r = h - 24 * d
    hh = int(r)
    mm = int(round((r - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    if d:
        return f"{d}d {hh}h {mm:02d}m"
    return f"{hh}h {mm:02d}m"


# --------------------------------------------------------------------------
# Public: plan routes
# --------------------------------------------------------------------------
def plan(origin: str, dest: str, profile_key: str = "balanced",
         ts: Optional[str] = None, k: int = 3,
         overrides: Optional[dict] = None) -> dict:
    if origin not in NODES or dest not in NODES:
        raise ValueError("unknown origin or destination node")
    if origin == dest:
        raise ValueError("origin and destination are the same")

    profile = PROFILES.get(profile_key, PROFILES["balanced"])
    state = inference.network_state(ts, overrides)

    # ---- Escalation ladder -------------------------------------------------
    # Emergency routing starts strict and relaxes only as far as it must,
    # reporting exactly which constraint had to be given up. An operator must
    # never be shown a "safe" route that silently crossed a failing slope.
    if profile.key == "emergency":
        ladder = [
            (True, 0.75, None),
            (True, 0.88, "No route avoids every segment above 0.75 failure "
                         "probability; best available stays under 0.88."),
            (True, 1.01, "No route avoids high-probability-failure segments. "
                         "Route shown avoids only segments already predicted "
                         "BLOCKED."),
            (False, 1.01, "No fully open route exists. Showing least-bad options "
                          "— these cross at least one segment predicted BLOCKED. "
                          "Evaluate air lift (Guwahati / Dimapur / Imphal / "
                          "Agartala) or rail before committing road movement."),
        ]
    else:
        ladder = [(False, 1.01, None)]

    paths: List[List[str]] = []
    fallback_note = None
    for avoid_blocked, ceiling, note in ladder:
        G = build_graph(state, profile, hard_avoid_blocked=avoid_blocked,
                        risk_ceiling=ceiling)
        paths = k_diverse_paths(G, origin, dest, k=k)
        if paths:
            fallback_note = note
            break

    if not paths:
        return {"origin": origin, "dest": dest, "routes": [],
                "error": "no road route exists between these nodes"}

    routes = [score_route(state, p, profile, explain=(i == 0))
              for i, p in enumerate(paths)]
    routes = [r for r in routes if r]

    # ---- Reject absurd alternates -----------------------------------------
    # The diversity penalty will happily propose a 1,500 km loop. A control room
    # cannot use that. Keep alternates within 2.2x the best route's distance.
    if routes:
        shortest = min(r["distance_km"] for r in routes)
        kept = [r for r in routes if r["distance_km"] <= shortest * 2.2]
        dropped = len(routes) - len(kept)
        routes = kept or routes
    else:
        dropped = 0

    routes.sort(key=lambda r: (r["n_blocked"], r["composite_score"]))
    for i, r in enumerate(routes):
        r["rank"] = i + 1
        r["recommended"] = (i == 0)
    best = routes[0] if routes else None

    # Why this one rather than the runner-up? Stated, not implied.
    comparison_note = None
    if len(routes) >= 2:
        a, b = routes[0], routes[1]
        dt = b["eta_hours"] - a["eta_hours"]
        dr = b["max_risk"] - a["max_risk"]
        if dt > 0 and dr > 0:
            comparison_note = (f"Recommended route is {abs(dt):.1f} h faster AND "
                               f"{abs(dr):.2f} lower in peak risk than the next option.")
        elif dt < 0 and dr > 0:
            comparison_note = (f"Recommended route is {abs(dt):.1f} h slower but cuts "
                               f"peak segment risk by {abs(dr):.2f} — under the "
                               f"'{profile.label}' profile that trade is taken.")
        elif dt > 0 and dr <= 0:
            comparison_note = (f"Recommended route saves {abs(dt):.1f} h; peak risk is "
                               f"{abs(dr):.2f} higher, within the profile's tolerance.")

    return {
        "ts": state["ts"],
        "origin": origin, "origin_name": NODES[origin].name,
        "dest": dest, "dest_name": NODES[dest].name,
        "profile": {"key": profile.key, "label": profile.label,
                    "description": profile.description},
        "routes": routes,
        "recommended": best["path_names"] if best else None,
        "comparison_note": comparison_note,
        "fallback_note": fallback_note,
        "alternates_discarded_as_impractical": dropped,
        "counterfactual": bool(overrides),
    }


# --------------------------------------------------------------------------
# CRITICALITY — single points of failure
# --------------------------------------------------------------------------
def criticality(ts: Optional[str] = None, hub: str = SUPPLY_HUB,
                overrides: Optional[dict] = None) -> dict:
    """
    For every segment: delete it, re-solve reachability and travel time from the
    regional supply hub, and report what breaks.
    """
    state = inference.network_state(ts, overrides)
    G = distance_graph(state)

    base = nx.single_source_dijkstra_path_length(G, hub, weight="hours")
    reachable0 = set(base)
    n_nodes = G.number_of_nodes()

    # Structural view (weather-independent): cut vertices and bridge edges
    art = set(nx.articulation_points(G))
    bridges = {tuple(sorted(e)) for e in nx.bridges(G)}

    rows: List[dict] = []
    for rid, s in state["segments"].items():
        u, v = s["u"], s["v"]
        if not G.has_edge(u, v):
            continue
        data = G[u][v]
        G.remove_edge(u, v)
        try:
            after = nx.single_source_dijkstra_path_length(G, hub, weight="hours")
        except nx.NodeNotFound:
            after = {}
        isolated = sorted(reachable0 - set(after))
        extra = 0.0
        worst_extra = 0.0
        for n2, h0 in base.items():
            if n2 in after:
                d = after[n2] - h0
                if d > 0.01:
                    extra += d
                    worst_extra = max(worst_extra, d)
        G.add_edge(u, v, **data)

        pop_cut = sum(NODES[n].population_k for n in isolated)
        is_bridge = tuple(sorted((u, v))) in bridges
        score = (len(isolated) * 40.0) + (pop_cut * 0.08) + extra + worst_extra * 2.0
        rows.append({
            "road_id": rid, "corridor": s["corridor"], "name": s["name"],
            "state": s["state"], "length_km": s["length_km"],
            "terrain": s["terrain"], "strategic": s["strategic"],
            "coords": s["coords"],
            "current_risk_status": s["risk_status"],
            "current_risk_score": s["risk_score"],
            "is_structural_bridge": is_bridge,
            "isolates_nodes": len(isolated),
            "isolated_names": [NODES[n].name for n in isolated][:14],
            "population_cut_off_k": pop_cut,
            "extra_network_hours": round(extra, 2),
            "worst_single_detour_hours": round(worst_extra, 2),
            "criticality_score": round(score, 2),
            "notes": s["notes"],
        })

    rows.sort(key=lambda r: -r["criticality_score"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        # Exposure = how critical it is × how likely it is to fail right now
        r["exposure_score"] = round(r["criticality_score"] * r["current_risk_score"], 2)

    by_exposure = sorted(rows, key=lambda r: -r["exposure_score"])

    return {
        "ts": state["ts"], "hub": hub, "hub_name": NODES[hub].name,
        "nodes_total": n_nodes,
        "nodes_reachable_from_hub": len(reachable0),
        "articulation_points": sorted(NODES[a].name for a in art),
        "structural_bridge_count": len(bridges),
        "ranked_by_criticality": rows[:25],
        "ranked_by_live_exposure": by_exposure[:15],
        "interpretation": (
            "Criticality is structural: how much of the network depends on this "
            "one segment. Live exposure multiplies that by today's predicted "
            "failure probability — it is the watch-list a control room should "
            "open the morning with."),
    }


# --------------------------------------------------------------------------
# DISTRICT ACCESSIBILITY INDEX
# --------------------------------------------------------------------------
def accessibility(ts: Optional[str] = None, hub: str = SUPPLY_HUB,
                  overrides: Optional[dict] = None) -> dict:
    """
    Composite 0-100 index per node. Every sub-score is published so the number
    is auditable rather than a black box.
      reach       : travel time from the supply hub (risk-adjusted)
      reliability : 1 - mean risk along the best route
      redundancy  : independent route options that do not share a cut edge
      terrain     : share of hill/pass km on the approach
      disruption  : live blockade penalty
    """
    state = inference.network_state(ts, overrides)
    G = distance_graph(state)
    profile = PROFILES["balanced"]
    Gc = build_graph(state, profile)

    t = datetime.fromisoformat(state["ts"])
    hours = nx.single_source_dijkstra_path_length(G, hub, weight="hours")
    paths = nx.single_source_dijkstra_path(G, hub, weight="hours")
    bridges = {tuple(sorted(e)) for e in nx.bridges(G)}

    out: List[dict] = []
    for nid, node in NODES.items():
        if nid == hub:
            continue
        if nid not in hours:
            out.append({
                "node_id": nid, "district": node.name, "state": node.state,
                "state_name": STATES[node.state], "lat": node.lat, "lon": node.lon,
                "reachable": False, "index": 0.0, "grade": "F",
                "travel_hours": None, "components": {}, "bottleneck": None,
            })
            continue

        path = paths[nid]
        segs = [state["segments"][G[path[i]][path[i + 1]]["road_id"]]
                for i in range(len(path) - 1)]
        th = hours[nid]
        mean_risk = float(np.mean([s["risk_score"] for s in segs])) if segs else 0.0
        max_risk = float(np.max([s["risk_score"] for s in segs])) if segs else 0.0
        hill_km = sum(s["length_km"] for s in segs
                      if s["terrain"] in ("hilly", "high_pass"))
        total_km = sum(s["length_km"] for s in segs) or 1.0
        n_bridge_edges = sum(1 for i in range(len(path) - 1)
                             if tuple(sorted((path[i], path[i + 1]))) in bridges)
        disr = sum(1 for s in segs if s["disruption"])

        alts = k_diverse_paths(Gc, hub, nid, k=3)
        # count alternates that are meaningfully different (<60% shared edges)
        base_edges = {tuple(sorted((path[i], path[i + 1])))
                      for i in range(len(path) - 1)}
        independent = 1
        for p in alts[1:]:
            e = {tuple(sorted((p[i], p[i + 1]))) for i in range(len(p) - 1)}
            if len(e & base_edges) / max(len(e), 1) < 0.6:
                independent += 1

        # --- sub-scores, all 0..100
        reach = 100.0 * math.exp(-th / 14.0)                    # 14 h ~ 37 pts
        reliability = 100.0 * (1.0 - (0.6 * mean_risk + 0.4 * max_risk))
        redundancy = {1: 22.0, 2: 68.0, 3: 92.0}.get(independent, 100.0)
        if n_bridge_edges:
            redundancy = min(redundancy, 18.0)                   # a true cut edge
        terrain = 100.0 * (1.0 - min(hill_km / total_km, 1.0) * 0.75)
        disruption = 100.0 if disr == 0 else max(0.0, 100.0 - 55.0 * disr)

        index = (0.28 * reach + 0.27 * reliability + 0.22 * redundancy +
                 0.13 * terrain + 0.10 * disruption)
        grade = ("A" if index >= 80 else "B" if index >= 65 else
                 "C" if index >= 50 else "D" if index >= 35 else "E")

        if n_bridge_edges:
            redundancy_note = (
                f"{independent} route option(s) exist on paper, but all of them "
                f"funnel through {n_bridge_edges} cut edge(s) — a single "
                f"segment failure severs this district entirely. Redundancy is "
                f"scored on the cut, not the count.")
        elif independent == 1:
            redundancy_note = ("Only one practical corridor serves this district; "
                               "alternates share most of their length.")
        else:
            redundancy_note = (f"{independent} structurally independent corridors "
                               f"available.")

        bottleneck = None
        if segs:
            w = max(segs, key=lambda s: s["risk_score"])
            bottleneck = {"road_id": w["road_id"], "name": w["name"],
                          "corridor": w["corridor"],
                          "risk_status": w["risk_status"],
                          "risk_score": w["risk_score"]}

        out.append({
            "node_id": nid, "district": node.name, "state": node.state,
            "state_name": STATES[node.state], "lat": node.lat, "lon": node.lon,
            "population_k": node.population_k,
            "reachable": True,
            "travel_hours": round(th, 2),
            "travel_text": _hours_text(th),
            "index": round(index, 1),
            "grade": grade,
            "independent_routes": independent,
            "on_structural_cut_edge": n_bridge_edges > 0,
            "cut_edges_on_path": n_bridge_edges,
            "redundancy_note": redundancy_note,
            "components": {
                "reach": round(reach, 1),
                "reliability": round(reliability, 1),
                "redundancy": round(redundancy, 1),
                "terrain": round(terrain, 1),
                "disruption": round(disruption, 1),
            },
            "bottleneck": bottleneck,
        })

    out.sort(key=lambda d: d["index"])
    by_state: Dict[str, List[float]] = defaultdict(list)
    for d in out:
        by_state[d["state"]].append(d["index"])
    state_avg = sorted(
        [{"state": k, "state_name": STATES[k],
          "mean_index": round(float(np.mean(v)), 1), "districts": len(v)}
         for k, v in by_state.items()],
        key=lambda d: d["mean_index"])

    return {
        "ts": state["ts"], "hub": hub, "hub_name": NODES[hub].name,
        "weights": {"reach": 0.28, "reliability": 0.27, "redundancy": 0.22,
                    "terrain": 0.13, "disruption": 0.10},
        "districts": out,
        "most_vulnerable": out[:12],
        "by_state": state_avg,
        "unreachable": [d["district"] for d in out if not d["reachable"]],
    }


# --------------------------------------------------------------------------
# DEPARTURE WINDOW OPTIMISER
# --------------------------------------------------------------------------
def departure_windows(origin: str, dest: str, profile_key: str = "balanced",
                      horizon_hours: int = 72, ts: Optional[str] = None) -> dict:
    """
    Run the full pipeline across every forecast window in the horizon and find
    the cheapest time to leave. This is the single most operationally useful
    output in the product: "hold the convoy 9 hours and you save 6."
    """
    profile = PROFILES.get(profile_key, PROFILES["balanced"])
    t0 = ts or inference.now_ts()
    slots = [t0] + inference.available_weather_ts(t0, horizon_hours)

    out: List[dict] = []
    for s in slots:
        try:
            st = inference.network_state(s)
            emerg = profile.key == "emergency"
            G = build_graph(st, profile, hard_avoid_blocked=emerg,
                            risk_ceiling=0.88 if emerg else 1.01)
            paths = k_diverse_paths(G, origin, dest, k=1)
            if not paths:        # relax, same as plan()'s ladder
                G = build_graph(st, profile, hard_avoid_blocked=False)
                paths = k_diverse_paths(G, origin, dest, k=1)
            if not paths:
                out.append({"depart_ts": s, "feasible": False})
                continue
            r = score_route(st, paths[0], profile)
            out.append({
                "depart_ts": s,
                "feasible": True,
                "eta_hours": r["eta_hours"],
                "arrive_ts": (datetime.fromisoformat(s) +
                              timedelta(hours=r["eta_hours"])
                              ).isoformat(timespec="minutes"),
                "delay_hours": r["predicted_delay_hours"],
                "max_risk": r["max_risk"],
                "avg_risk": r["avg_risk"],
                "n_blocked": r["n_blocked"],
                "status": r["status"],
                "distance_km": r["distance_km"],
                "composite_score": r["composite_score"],
                "path_names": r["path_names"],
            })
        except Exception as e:       # never let one bad slot kill the sweep
            out.append({"depart_ts": s, "feasible": False, "error": str(e)})

    feas = [o for o in out if o.get("feasible")]
    best_time = min(feas, key=lambda o: o["eta_hours"]) if feas else None
    best_safe = min(feas, key=lambda o: (o["n_blocked"], o["max_risk"])) if feas else None
    now_slot = out[0] if out else None

    rec = None
    if best_safe and now_slot and now_slot.get("feasible"):
        dh = (datetime.fromisoformat(best_safe["depart_ts"]) -
              datetime.fromisoformat(now_slot["depart_ts"])).total_seconds() / 3600.0
        saved = now_slot["eta_hours"] - best_safe["eta_hours"]
        risk_cut = now_slot["max_risk"] - best_safe["max_risk"]
        if dh <= 0.5:
            rec = "Depart now — no later window in the horizon is better."
        elif saved > 0.75 or risk_cut > 0.12:
            rec = (f"Hold {dh:.0f} h and depart at "
                   f"{datetime.fromisoformat(best_safe['depart_ts']):%d %b %H:%M}: "
                   f"transit {saved:+.1f} h, peak segment risk {risk_cut:+.2f}. "
                   f"Waiting is cheaper than driving.")
        else:
            rec = ("Depart now — later windows are not materially better and "
                   "holding cargo has its own cost.")

    return {
        "origin": origin, "origin_name": NODES[origin].name,
        "dest": dest, "dest_name": NODES[dest].name,
        "profile": profile.key,
        "horizon_hours": horizon_hours,
        "windows": out,
        "best_by_time": best_time,
        "best_by_safety": best_safe,
        "depart_now": now_slot,
        "recommendation": rec,
    }


# --------------------------------------------------------------------------
# COLD-CHAIN / CRITICAL CARGO VIABILITY
# --------------------------------------------------------------------------
CARGO = {
    "blood": {"label": "Whole blood / platelets", "max_hours": 8,
              "note": "Platelets 5-day shelf life; transport window is hours."},
    "vaccine": {"label": "Vaccines (2–8 °C cold chain)", "max_hours": 36,
                "note": "Passive cold box holds 2–8 °C for ~36 h in NER ambient."},
    "oxygen": {"label": "Medical oxygen cylinders", "max_hours": 18,
               "note": "Hospital buffer stock typically under a day."},
    "medicine": {"label": "General pharmaceuticals", "max_hours": 72,
                 "note": "Temperature-stable, tolerant of delay."},
    "perishable": {"label": "Fresh produce / fish", "max_hours": 24,
                   "note": "Spoilage accelerates above 24 h without reefer."},
    "relief": {"label": "Disaster relief material", "max_hours": 48,
               "note": "Time-critical in the first 72 h after an event."},
    "general": {"label": "General freight", "max_hours": 240, "note": ""},
}


def cargo_viability(route: dict, cargo_type: str) -> dict:
    spec = CARGO.get(cargo_type, CARGO["general"])
    eta = route["eta_hours"]
    limit = spec["max_hours"]
    margin = limit - eta
    if margin >= limit * 0.35:
        verdict, sev = "viable", "ok"
    elif margin >= 0:
        verdict, sev = "tight", "warn"
    else:
        verdict, sev = "not viable by road", "critical"
    return {
        "cargo_type": cargo_type, "cargo_label": spec["label"],
        "max_transit_hours": limit,
        "predicted_eta_hours": round(eta, 2),
        "margin_hours": round(margin, 2),
        "verdict": verdict, "severity": sev,
        "note": spec["note"],
        "advice": (
            "Road movement is within the cargo's viability window."
            if verdict == "viable" else
            "Margin is thin — dispatch immediately, pre-position a relay vehicle, "
            "and brief the receiving facility on the risk."
            if verdict == "tight" else
            f"Predicted road transit ({_hours_text(eta)}) exceeds the "
            f"{limit} h limit for {spec['label']}. Escalate to air lift "
            f"(Guwahati/Dimapur/Imphal/Agartala airfields) or split the "
            f"consignment to a forward cold-chain node."),
    }
