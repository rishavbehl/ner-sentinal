"""
Single source of truth for the ML feature contract.

Both the training pipeline and the live inference path build feature vectors
through THIS module. That removes train/serve skew -- the classic way a
hackathon ML demo silently breaks.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------
# Categorical encodings (ordinal, deliberately: these are ordered concepts and
# tree models split them cleanly without one-hot blowup)
# --------------------------------------------------------------------------
ROAD_CLASS_ENC = {"RR": 0, "SH": 1, "NH": 2}
TERRAIN_ENC = {"plain": 0, "valley": 1, "hilly": 2, "high_pass": 3}
WEATHER_ENC = {
    "clear": 0, "cloudy": 1, "light_rain": 2, "rain": 3,
    "heavy_rain": 4, "storm": 5, "snow": 6, "fog": 7,
}
WEATHER_SEVERITY = {          # used for human-readable alerts
    "clear": 0.0, "cloudy": 0.05, "light_rain": 0.2, "rain": 0.4,
    "heavy_rain": 0.8, "storm": 1.0, "snow": 0.9, "fog": 0.35,
}

RISK_LABELS = {0: "safe", 1: "risky", 2: "blocked"}
RISK_LABEL_IDS = {v: k for k, v in RISK_LABELS.items()}

# --------------------------------------------------------------------------
# Feature order — NEVER reorder without retraining.
# --------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    # --- static road geometry / engineering
    "length_km",
    "road_class_enc",
    "lanes",
    "terrain_enc",
    "avg_slope_pct",
    "max_elev_m",
    "bridge_count",
    # --- static physical susceptibility
    "flood_exposure",
    "landslide_base",
    "snow_exposure",
    # --- live hydro-meteorology
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "api_7d",                 # antecedent precipitation index -> soil saturation
    "weather_enc",
    "temperature_c",
    "wind_kmph",
    # --- temporal context
    "month",
    "is_monsoon",
    "hour",
    "is_night",
    # --- incident memory
    "hist_incidents_90d",
    "days_since_last_incident",
    "disruption_flag",        # active bandh / blockade / strike on corridor
]

# Human-readable names for the Explainable-AI panel
FEATURE_LABELS: Dict[str, str] = {
    "length_km": "Segment length",
    "road_class_enc": "Road class (NH/SH/rural)",
    "lanes": "Lane count",
    "terrain_enc": "Terrain type",
    "avg_slope_pct": "Average gradient",
    "max_elev_m": "Maximum elevation",
    "bridge_count": "Bridges / culverts on segment",
    "flood_exposure": "Floodplain exposure",
    "landslide_base": "Landslide susceptibility (geology + slope)",
    "snow_exposure": "Snow / ice exposure",
    "rainfall_24h_mm": "Rainfall last 24 h",
    "rainfall_72h_mm": "Rainfall last 72 h",
    "api_7d": "Soil saturation (7-day antecedent rainfall)",
    "weather_enc": "Current weather condition",
    "temperature_c": "Temperature",
    "wind_kmph": "Wind speed",
    "month": "Month of year",
    "is_monsoon": "Monsoon season",
    "hour": "Hour of day",
    "is_night": "Night-time driving",
    "hist_incidents_90d": "Incidents here in last 90 days",
    "days_since_last_incident": "Days since last incident",
    "disruption_flag": "Active blockade / bandh",
}

FEATURE_UNITS: Dict[str, str] = {
    "length_km": "km", "avg_slope_pct": "%", "max_elev_m": "m",
    "rainfall_24h_mm": "mm", "rainfall_72h_mm": "mm", "api_7d": "mm",
    "temperature_c": "°C", "wind_kmph": "km/h",
    "days_since_last_incident": "days",
}

MONSOON_MONTHS = {5, 6, 7, 8, 9}      # May–Sept: NER monsoon incl. pre-monsoon tail


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
def build_feature_dict(seg: dict, wx: dict, ctx: dict) -> Dict[str, float]:
    """
    seg : static road-segment attributes (from DB / geography.Segment)
    wx  : live or forecast weather for the segment's district
    ctx : temporal + incident context
    """
    month = int(ctx["month"])
    hour = int(ctx["hour"])
    return {
        "length_km": float(seg["length_km"]),
        "road_class_enc": float(ROAD_CLASS_ENC.get(seg["road_class"], 1)),
        "lanes": float(seg["lanes"]),
        "terrain_enc": float(TERRAIN_ENC.get(seg["terrain"], 0)),
        "avg_slope_pct": float(seg["avg_slope_pct"]),
        "max_elev_m": float(seg["max_elev_m"]),
        "bridge_count": float(seg["bridge_count"]),
        "flood_exposure": float(seg["flood_exposure"]),
        "landslide_base": float(seg["landslide_base"]),
        "snow_exposure": float(seg["snow_exposure"]),
        "rainfall_24h_mm": float(wx["rainfall_24h_mm"]),
        "rainfall_72h_mm": float(wx["rainfall_72h_mm"]),
        "api_7d": float(wx["api_7d"]),
        "weather_enc": float(WEATHER_ENC.get(wx["weather_condition"], 0)),
        "temperature_c": float(wx["temperature_c"]),
        "wind_kmph": float(wx["wind_kmph"]),
        "month": float(month),
        "is_monsoon": 1.0 if month in MONSOON_MONTHS else 0.0,
        "hour": float(hour),
        "is_night": 1.0 if (hour >= 19 or hour < 5) else 0.0,
        "hist_incidents_90d": float(ctx.get("hist_incidents_90d", 0)),
        "days_since_last_incident": float(ctx.get("days_since_last_incident", 999)),
        "disruption_flag": float(ctx.get("disruption_flag", 0)),
    }


def to_vector(fd: Dict[str, float]) -> np.ndarray:
    return np.array([fd[c] for c in FEATURE_COLUMNS], dtype=np.float64)


def to_matrix(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=np.float64)


# --------------------------------------------------------------------------
# Route-level features (second model: route delay regressor)
# --------------------------------------------------------------------------
ROUTE_FEATURE_COLUMNS: List[str] = [
    "route_length_km",
    "n_segments",
    "freeflow_hours",
    "avg_risk_score",
    "max_risk_score",
    "n_risky_segments",
    "n_blocked_segments",
    "frac_hilly",
    "frac_nh",
    "total_ascent_m",
    "max_rainfall_24h",
    "mean_api_7d",
    "n_incidents_on_route",
    "n_bridges",
    "n_state_crossings",
    "departs_at_night",
    "is_monsoon",
    "disruption_segments",
]

ROUTE_FEATURE_LABELS = {
    "route_length_km": "Total route length",
    "n_segments": "Number of segments",
    "freeflow_hours": "Free-flow driving time",
    "avg_risk_score": "Average segment risk",
    "max_risk_score": "Worst segment risk",
    "n_risky_segments": "Risky segments on route",
    "n_blocked_segments": "Blocked segments on route",
    "frac_hilly": "Share of hill/pass terrain",
    "frac_nh": "Share on national highway",
    "total_ascent_m": "Total climb",
    "max_rainfall_24h": "Peak 24 h rainfall on route",
    "mean_api_7d": "Mean soil saturation on route",
    "n_incidents_on_route": "Recent incidents on route",
    "n_bridges": "Bridges crossed",
    "n_state_crossings": "Inter-state crossings (checkposts)",
    "departs_at_night": "Night departure",
    "is_monsoon": "Monsoon season",
    "disruption_segments": "Segments under blockade",
}


def route_vector(rf: Dict[str, float]) -> np.ndarray:
    return np.array([rf[c] for c in ROUTE_FEATURE_COLUMNS], dtype=np.float64)
