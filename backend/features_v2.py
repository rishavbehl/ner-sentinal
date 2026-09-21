"""
Feature Contract v2 for Neural Prediction Core & Tree Baseline.

Implements PRD Appendix A:
  - Preserves exact FEATURE_COLUMNS order (23 features).
  - Supplies transform_v2_vector / transform_v2_matrix for neural networks:
      * Categorical extraction for embeddings: road_class_enc (3), terrain_enc (4), weather_enc (8)
      * Continuous transforms: log/scale for rainfall, API, wind, incidents
      * Sentinel handling: days_since_last_incident -> exp(-min(d, 365)/45), log1p, plus missing flag for 999
      * Cyclic time: sin/cos for month (1..12) and hour (0..23)
  - Provides a strict input-leakage guard testable in CI.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .features import FEATURE_COLUMNS

# Categorical column positions in FEATURE_COLUMNS
CAT_COLS = ["road_class_enc", "terrain_enc", "weather_enc"]
CAT_INDICES = [FEATURE_COLUMNS.index(c) for c in CAT_COLS]
CAT_CARDINALITIES = [3, 4, 8]  # road_class (0..2), terrain (0..3), weather (0..7)

# Continuous columns indices (excluding the 3 categoricals)
NUM_INDICES = [i for i in range(len(FEATURE_COLUMNS)) if i not in CAT_INDICES]


def transform_v2_row(raw_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a single raw 23-feature vector into (continuous_features, categorical_features).

    Continuous output dimensions:
      length_km: log1p
      lanes: raw/4.0
      avg_slope_pct: raw/25.0
      max_elev_m: raw/3000.0
      bridge_count: raw/10.0
      flood_exposure: raw (0..1)
      landslide_base: raw (0..1)
      snow_exposure: raw (0..1)
      rainfall_24h_mm: raw/100.0 and log1p(raw)
      rainfall_72h_mm: raw/200.0 and log1p(raw)
      api_7d: raw/250.0 and log1p(raw)
      temperature_c: (raw - 20.0)/15.0
      wind_kmph: log1p(raw)/4.0
      month: sin(2pi*m/12), cos(2pi*m/12)
      is_monsoon: raw (0 or 1)
      hour: sin(2pi*h/24), cos(2pi*h/24)
      is_night: raw (0 or 1)
      hist_incidents_90d: log1p(min(raw, 6.0))
      days_since_last_incident: exp(-min(d, 365)/45.0), log1p(min(d, 365)), is_missing (1 if d >= 900 else 0)
      disruption_flag: raw (0 or 1)
    Categorical output dimensions:
      [road_class (0..2), terrain (0..3), weather (0..7)]
    """
    col_map = {FEATURE_COLUMNS[i]: raw_row[i] for i in range(len(FEATURE_COLUMNS))}

    # Continuous engineering
    length_km = math.log1p(max(0.0, col_map["length_km"]))
    lanes = col_map["lanes"] / 4.0
    avg_slope_pct = col_map["avg_slope_pct"] / 25.0
    max_elev_m = col_map["max_elev_m"] / 3000.0
    bridge_count = col_map["bridge_count"] / 10.0
    flood_exposure = col_map["flood_exposure"]
    landslide_base = col_map["landslide_base"]
    snow_exposure = col_map["snow_exposure"]

    r24 = max(0.0, col_map["rainfall_24h_mm"])
    r24_scaled = r24 / 100.0
    r24_log = math.log1p(r24)

    r72 = max(0.0, col_map["rainfall_72h_mm"])
    r72_scaled = r72 / 200.0
    r72_log = math.log1p(r72)

    api = max(0.0, col_map["api_7d"])
    api_scaled = api / 250.0
    api_log = math.log1p(api)

    temp_c = (col_map["temperature_c"] - 20.0) / 15.0
    wind = math.log1p(max(0.0, col_map["wind_kmph"])) / 4.0

    m = col_map["month"]
    m_sin = math.sin(2.0 * math.pi * m / 12.0)
    m_cos = math.cos(2.0 * math.pi * m / 12.0)
    is_monsoon = col_map["is_monsoon"]

    h = col_map["hour"]
    h_sin = math.sin(2.0 * math.pi * h / 24.0)
    h_cos = math.cos(2.0 * math.pi * h / 24.0)
    is_night = col_map["is_night"]

    inc = math.log1p(min(max(0.0, col_map["hist_incidents_90d"]), 6.0))
    d_raw = col_map["days_since_last_incident"]
    d_clipped = min(max(0.0, d_raw), 365.0)
    d_decay = math.exp(-d_clipped / 45.0)
    d_log = math.log1p(d_clipped)
    d_missing = 1.0 if d_raw >= 900.0 else 0.0

    disr = col_map["disruption_flag"]

    num_vec = np.array([
        length_km, lanes, avg_slope_pct, max_elev_m, bridge_count,
        flood_exposure, landslide_base, snow_exposure,
        r24_scaled, r24_log, r72_scaled, r72_log, api_scaled, api_log,
        temp_c, wind, m_sin, m_cos, is_monsoon, h_sin, h_cos, is_night,
        inc, d_decay, d_log, d_missing, disr
    ], dtype=np.float32)

    cat_vec = np.array([
        int(min(max(0, col_map["road_class_enc"]), 2)),
        int(min(max(0, col_map["terrain_enc"]), 3)),
        int(min(max(0, col_map["weather_enc"]), 7)),
    ], dtype=np.int64)

    return num_vec, cat_vec


def transform_v2_batch(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform 2D batch of shape (N, 23) into (N, 27) continuous and (N, 3) categorical arrays."""
    n = len(X_raw)
    nums = []
    cats = []
    for i in range(n):
        num, cat = transform_v2_row(X_raw[i])
        nums.append(num)
        cats.append(cat)
    return np.stack(nums), np.stack(cats)


def transform_route_v2(rf_raw: np.ndarray) -> np.ndarray:
    """Standardise and log-scale route-level feature vectors (18 features)."""
    # 0: route_length_km, 1: n_segments, 2: freeflow_hours, 3: avg_risk_score,
    # 4: max_risk_score, 5: n_risky_segments, 6: n_blocked_segments, 7: frac_hilly,
    # 8: frac_nh, 9: total_ascent_m, 10: max_rainfall_24h, 11: mean_api_7d,
    # 12: n_incidents_on_route, 13: n_bridges, 14: n_state_crossings,
    # 15: departs_at_night, 16: is_monsoon, 17: disruption_segments
    x = rf_raw.copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out = x.copy()
    out[:, 0] = np.log1p(np.maximum(0.0, out[:, 0])) / 5.0  # length
    out[:, 1] = out[:, 1] / 15.0                           # n_segments
    out[:, 2] = np.log1p(np.maximum(0.0, out[:, 2])) / 3.0  # freeflow
    out[:, 9] = np.log1p(np.maximum(0.0, out[:, 9])) / 8.0  # ascent
    out[:, 10] = np.log1p(np.maximum(0.0, out[:, 10])) / 4.0 # rain
    out[:, 11] = np.log1p(np.maximum(0.0, out[:, 11])) / 4.0 # api
    out[:, 12] = np.log1p(np.maximum(0.0, out[:, 12])) / 3.0 # incidents
    out[:, 13] = np.log1p(np.maximum(0.0, out[:, 13])) / 3.0 # bridges
    return out.astype(np.float32)


def check_leakage(candidate_cols: List[str]) -> bool:
    """Verify that forbidden label/target columns never leak into model input features."""
    forbidden = {"hazard", "risk_label", "delay_hours", "pred_label", "pred_severity", "pred_delay_hours"}
    intersection = set(candidate_cols) & forbidden
    if intersection:
        raise ValueError(f"Input feature leakage detected! Forbidden columns present: {intersection}")
    return True
