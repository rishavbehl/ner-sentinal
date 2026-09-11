"""SQLite storage layer + schema."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "sentinel.db")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

for _d in (DATA_DIR, ARTIFACT_DIR, UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS roads (
    road_id TEXT PRIMARY KEY,
    corridor TEXT, name TEXT, road_class TEXT, lanes INTEGER, strategic INTEGER,
    u TEXT, v TEXT, u_name TEXT, v_name TEXT,
    state TEXT, district TEXT,
    start_lat REAL, start_lon REAL, end_lat REAL, end_lon REAL,
    length_km REAL, terrain TEXT, avg_slope_pct REAL, max_elev_m INTEGER,
    freeflow_kmph REAL, flood_exposure REAL, landslide_base REAL,
    snow_exposure REAL, bridge_count INTEGER, notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_roads_corridor ON roads(corridor);
CREATE INDEX IF NOT EXISTS idx_roads_state ON roads(state);

CREATE TABLE IF NOT EXISTS districts (
    district TEXT PRIMARY KEY, node_id TEXT, state TEXT, state_name TEXT,
    lat REAL, lon REAL, elev_m INTEGER, kind TEXT, population_k INTEGER,
    floodplain INTEGER
);

CREATE TABLE IF NOT EXISTS weather (
    district TEXT, ts TEXT,
    rainfall_mm REAL, rainfall_24h_mm REAL, rainfall_72h_mm REAL, api_7d REAL,
    weather_condition TEXT, temperature_c REAL, wind_kmph REAL,
    is_forecast INTEGER DEFAULT 0,
    PRIMARY KEY (district, ts)
);
CREATE INDEX IF NOT EXISTS idx_weather_ts ON weather(ts);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    road_id TEXT, district TEXT, state TEXT,
    lat REAL, lon REAL,
    issue_type TEXT, severity TEXT, description TEXT,
    reported_at TEXT, reporter_id TEXT, source TEXT,
    photo_url TEXT, verified INTEGER DEFAULT 0,
    cleared_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_inc_road ON incidents(road_id);
CREATE INDEX IF NOT EXISTS idx_inc_time ON incidents(reported_at);

CREATE TABLE IF NOT EXISTS observations (
    road_id TEXT, ts TEXT,
    length_km REAL, road_class_enc REAL, lanes REAL, terrain_enc REAL,
    avg_slope_pct REAL, max_elev_m REAL, bridge_count REAL,
    flood_exposure REAL, landslide_base REAL, snow_exposure REAL,
    rainfall_24h_mm REAL, rainfall_72h_mm REAL, api_7d REAL, weather_enc REAL,
    temperature_c REAL, wind_kmph REAL,
    month REAL, is_monsoon REAL, hour REAL, is_night REAL,
    hist_incidents_90d REAL, days_since_last_incident REAL, disruption_flag REAL,
    hazard REAL, risk_label INTEGER, delay_hours REAL,
    PRIMARY KEY (road_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts);

CREATE TABLE IF NOT EXISTS route_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT, dest TEXT, ts TEXT,
    route_length_km REAL, n_segments REAL, freeflow_hours REAL,
    avg_risk_score REAL, max_risk_score REAL, n_risky_segments REAL,
    n_blocked_segments REAL, frac_hilly REAL, frac_nh REAL, total_ascent_m REAL,
    max_rainfall_24h REAL, mean_api_7d REAL, n_incidents_on_route REAL,
    n_bridges REAL, n_state_crossings REAL, departs_at_night REAL,
    is_monsoon REAL, disruption_segments REAL,
    route_delay_hours REAL
);

CREATE TABLE IF NOT EXISTS disruptions (
    disruption_id TEXT PRIMARY KEY,
    corridor TEXT, kind TEXT, start_ts TEXT, end_ts TEXT, note TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    created_at TEXT, severity TEXT, scope TEXT, target TEXT,
    title_en TEXT, body_en TEXT, payload TEXT, acknowledged INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(created_at);

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id TEXT PRIMARY KEY,
    label TEXT, cargo_type TEXT, priority TEXT,
    origin TEXT, dest TEXT, created_at TEXT,
    route_json TEXT, status TEXT,
    max_transit_hours REAL
);

-- key/value store: live-weather provenance, runtime flags
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- REAL GPS: current state of each tracked device
CREATE TABLE IF NOT EXISTS live_vehicles (
    vehicle_id TEXT PRIMARY KEY,
    label TEXT, cargo_type TEXT, driver TEXT,
    lat REAL, lon REAL,
    speed_kmph REAL, heading REAL, accuracy_m REAL,
    road_id TEXT, snap_distance_km REAL,
    risk_label INTEGER, risk_status TEXT, severity REAL,
    first_seen TEXT, last_seen TEXT,
    points INTEGER DEFAULT 0,
    dest TEXT
);

-- REAL GPS: the breadcrumb trail
CREATE TABLE IF NOT EXISTS track_points (
    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT, ts TEXT,
    lat REAL, lon REAL, speed_kmph REAL, heading REAL, accuracy_m REAL,
    road_id TEXT, severity REAL
);
CREATE INDEX IF NOT EXISTS idx_track_vehicle ON track_points(vehicle_id, ts);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def cursor(commit: bool = False):
    conn = connect()
    try:
        yield conn
        if commit:
            conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with cursor(commit=True) as c:
        c.executescript(SCHEMA)


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with cursor() as c:
        return rows_to_dicts(c.execute(sql, params).fetchall())


def query_one(sql: str, params: tuple = ()) -> Dict[str, Any] | None:
    with cursor() as c:
        r = c.execute(sql, params).fetchone()
        return dict(r) if r else None


def execute(sql: str, params: tuple = ()) -> None:
    with cursor(commit=True) as c:
        c.execute(sql, params)


def executemany(sql: str, seq) -> None:
    with cursor(commit=True) as c:
        c.executemany(sql, seq)
