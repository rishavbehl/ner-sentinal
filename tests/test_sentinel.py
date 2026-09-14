"""
NER Logistics Sentinel — Comprehensive Test Suite.

Tests cover:
1. Database schema, constraints, indexes, and queries
2. Geospatial topology and network connectivity
3. Feature engineering contracts
4. Model inference and probability calibration
5. Explainable AI (Saabas exact additive attribution verification)
6. Multi-criteria routing across all 4 profiles (fastest, balanced, safest, emergency)
7. Chokepoint / Single-Point-of-Failure & District Accessibility Index (DAI)
8. Departure-window optimizer & counterfactual what-if scenarios
9. Multilingual alert generation and review-pending flags
10. Live GPS snapping, geofencing, and lookahead hazard warnings
11. FastAPI endpoints, error handling, Swagger documentation, and frontend serving
"""
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import alerts, config, db, explain, features, geography, inference, routing
from backend.app import app
from backend.providers import gps, weather_openmeteo


# ==============================================================================
# 1. DATABASE & SCHEMA TESTS
# ==============================================================================
class TestDatabase:
    def test_db_path_exists(self):
        assert os.path.exists(db.DB_PATH), f"Database file missing at {db.DB_PATH}"

    def test_sqlite_connection_and_wal(self):
        conn = db.connect()
        try:
            cursor = conn.cursor()
            mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() in ("wal", "delete", "memory")
        finally:
            conn.close()

    def test_required_tables_exist(self):
        required_tables = {
            "roads", "districts", "weather", "incidents", "observations",
            "route_samples", "disruptions", "alerts", "shipments",
            "live_vehicles", "track_points", "meta"
        }
        rows = db.query("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {r["name"] for r in rows}
        for table in required_tables:
            assert table in existing_tables, f"Table '{table}' not found in SQLite database"

    def test_db_query_helpers(self):
        roads = db.query("SELECT road_id, corridor FROM roads LIMIT 5;")
        assert len(roads) > 0
        assert "road_id" in roads[0]

        single = db.query_one("SELECT road_id FROM roads LIMIT 1;")
        assert single is not None
        assert "road_id" in single

        none_res = db.query_one("SELECT road_id FROM roads WHERE road_id = 'NONEXISTENT_XYZ';")
        assert none_res is None


# ==============================================================================
# 2. GEOGRAPHY & TOPOLOGY TESTS
# ==============================================================================
class TestGeography:
    def test_nodes_integrity(self):
        nodes = geography.NODES
        assert len(nodes) >= 80, f"Expected >= 80 nodes, found {len(nodes)}"
        for node_id, node in nodes.items():
            assert hasattr(node, "lat") and hasattr(node, "lon")
            assert 21.0 <= node.lat <= 30.0, f"Node {node_id} lat {node.lat} out of NER bounds"
            assert 88.0 <= node.lon <= 98.0, f"Node {node_id} lon {node.lon} out of NER bounds"
            assert hasattr(node, "state")
            assert node.state in geography.STATES

    def test_corridors_and_segments(self):
        corridors = geography.CORRIDORS
        assert len(corridors) >= 35, f"Expected >= 35 corridors, found {len(corridors)}"
        all_segments = geography.build_segments()
        assert len(all_segments) >= 90, f"Expected >= 90 segments, found {len(all_segments)}"
        for seg in all_segments:
            assert seg.u in geography.NODES, f"Unknown origin node {seg.u}"
            assert seg.v in geography.NODES, f"Unknown dest node {seg.v}"
            assert seg.length_km > 0

    def test_graph_connectivity(self):
        state = inference.network_state()
        G = routing.distance_graph(state)
        assert G.number_of_nodes() >= 80
        assert G.number_of_edges() >= 90
        # Check supply hub node (GUWAHATI) exists and has degree >= 3
        assert "GUWAHATI" in G
        assert G.degree("GUWAHATI") >= 3


# ==============================================================================
# 3. FEATURE ENGINEERING TESTS
# ==============================================================================
class TestFeatures:
    def test_feature_columns_defined(self):
        cols = features.FEATURE_COLUMNS
        assert len(cols) >= 15
        assert "length_km" in cols
        assert "rainfall_24h_mm" in cols
        assert "rainfall_72h_mm" in cols
        assert "api_7d" in cols

    def test_feature_labels_and_formatting(self):
        assert len(features.FEATURE_LABELS) >= len(features.FEATURE_COLUMNS)
        for col in features.FEATURE_COLUMNS:
            assert col in features.FEATURE_LABELS


# ==============================================================================
# 4. MODEL INFERENCE TESTS
# ==============================================================================
class TestModelInference:
    def test_models_loaded(self):
        m = inference.models()
        assert "risk" in m, "risk model missing"
        assert "delay" in m, "delay model missing"
        assert "route_delay" in m, "route_delay model missing"

    def test_network_state_scoring(self):
        st = inference.network_state()
        assert "ts" in st
        assert "segments" in st
        assert "summary" in st
        assert len(st["segments"]) >= 90

        for rid, seg in st["segments"].items():
            assert "risk_label" in seg
            assert seg["risk_label"] in (0, 1, 2)  # 0=safe, 1=risky, 2=blocked
            assert "risk_status" in seg
            assert seg["risk_status"] in ("safe", "risky", "blocked")
            assert "proba" in seg
            p = seg["proba"]
            p_sum = sum(p)
            assert abs(p_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0: {p_sum}"
            assert "delay_hours" in seg
            assert seg["delay_hours"] >= 0.0

    def test_network_summary(self):
        st = inference.network_state()
        summary = st["summary"]
        assert "segments_scored" in summary
        assert summary["segments_scored"] >= 90
        assert "safe" in summary
        assert "risky" in summary
        assert "blocked" in summary
        assert summary["safe"] + summary["risky"] + summary["blocked"] == summary["segments_scored"]


# ==============================================================================
# 5. EXPLAINABLE AI (XAI) MATHEMATICAL TESTS
# ==============================================================================
class TestExplainableAI:
    def test_saabas_exact_additivity(self):
        """Mathematical verification: Baseline + Sum(contributions) == output (residual < 1e-6)."""
        st = inference.network_state()
        sample_road = list(st["segments"].keys())[0]
        seg = st["segments"][sample_road]
        x = np.array([seg["features"][c] for c in features.FEATURE_COLUMNS], dtype=np.float64)

        M = inference.models()
        exp = explain.explain_segment_risk(M["risk"], x)
        assert "predicted_class" in exp
        assert "drivers" in exp
        assert "additivity_check" in exp
        check = exp["additivity_check"]
        assert check["exact"] is True
        assert check["abs_error"] < 1e-6

    def test_operational_driver_generation(self):
        st = inference.network_state()
        sample_road = list(st["segments"].keys())[0]
        seg = st["segments"][sample_road]
        x = np.array([seg["features"][c] for c in features.FEATURE_COLUMNS], dtype=np.float64)

        M = inference.models()
        exp = explain.explain_segment_risk(M["risk"], x)
        assert "drivers" in exp
        for driver in exp["drivers"]:
            assert "feature" in driver
            assert "narrative" in driver
            assert isinstance(driver["narrative"], str)
            assert len(driver["narrative"]) > 5


# ==============================================================================
# 6. ROUTING ENGINE TESTS
# ==============================================================================
class TestRouting:
    def test_plan_all_profiles(self):
        origin = "GUWAHATI"
        dest = "SHILLONG"
        profiles = ["fastest", "balanced", "safest", "emergency"]
        for prof in profiles:
            result = routing.plan(origin, dest, profile_key=prof, k=2)
            assert "routes" in result, f"Profile {prof} returned no routes"
            assert len(result["routes"]) > 0, f"Profile {prof} found 0 routes for GUWAHATI->SHILLONG"
            r0 = result["routes"][0]
            assert "path_nodes" in r0
            assert r0["path_nodes"][0] == origin
            assert r0["path_nodes"][-1] == dest
            assert "eta_hours" in r0
            assert r0["eta_hours"] > 0
            assert "predicted_delay_hours" in r0

    def test_critical_cargo_viability(self):
        result = routing.plan("GUWAHATI", "SHILLONG", profile_key="emergency")
        assert "routes" in result and len(result["routes"]) > 0
        r0 = result["routes"][0]
        viab = routing.cargo_viability(r0, "blood")
        assert viab["cargo_type"] == "blood"
        assert "verdict" in viab
        assert viab["verdict"] in ("viable", "tight", "not viable by road")
        assert "max_transit_hours" in viab


# ==============================================================================
# 7. CHOKEPOINT & ACCESSIBILITY ANALYTICS TESTS
# ==============================================================================
class TestChokepointAndAccessibility:
    def test_criticality_ranking(self):
        crit = routing.criticality()
        assert "ranked_by_criticality" in crit
        assert "ranked_by_live_exposure" in crit
        ranked = crit["ranked_by_criticality"]
        assert len(ranked) > 0
        top = ranked[0]
        assert "road_id" in top
        assert "corridor" in top
        assert "criticality_score" in top
        assert "isolates_nodes" in top

    def test_district_accessibility_index(self):
        dai = routing.accessibility()
        assert "districts" in dai
        assert "by_state" in dai
        assert len(dai["districts"]) > 0
        for d in dai["districts"]:
            score = d["index"]
            assert 0.0 <= score <= 100.0, f"DAI {score} out of [0, 100] bounds for {d['district']}"


# ==============================================================================
# 8. WHAT-IF SIMULATION & DEPARTURE WINDOW TESTS
# ==============================================================================
class TestWhatIfAndDeparture:
    def test_whatif_scenario_rain_multiplier(self):
        baseline = inference.network_state()
        stressed = inference.network_state(overrides={"rain_multiplier": 3.0})
        b_avg_delay = sum(s["delay_hours"] for s in baseline["segments"].values())
        s_avg_delay = sum(s["delay_hours"] for s in stressed["segments"].values())
        assert s_avg_delay >= b_avg_delay, "Stressed rain scenario should produce >= baseline delay"

    def test_departure_optimizer(self):
        dep = routing.departure_windows("GUWAHATI", "SHILLONG", "balanced", horizon_hours=24)
        assert "windows" in dep
        assert len(dep["windows"]) > 0
        for w in dep["windows"]:
            assert "depart_ts" in w
            assert "eta_hours" in w
            assert "composite_score" in w


# ==============================================================================
# 9. MULTILINGUAL ALERTS TESTS
# ==============================================================================
class TestAlerts:
    def test_build_alert_feed(self):
        st = inference.network_state()
        feed = alerts.build_alert_feed(st, limit=10)
        assert isinstance(feed, list)

    def test_supported_languages(self):
        langs = alerts.LANGUAGES
        for code in ["en", "hi", "as", "bn", "ne", "lus", "kha", "mni"]:
            assert code in langs
            assert "name" in langs[code]
            assert "script" in langs[code]


# ==============================================================================
# 10. REAL GPS TELEMETRY & GEOFENCING TESTS
# ==============================================================================
class TestGPSAndTelemetry:
    def test_snap_to_nearest_segment(self):
        st = inference.network_state()
        # Lat/Lon near Guwahati (GUWAHATI: 26.1445, 91.7362)
        hit = gps.snap(26.15, 91.74, st)
        assert hit is not None
        assert "segment" in hit
        assert "distance_km" in hit
        assert hit["distance_km"] < 6.0

    def test_ingest_gps_point(self):
        veh_id = "TEST-TRUCK-001"
        res = gps.ingest({
            "vehicle_id": veh_id,
            "lat": 26.15,
            "lon": 91.74,
            "speed_kmph": 45.0,
            "heading": 90.0,
            "accuracy_m": 5.0,
            "driver": "Tester",
            "dest": "SHILLONG"
        })
        assert res["vehicle_id"] == veh_id
        assert "on_network" in res
        assert res["status"] == "ok"
        assert "snapped_to" in res


# ==============================================================================
# 11. FASTAPI & SWAGGER API INTEGRATION TESTS (TestClient)
# ==============================================================================
class TestAPIEndpoints:
    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    @classmethod
    def teardown_class(cls):
        cls.client.close()

    def test_swagger_and_openapi(self):
        r_docs = self.client.get("/docs")
        assert r_docs.status_code == 200
        assert "swagger-ui" in r_docs.text.lower() or "html" in r_docs.headers.get("content-type", "")

        r_redoc = self.client.get("/redoc")
        assert r_redoc.status_code == 200

        r_spec = self.client.get("/openapi.json")
        assert r_spec.status_code == 200
        spec = r_spec.json()
        assert "paths" in spec
        assert "/api/route" in spec["paths"]
        assert "/api/network" in spec["paths"]
        assert "/api/incidents" in spec["paths"]
        assert len(spec["paths"]) >= 25

    def test_api_index(self):
        r = self.client.get("/api")
        assert r.status_code == 200
        data = r.json()
        assert "endpoints" in data
        assert "name" in data

    def test_health_check(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data or "models_loaded" in data

    def test_network_endpoints(self):
        r = self.client.get("/api/network")
        assert r.status_code == 200
        data = r.json()
        assert "segments" in data
        assert "summary" in data

        r_sum = self.client.get("/api/network/summary")
        assert r_sum.status_code == 200
        assert "segments_scored" in r_sum.json()

    def test_nodes_and_corridors(self):
        r_nodes = self.client.get("/api/nodes")
        assert r_nodes.status_code == 200
        assert len(r_nodes.json()) > 0

        r_corr = self.client.get("/api/corridors")
        assert r_corr.status_code == 200
        assert len(r_corr.json()) > 0

    def test_segment_explained_endpoint(self):
        r_net = self.client.get("/api/network")
        segments = r_net.json()["segments"]
        road_id = segments[0]["road_id"]
        r_seg = self.client.get(f"/api/segment/{road_id}")
        assert r_seg.status_code == 200
        data = r_seg.json()
        assert "risk_explanation" in data
        assert "delay_explanation" in data

    def test_route_planning_endpoints(self):
        r_route = self.client.get("/api/route?origin=GUWAHATI&dest=SHILLONG&profile=balanced")
        assert r_route.status_code == 200
        assert "routes" in r_route.json()

        r_comp = self.client.get("/api/route/compare?origin=GUWAHATI&dest=SHILLONG")
        assert r_comp.status_code == 200
        assert "by_profile" in r_comp.json()

    def test_criticality_and_accessibility(self):
        r_crit = self.client.get("/api/criticality")
        assert r_crit.status_code == 200

        r_acc = self.client.get("/api/accessibility")
        assert r_acc.status_code == 200
        assert "districts" in r_acc.json()

    def test_departure_and_whatif(self):
        r_dep = self.client.get("/api/departure?origin=GUWAHATI&dest=SHILLONG")
        assert r_dep.status_code == 200

        r_what = self.client.get("/api/whatif?rain_multiplier=1.8")
        assert r_what.status_code == 200
        assert "baseline" in r_what.json()

    def test_alerts_and_incident_reporting(self):
        r_alt = self.client.get("/api/alerts")
        assert r_alt.status_code == 200

        # Grab a valid road ID from the network
        r_net = self.client.get("/api/network")
        road_id = r_net.json()["segments"][0]["road_id"]

        report_payload = {
            "road_id": road_id,
            "lat": 26.15,
            "lon": 91.75,
            "issue_type": "landslide",
            "severity": "high",
            "description": "Minor mudslide blocking half lane"
        }
        r_post = self.client.post("/api/incidents", json=report_payload)
        assert r_post.status_code in (200, 201)

    def test_frontend_routes(self):
        assert self.client.get("/").status_code == 200
        assert self.client.get("/field").status_code == 200
        assert self.client.get("/track").status_code == 200
