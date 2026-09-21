"""
Integration test using live real-world data from the internet (Open-Meteo).

Pulls real atmospheric conditions across 8 key North Eastern Region hubs:
  Guwahati, Shillong, Siliguri, Imphal, Aizawl, Agartala, Gangtok, Tawang.
Updates the database and executes full-stack inference and routing across:
  1. Neural Core (Deep Ensembles: HazardNet, DelayNet, RouteNet via ONNX)
  2. Tree Ensemble Baseline (RandomForest + XGBoost)
Asserts operational integrity, sensible prediction bounds, and exact attribution.
"""
from __future__ import annotations

import os
import pytest
import numpy as np

from backend import db, inference, routing
from backend.providers import weather_openmeteo
from backend.explain import explain_segment_risk, explain_delay


@pytest.fixture(scope="module")
def live_data():
    """Ensure live data from Open-Meteo is available in the database."""
    db.init_db()
    rows = db.query("SELECT COUNT(*) as cnt FROM weather WHERE is_forecast = 0")
    if rows[0]["cnt"] == 0:
        result = weather_openmeteo.refresh()
        assert result.get("ok") is True
        assert result.get("nodes", 0) > 0
    return True


def test_live_weather_ingestion(live_data):
    """Verify live internet data successfully ingested into SQLite."""
    rows = db.query("SELECT COUNT(*) as cnt FROM weather WHERE is_forecast = 0")
    assert rows[0]["cnt"] > 0, "Expected non-empty live weather observations"

    sample = db.query("SELECT * FROM weather WHERE is_forecast = 0 LIMIT 1")[0]
    assert "rainfall_24h_mm" in sample
    assert "temperature_c" in sample
    assert "api_7d" in sample
    assert sample["temperature_c"] is not None


def test_neural_inference_on_live_internet_data(live_data):
    """Test Neural Prediction Core inference on real-world internet data."""
    inference._MODELS = {}
    os.environ["SENTINEL_MODEL_BACKEND"] = "nn"

    state = inference.network_state()
    segments = state["segments"]
    assert len(segments) == 98, f"Expected 98 scored segments, got {len(segments)}"

    for seg_id, seg in segments.items():
        assert seg["risk_status"] in ("safe", "risky", "blocked")
        assert seg["risk_label"] in (0, 1, 2)
        assert 0.0 <= seg["severity"] <= 1.0
        assert seg["delay_hours"] >= 0.0
        assert len(seg["proba"]) == 3
        total_p = sum(seg["proba"])
        assert np.isclose(total_p, 1.0, atol=1e-2), f"Probabilities sum to {total_p}"

    summary = state["summary"]
    assert summary["segments_scored"] == 98
    assert summary["safe"] + summary["risky"] + summary["blocked"] == 98
    print(f"\n[Live Internet Test] Neural Core Summary: "
          f"{summary['safe']} safe, {summary['risky']} risky, "
          f"{summary['blocked']} blocked. Avg risk: {summary['mean_risk']:.3f}")


def test_tree_baseline_on_live_internet_data(live_data):
    """Test Tree Ensemble Baseline inference on real-world internet data."""
    inference._MODELS = {}
    os.environ["SENTINEL_MODEL_BACKEND"] = "tree"

    state = inference.network_state()
    segments = state["segments"]
    assert len(segments) == 98

    for seg_id, seg in segments.items():
        assert seg["risk_status"] in ("safe", "risky", "blocked")
        assert seg["risk_label"] in (0, 1, 2)
        assert seg["delay_hours"] >= 0.0

    summary = state["summary"]
    assert summary["segments_scored"] == 98
    print(f"\n[Live Internet Test] Tree Baseline Summary: "
          f"{summary['safe']} safe, {summary['risky']} risky, "
          f"{summary['blocked']} blocked. Avg risk: {summary['mean_risk']:.3f}")


def test_routing_with_live_internet_conditions(live_data):
    """Test full route planning using live internet weather conditions."""
    plan = routing.plan(
        origin="GUWAHATI",
        dest="TAWANG",
        profile_key="balanced",
    )
    assert "routes" in plan and len(plan["routes"]) > 0
    primary = plan["routes"][0]
    assert primary["distance_km"] > 0
    assert primary["predicted_delay_hours"] >= 0
    print(f"\n[Live Internet Test] Guwahati->Tawang planned: {primary['distance_km']} km, "
          f"ETA delay {primary['predicted_delay_hours']:.2f} h, status: {primary['status']}")


def test_explainability_on_live_internet_data(live_data):
    """Verify Saabas exact additive attribution on live internet observations."""
    inference._MODELS = {}
    os.environ["SENTINEL_MODEL_BACKEND"] = "nn"
    models = inference.models()

    state = inference.network_state()
    seg_id = state["order"][0]
    sample_seg = state["segments"][seg_id]
    fd = sample_seg["features"]
    from backend.features import FEATURE_COLUMNS
    x = np.array([fd[c] for c in FEATURE_COLUMNS], dtype=np.float64)

    # Explain risk
    exp_risk = explain_segment_risk(models["risk"], x)
    assert exp_risk["additivity_check"]["exact"] is True
    assert len(exp_risk["drivers"]) > 0

    # Explain delay
    exp_delay = explain_delay(models["delay"], x)
    assert exp_delay["additivity_check"]["exact"] is True
    assert len(exp_delay["drivers"]) > 0
    print(f"\n[Live Internet Test] Explainability: exact additivity verified on live segment {seg_id}")
