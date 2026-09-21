"""
ONNX Runtime Deep Ensemble Wrappers for NER Logistics Sentinel:
  - NeuralRiskEnsemble: Evaluates 5-member HazardNet ensemble on CPU via ONNX Runtime
  - NeuralDelayEnsemble: Evaluates 5-member DelayNet ensemble with class conditioning
  - NeuralRouteEnsemble: Evaluates 5-member RouteNet ensemble for multi-segment route delays

Exposes exact scikit-learn API: predict, predict_proba, classes_ for drop-in compatibility.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from . import db
from .features_v2 import transform_v2_batch, transform_route_v2


class NeuralRiskEnsemble:
    """Drop-in replacement for risk_clf: evaluates 5-member HazardNet ONNX models."""
    def __init__(self, artifact_dir: str = db.ARTIFACT_DIR, n_members: int = 5):
        self.classes_ = np.array([0, 1, 2])
        self.sessions: List[ort.InferenceSession] = []
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        for i in range(n_members):
            path = os.path.join(artifact_dir, f"hazard_net_{i}.onnx")
            if os.path.exists(path):
                self.sessions.append(ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"]))

        if not self.sessions:
            raise FileNotFoundError(f"No hazard_net_*.onnx models found in {artifact_dir}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Input shape: (N, 23) -> Output shape: (N, 3)."""
        X_num, X_cat = transform_v2_batch(X)
        all_probs = []
        for sess in self.sessions:
            inputs = {"x_num": X_num, "x_cat": X_cat}
            haz, probs = sess.run(["hazard", "probabilities"], inputs)
            all_probs.append(probs)

        # Average ensemble probabilities
        ens_probs = np.mean(all_probs, axis=0)
        return ens_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_hazard_and_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_num, X_cat = transform_v2_batch(X)
        all_hazards = []
        for sess in self.sessions:
            inputs = {"x_num": X_num, "x_cat": X_cat}
            haz, probs = sess.run(["hazard", "probabilities"], inputs)
            all_hazards.append(haz)
        mean_haz = np.mean(all_hazards, axis=0)
        std_haz = np.std(all_hazards, axis=0)
        return mean_haz, std_haz


class NeuralDelayEnsemble:
    """Drop-in replacement for delay_reg: evaluates 5-member DelayNet ONNX models."""
    def __init__(self, risk_ensemble: NeuralRiskEnsemble, artifact_dir: str = db.ARTIFACT_DIR, n_members: int = 5):
        self.risk_ens = risk_ensemble
        self.sessions: List[ort.InferenceSession] = []
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1

        for i in range(n_members):
            path = os.path.join(artifact_dir, f"delay_net_{i}.onnx")
            if os.path.exists(path):
                self.sessions.append(ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_num, X_cat = transform_v2_batch(X)
        probs = self.risk_ens.predict_proba(X)
        mu, _ = self.risk_ens.predict_hazard_and_uncertainty(X)

        all_delays = []
        for sess in self.sessions:
            inputs = {"x_num": X_num, "x_cat": X_cat, "mu": mu, "probs": probs}
            mean_d, p50_d, p90_d = sess.run(["mean_delay", "p50_delay", "p90_delay"], inputs)
            all_delays.append(mean_d)

        return np.maximum(0.0, np.mean(all_delays, axis=0))


class NeuralRouteEnsemble:
    """Drop-in replacement for route_delay_reg: evaluates 5-member RouteNet ONNX models."""
    def __init__(self, artifact_dir: str = db.ARTIFACT_DIR, n_members: int = 5):
        self.sessions: List[ort.InferenceSession] = []
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1

        for i in range(n_members):
            path = os.path.join(artifact_dir, f"route_net_{i}.onnx")
            if os.path.exists(path):
                self.sessions.append(ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"]))

    def predict(self, X_route: np.ndarray) -> np.ndarray:
        X_tf = transform_route_v2(X_route)
        all_delays = []
        for sess in self.sessions:
            inputs = {"x_route": X_tf}
            mean_rd, p90_rd = sess.run(["mean_route_delay", "p90_route_delay"], inputs)
            all_delays.append(mean_rd)
        return np.maximum(0.0, np.mean(all_delays, axis=0))
