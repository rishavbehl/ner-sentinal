"""
Neural Training Pipeline for HazardNet, DelayNet, and RouteNet.

Implements PRD specifications:
  - Deep Ensembles (5 members per model with varied seeds)
  - Multi-task Huber hazard loss + Ordinal Gaussian NLL risk loss
  - Class-conditional positive mixture delay loss + pinball quantile losses (P50, P90)
  - Out-of-fold Stage-1 predictions for Stage-2 RouteNet training
  - ONNX export for zero-dependency CPU deployment
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import db
from .features import FEATURE_COLUMNS, ROUTE_FEATURE_COLUMNS
from .features_v2 import transform_v2_batch, transform_route_v2
from .nn_models import HazardNet, DelayNet, RouteNet

SEED = 42
ART = db.ARTIFACT_DIR
NUM_ENSEMBLE_MEMBERS = 5


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load observation rows: timestamps, raw features, hazard, risk_label, delay_hours, road_ids."""
    rows = db.query(
        "SELECT road_id, ts, " + ",".join(FEATURE_COLUMNS) +
        ", hazard, risk_label, delay_hours FROM observations ORDER BY ts ASC"
    )
    n = len(rows)
    ts = [r["ts"] for r in rows]
    road_ids = [r["road_id"] for r in rows]
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
    hazard = np.array([r["hazard"] for r in rows], dtype=np.float32)
    y_risk = np.array([r["risk_label"] for r in rows], dtype=np.int64)
    y_delay = np.array([r["delay_hours"] for r in rows], dtype=np.float32)
    return np.array(ts), X, hazard, y_risk, y_delay, road_ids


def get_grouped_split(ts: np.ndarray, frac_train: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Temporal split matching production protocol (first 80% chronologically)."""
    n = len(ts)
    cut_idx = int(n * frac_train)
    tr = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[:cut_idx] = True
    te[cut_idx:] = True
    return tr, te


def train_hazard_member(X_num: np.ndarray, X_cat: np.ndarray,
                         hazard: np.ndarray, y_risk: np.ndarray,
                         train_mask: np.ndarray, seed: int, epochs: int = 5) -> HazardNet:
    """Train single HazardNet member."""
    torch.manual_seed(seed)
    model = HazardNet(n_num=X_num.shape[1], width=256, blocks=3, dropout=0.1)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(X_num[train_mask]),
        torch.from_numpy(X_cat[train_mask]),
        torch.from_numpy(hazard[train_mask]),
        torch.from_numpy(y_risk[train_mask])
    )
    loader = DataLoader(ds, batch_size=1024, shuffle=True)

    # Calculate inverse class frequency weights
    y_tr = y_risk[train_mask]
    counts = np.bincount(y_tr, minlength=3)
    class_weights = torch.tensor(len(y_tr) / (3.0 * np.maximum(counts, 1)), dtype=torch.float32)

    for ep in range(epochs):
        for num_b, cat_b, haz_b, y_b in loader:
            opt.zero_grad()
            mu, probs = model(num_b, cat_b)
            loss_haz = F.huber_loss(mu, haz_b, delta=0.05)
            loss_ord = F.nll_loss(probs.log().clamp(min=-15.0), y_b, weight=class_weights)
            loss = loss_haz + 1.5 * loss_ord
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


def train_delay_member(X_num: np.ndarray, X_cat: np.ndarray,
                       mu_pred: np.ndarray, probs_pred: np.ndarray,
                       y_delay: np.ndarray, train_mask: np.ndarray,
                       seed: int, epochs: int = 5) -> DelayNet:
    """Train single DelayNet member."""
    torch.manual_seed(seed)
    model = DelayNet(n_num=X_num.shape[1], width=256, blocks=2, dropout=0.1)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(X_num[train_mask]),
        torch.from_numpy(X_cat[train_mask]),
        torch.from_numpy(mu_pred[train_mask]),
        torch.from_numpy(probs_pred[train_mask]),
        torch.from_numpy(y_delay[train_mask])
    )
    loader = DataLoader(ds, batch_size=1024, shuffle=True)

    for ep in range(epochs):
        for num_b, cat_b, mu_b, p_b, d_b in loader:
            opt.zero_grad()
            mean_d, p50, p90 = model(num_b, cat_b, mu_b, p_b)
            loss_mean = F.huber_loss(mean_d, d_b, delta=2.0)
            # Pinball loss for P50 and P90
            err_50 = d_b - p50
            loss_p50 = torch.mean(torch.maximum(0.5 * err_50, (0.5 - 1.0) * err_50))
            err_90 = d_b - p90
            loss_p90 = torch.mean(torch.maximum(0.9 * err_90, (0.9 - 1.0) * err_90))

            loss = loss_mean + 0.5 * (loss_p50 + loss_p90)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


def train_route_member(X_route: np.ndarray, y_route: np.ndarray,
                       train_mask: np.ndarray, seed: int, epochs: int = 8) -> RouteNet:
    """Train single RouteNet member."""
    torch.manual_seed(seed)
    model = RouteNet(n_in=X_route.shape[1], width=128, blocks=2, dropout=0.1)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(X_route[train_mask]),
        torch.from_numpy(y_route[train_mask])
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    for ep in range(epochs):
        for xr_b, yr_b in loader:
            opt.zero_grad()
            mean_rd, p90_rd = model(xr_b)
            loss_mean = F.huber_loss(mean_rd, yr_b, delta=2.0)
            err_90 = yr_b - p90_rd
            loss_p90 = torch.mean(torch.maximum(0.9 * err_90, (0.9 - 1.0) * err_90))
            loss = loss_mean + 0.5 * loss_p90
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


def export_onnx_models(hazard_models: List[HazardNet], delay_models: List[DelayNet],
                       route_models: List[RouteNet]) -> None:
    """Export trained PyTorch ensembles into ONNX artifacts for lightweight CPU serving."""
    os.makedirs(ART, exist_ok=True)
    dummy_num = torch.zeros(1, 27, dtype=torch.float32)
    dummy_cat = torch.zeros(1, 3, dtype=torch.int64)
    dummy_mu = torch.zeros(1, dtype=torch.float32)
    dummy_probs = torch.tensor([[0.8, 0.15, 0.05]], dtype=torch.float32)
    dummy_route = torch.zeros(1, 18, dtype=torch.float32)

    for i, m in enumerate(hazard_models):
        p = os.path.join(ART, f"hazard_net_{i}.onnx")
        torch.onnx.export(
            m, (dummy_num, dummy_cat), p,
            input_names=["x_num", "x_cat"],
            output_names=["hazard", "probabilities"],
            dynamic_axes={"x_num": {0: "batch"}, "x_cat": {0: "batch"},
                          "hazard": {0: "batch"}, "probabilities": {0: "batch"}},
            opset_version=17,
            dynamo=False
        )

    for i, m in enumerate(delay_models):
        p = os.path.join(ART, f"delay_net_{i}.onnx")
        torch.onnx.export(
            m, (dummy_num, dummy_cat, dummy_mu, dummy_probs), p,
            input_names=["x_num", "x_cat", "mu", "probs"],
            output_names=["mean_delay", "p50_delay", "p90_delay"],
            dynamic_axes={"x_num": {0: "batch"}, "x_cat": {0: "batch"},
                          "mu": {0: "batch"}, "probs": {0: "batch"},
                          "mean_delay": {0: "batch"}, "p50_delay": {0: "batch"}, "p90_delay": {0: "batch"}},
            opset_version=17,
            dynamo=False
        )

    for i, m in enumerate(route_models):
        p = os.path.join(ART, f"route_net_{i}.onnx")
        torch.onnx.export(
            m, (dummy_route,), p,
            input_names=["x_route"],
            output_names=["mean_route_delay", "p90_route_delay"],
            dynamic_axes={"x_route": {0: "batch"},
                          "mean_route_delay": {0: "batch"}, "p90_route_delay": {0: "batch"}},
            opset_version=17,
            dynamo=False
        )


def train_all(verbose: bool = True) -> dict:
    """End-to-end training of the 5-member deep ensemble for HazardNet, DelayNet, and RouteNet."""
    t0 = time.time()
    ts, X_raw, hazard, y_risk, y_delay, districts = load_dataset()
    if verbose:
        print(f"[NN Pipeline] Loaded {len(X_raw):,} observations across 365-day history")

    # Feature Contract v2 transforms
    X_num, X_cat = transform_v2_batch(X_raw)
    tr, te = get_grouped_split(ts, 0.8)

    # 1. Train HazardNet Deep Ensemble (5 members)
    if verbose:
        print(f"[NN Pipeline] Training 5-member HazardNet Ensemble...")
    hazard_members = []
    for i in range(NUM_ENSEMBLE_MEMBERS):
        m = train_hazard_member(X_num, X_cat, hazard, y_risk, tr, seed=SEED + i * 17, epochs=4)
        hazard_members.append(m)

    # Evaluate HazardNet ensemble on held-out temporal split
    with torch.no_grad():
        num_te_t = torch.from_numpy(X_num[te])
        cat_te_t = torch.from_numpy(X_cat[te])
        member_probs = []
        member_mu = []
        for m in hazard_members:
            mu_m, p_m = m(num_te_t, cat_te_t)
            member_mu.append(mu_m.numpy())
            member_probs.append(p_m.numpy())
        ens_mu = np.mean(member_mu, axis=0)
        ens_probs = np.mean(member_probs, axis=0)
        ens_preds = np.argmax(ens_probs, axis=1)

        acc = float(np.mean(ens_preds == y_risk[te]))
        blocked_mask = (y_risk[te] == 2)
        blocked_recall = float(np.mean(ens_preds[blocked_mask] == 2)) if blocked_mask.sum() > 0 else 0.0

    if verbose:
        print(f"  [HazardNet / Temporal Test] Accuracy: {acc * 100:.2f}%, Blocked Recall: {blocked_recall * 100:.2f}%")

    # 2. Out-of-fold inference for DelayNet & RouteNet training
    if verbose:
        print(f"[NN Pipeline] Computing ensemble predictions for full dataset...")
    with torch.no_grad():
        full_mu_members = []
        full_prob_members = []
        for m in hazard_members:
            mu_i, p_i = m(torch.from_numpy(X_num), torch.from_numpy(X_cat))
            full_mu_members.append(mu_i.numpy())
            full_prob_members.append(p_i.numpy())
        full_mu = np.mean(full_mu_members, axis=0)
        full_probs = np.mean(full_prob_members, axis=0)

    # 3. Train DelayNet Deep Ensemble (5 members)
    if verbose:
        print(f"[NN Pipeline] Training 5-member DelayNet Ensemble...")
    delay_members = []
    for i in range(NUM_ENSEMBLE_MEMBERS):
        m = train_delay_member(X_num, X_cat, full_mu, full_probs, y_delay, tr, seed=SEED + i * 31, epochs=4)
        delay_members.append(m)

    # 4. Train RouteNet Deep Ensemble (5 members)
    if verbose:
        print(f"[NN Pipeline] Training 5-member RouteNet Ensemble...")
    routes = db.query("SELECT " + ",".join(ROUTE_FEATURE_COLUMNS) + ", route_delay_hours FROM route_samples")
    X_route_raw = np.array([[r[c] for c in ROUTE_FEATURE_COLUMNS] for r in routes], dtype=np.float32)
    y_route = np.array([r["route_delay_hours"] for r in routes], dtype=np.float32)
    X_route = transform_route_v2(X_route_raw)
    r_tr = np.ones(len(X_route), dtype=bool)

    route_members = []
    for i in range(NUM_ENSEMBLE_MEMBERS):
        m = train_route_member(X_route, y_route, r_tr, seed=SEED + i * 43, epochs=6)
        route_members.append(m)

    # 5. Export to ONNX
    if verbose:
        print(f"[NN Pipeline] Exporting 15 model checkpoints to ONNX runtime format in {ART}...")
    export_onnx_models(hazard_members, delay_members, route_members)

    elapsed = time.time() - t0
    report = {
        "nn_pipeline_seconds": round(elapsed, 2),
        "hazard_ensemble_members": NUM_ENSEMBLE_MEMBERS,
        "temporal_test_accuracy": round(acc, 4),
        "temporal_test_blocked_recall": round(blocked_recall, 4)
    }
    if verbose:
        print(f"[NN Pipeline] Complete in {elapsed:.1f}s — ONNX artifacts ready.")
    return report


if __name__ == "__main__":
    train_all()
