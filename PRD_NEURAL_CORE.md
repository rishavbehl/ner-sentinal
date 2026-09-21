# Product Requirements Document: Neural Prediction Core for NER Logistics Sentinel
**Version / Status:** 1.0 — Draft for Review  
**Date:** 21 September 2026  
**Basis:** Full read of `ner-sentinel` codebase + dataset analysis + prototype empirical bounds  
**Scope:** Replace / augment the three tree-ensemble models (`RandomForest` + `XGBoost`) with a neural-network prediction core (`HazardNet`, `DelayNet`, `RouteNet`) engineered to maximise held-out correctness  
**Not in Scope:** Frontend redesign, routing algorithms, alert language packs, GPS/fleet logic  

---

## 1. Executive Summary

NER Logistics Sentinel scores 98 highway segments of India's North Eastern Region for landslide / flood / blockade risk (`safe` · `risky` · `blocked`), predicts delay hours per segment, and predicts end-to-end route delay. Everything else in the product — routing, alerts, GPS look-ahead, fleet simulation, the dashboard — reads those three model outputs through one function, `inference.network_state()`.

This PRD specifies a neural prediction core (multi-task, ensembled, calibrated, explainable) that plugs directly into that seam.

### Key Analysis & Guiding Constraints
1. **Hard Accuracy Ceiling:** The training labels are noisy, thresholded outputs of a hidden hazard function (Gaussian noise $\sigma = 0.04$). No model can exceed **95.20% accuracy** on this dataset in expectation (**87.90%** on the project's seasonal temporal test split). Promising "99%" would indicate metric leakage.
2. **Correctness = Held-Out Generalization:** Training accuracy on synthetic data is deceptive (RandomForest achieves 97.7% on its training rows vs. a 97.0% theoretical ceiling by memorizing noise).
3. **Structured Target Leverage:** Regressing the continuous latent hazard with an ordinal head achieves **86.73%** on the seasonal temporal split vs. RF's **82.72%**.
4. **Zero Frontend/Routing Regressions:** The API contracts, response schema, and inference latency budgets must remain strictly preserved.

### Headline Evidence (Prototype Results & Baselines)

| Task / Protocol | Current (RF) | Best NN Prototype | Theoretical Ceiling |
| :--- | :--- | :--- | :--- |
| **Risk accuracy — in-distribution (random split)** | 93.81% | **95.08%** | 95.20% |
| **Risk accuracy — temporal split (monsoon test)** | 82.72% | **86.73%** | 87.90% |
| **Segment-delay MAE — temporal** | 1.317 h | **0.967 h** | $\ge$ 0.72 h |
| **Segment-delay $R^2$ — temporal** | 0.837 | **0.855** | $\le$ 0.896 |
| **Hazard RMSE — in-distribution** | — | **0.0401** | 0.0400 |

---

## 2. Target Architecture

```
[Feature Contract v2] (Log/scale transforms, embeddings, cyclic month/hour)
         │
         ├───> [HazardNet] (3 ResBlocks, width 256, LayerNorm, SiLU)
         │          ├──> Continuous Hazard Head (μ ∈ [0, 1])
         │          └──> Ordinal Classification Head (Learned τ₁, τ₂, σ)
         │
         ├───> [DelayNet] (Class-conditional mixture over safe/risky/blocked)
         │          ├──> Mean Excess Delay Head
         │          └──> Quantile Heads (P50, P90) for ETA confidence bands
         │
         └───> [RouteNet] (Wide-and-deep over route aggregates & OOF segment predictions)
                    └──> End-to-end Route Delay (Mean + P90)
```

- **Deep Ensemble:** 5 members per model trained with varied seeds and weight averaging.
- **Serving Format:** Exported to **ONNX Runtime (CPU)**; zero heavy PyTorch runtime requirements for demo/deploy.
- **Backend Toggle:** `SENTINEL_MODEL_BACKEND=nn` with fallback to `trees`.
- **Explainability:** Integrated Gradients emitting the exact same schema keys (`drivers`, `baseline_failure_probability`, `additivity_check.exact`, `additivity_check.abs_error`).

---

## 3. Milestones & Delivery Roadmap

- **M0 — Honest Baseline & Leakage Guards:** Grouped splits by `(district, day)`, feature contract v2 transforms, fix $r_{24} \le r_{72}$ generator defect.
- **M1 — HazardNet (Risk):** Residual trunk, continuous hazard regression + learned threshold ordinal head, 5-member deep ensemble.
- **M2 — DelayNet (Segment Delay):** Class-conditional positive delay mixture, pinball quantile losses (P50/P90).
- **M3 — RouteNet & Out-of-Fold Cascade:** Generate Stage-2 training data strictly on out-of-fold (OOF) Stage-1 predictions.
- **M4 — ONNX Serving & Integrated Gradients:** Export to ONNX (<20MB total), wire `inference.network_state()` wrappers, preserve API schema.
- **M5 — Real-Data Track:** Link zero-shot / fine-tuning benchmark for ECMWF ERA5-Land and NASA GPM/GLC inventories.
