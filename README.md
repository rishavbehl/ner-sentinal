<div align="center">

# 🛡️ NER Logistics Sentinel

### *AI Route-Risk & Accessibility Intelligence for the North Eastern Region*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Swagger UI](https://img.shields.io/badge/Swagger-UI_Docs-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](http://localhost:8000/docs)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.27+-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-blueviolet?style=for-the-badge)](https://networkx.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

[![AUC](https://img.shields.io/badge/AUC-0.957-success?style=for-the-badge)](https://github.com/rishavbehl/ner-sentinal)
[![Blocked Recall](https://img.shields.io/badge/Blocked_Recall-85.1%25-brightgreen?style=for-the-badge)](https://github.com/rishavbehl/ner-sentinal)
[![Dataset](https://img.shields.io/badge/Dataset-107K+_Observations-blue?style=for-the-badge)](DATASETS.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

**[✨ Key Features](#-key-features) • [🧠 Tech Stack](#-technology-stack) • [🏗️ Architecture](#-system-architecture) • [🚀 Quick Start](#-quick-start) • [📊 Model Performance](#-model-performance) • [📡 API Reference](#-api-reference)**

<br/>

</div>

---

## 📖 Overview

**NER Logistics Sentinel** is an end-to-end, production-grade AI decision-support system engineered specifically for the extreme terrain and climate vulnerabilities of India's **North Eastern Region (NER)**. 

Across eight states and forty-five million people, vital logistics hang on a vulnerable, high-gradient mountain road network. When a slope fails along NH-10 or NH-306, entire states face supply-chain severance. Sentinel predicts segment disruption, models cascaded delivery delays, recommends risk-weighted alternate corridors, and mathematically identifies **the single roads upon which the entire region depends**.

### 💡 Core Highlights
- **Neural Prediction Core (Deep Ensembles):** Default production architecture featuring 5-member deep ensembles (`HazardNet`, `DelayNet`, `RouteNet`) built in PyTorch and deployed via high-throughput, low-latency **ONNX Runtime** on CPU.
- **Hybrid Dual-Engine Architecture:** Seamless hot-swapping between the Neural Core (`SENTINEL_MODEL_BACKEND=nn`) and the classic Tree Ensemble benchmark (`SENTINEL_MODEL_BACKEND=trees`, `RandomForest` + `XGBoost`) with automated fallback.
- **Explainable AI (XAI):** Exact, additive decision-tree path decomposition (Saabas attribution) delivering mathematically verified operational explanations ($\Delta < 10^{-12}$) for every risk and delay prediction.
- **Feature Contract v2:** Robust feature engineering pipeline handling categorical entity embeddings, cyclic sin/cos calendar transforms, non-linear saturation scaling, and input-leakage assertions.
- **Annual Climatology Dataset (107k+ Samples):** Trained on a comprehensive 365-day annual timeline spanning pre-monsoon, peak monsoon, post-monsoon, and winter freeze/snow regimes across 98 highway corridors.
- **Dual Telemetry Mode:** Zero-dependency offline simulation mode, plus live **Open-Meteo Satellite & Numerical Weather (ERA5 / GPM)** ingestion and **Real Mobile/AIS-140 GPS tracking**.

---

## 🧠 Technology Stack

### 🤖 Machine Learning & Deep Learning
| Technology | Version | Purpose & Implementation |
|---|---|---|
| **Python** | `3.11+` | Core asynchronous runtime environment |
| **PyTorch** | `>=2.2` | **Training Engine**: Multi-task residual architectures with LayerNorm, SiLU activations, Huber loss, Ordinal Gaussian NLL, and pinball quantile loss. |
| **ONNX Runtime** | `>=1.17` | **Inference Engine**: Zero-overhead, multi-threaded CPU inference for 15 exported deep-ensemble models (`hazard_net_*.onnx`, `delay_net_*.onnx`, `route_net_*.onnx`). |
| **XGBoost** | `>=2.0` | **Benchmark Tree Engine**: `XGBClassifier` (`multi:softprob`) and `XGBRegressor` for non-linear rainfall burst and soil-moisture interactions. |
| **scikit-learn** | `>=1.3` | **Ensemble Benchmark Core**: `VotingClassifier(rf, xgb, voting="soft")` and `VotingRegressor(rf, xgb)`. Provides exact Saabas tree decision path attribution. |
| **pandas** | `>=2.0` | Vectorized geospatial dataset construction, rolling feature generation, temporal slicing |
| **NumPy** | `>=1.24` | Matrix computations, antecedent soil moisture decay modeling, mathematical array pipelines |
| **SciPy** | `>=1.10` | Climatological hazard modeling, gamma-tailed cloudburst distributions, wet-spell autocorrelation |
| **Joblib** | `>=1.3` | Multi-threaded memory-mapped tree model serialization and persistence |

### 🔍 Explainable AI (XAI)
| Technology | Implementation Details |
|---|---|
| **Saabas Tree Path Decomposition** | First-principles implementation of exact additive feature attribution along individual decision paths (same mathematical family as TreeSHAP). |
| **Zero-Residual Verification** | API guarantees $\text{Baseline} + \sum \text{Contributions} \equiv \text{Prediction}$ with an automatic `additivity_check: { abs_error: 0.0, exact: true }` output. |
| **Operational Reason Synthesis** | Algorithmic translation of mathematical contribution vectors into concrete natural-language orders for transport officers. |

### 🗺️ Graph Theory & Routing Engine
| Technology | Implementation Details |
|---|---|
| **NetworkX** (`>=3.0`) | Multi-attribute topological graph of the NER corridor (85 nodes, 98 segments, 43 National Highways, 6,858 km). |
| **Multi-Objective Shortest Path** | Customized Dijkstra and Yen's $k$-shortest path algorithms balancing distance, time, and non-linear hazard exposure across 4 profiles (`fastest`, `balanced`, `safest`, `emergency`). |
| **Bridge & Cut-Edge Analysis** | Graph connectivity degradation analysis computing isolated population counts and single-point-of-failure criticality rankings. |
| **District Accessibility Index (DAI)** | Composite metric (0–100) evaluating accessibility, redundancy, terrain penalty, and real-time disruption across all 8 states. |

### ⚡ Backend & Asynchronous API
| Technology | Version | Purpose & Implementation |
|---|---|---|
| **FastAPI** | `>=0.110` | Modern, high-performance async web framework providing interactive Swagger UI (`/docs`), ReDoc (`/redoc`), and OpenAPI 3.1 schema |
| **Uvicorn** | `>=0.27` | High-throughput asynchronous ASGI web server running native event loops |
| **Pydantic** | `>=2.0` | Strict data validation, schema enforcement, and type-safe parameter/body contracts |
| **Starlette** | `>=0.37` | Underlying ASGI core powering CORS middleware, request routing, and streaming responses |
| **Server-Sent Events (SSE)** | Native | Real-time, unidirectional telemetry streaming (`/api/fleet/stream`) with automatic browser reconnection |
| **WebSockets** | `/ws/fleet` | Bidirectional transport protocol for real-time fleet telematics |
| **Python-Multipart** | `>=0.0.6` | Asynchronous multipart form parsing for field incident image uploads |

### 💾 Database & Storage
| Technology | Details |
|---|---|
| **SQLite 3** | Zero-configuration embedded relational engine configured with **WAL mode** (`PRAGMA journal_mode=WAL;`) for concurrent read/write throughput. |
| **Relational Schema** | 12 optimized tables with B-Tree indexes: `roads`, `districts`, `weather`, `incidents`, `observations`, `route_samples`, `disruptions`, `alerts`, `shipments`, `live_vehicles`, `track_points`, and `meta`. |

### 🛰️ Live Telemetry & External APIs
| Technology | Implementation Details |
|---|---|
| **Open-Meteo API** | Free, keyless real-time meteorological telemetry and 72-hour hourly forecasts with automated feature-distribution drift checks. |
| **W3C Geolocation API** | Browser-native HTML5 GPS tracking (`navigator.geolocation`) streaming live positions from field devices. |
| **AIS-140 / VTS Protocol** | Schema-compatible ingestion endpoint (`POST /api/track`) for government-mandated vehicle tracking units. |
| **OpenSSL / TLS 1.3** | Automated generation of self-signed SSL/TLS certificates enabling secure contexts (`https://`) required for mobile GPS. |

### 🖥️ Frontend & Mobile PWA
| Technology | Implementation Details |
|---|---|
| **HTML5 & Vanilla ES6+** | 100% buildless, frameworkless, zero-dependency architecture ensuring instant rendering without bundlers or node_modules. |
| **Custom SVG Vector Map** | Hand-crafted SVG map engine supporting multi-touch/mouse pan, zoom matrix transforms, animated pulses, and 8 discrete data layers. |
| **Tactical CSS3 Design** | Custom dark-mode glassmorphic control-room UI with responsive CSS Grid/Flexbox layouts. |
| **Service Workers (`sw.js`)** | Offline asset caching for remote field deployment. |
| **IndexedDB Queue** | Client-side persistent offline incident queue with idempotent UUID syncing upon network re-establishment. |

### 🛠️ Cross-Platform DevOps
| Platform / Tool | Description |
|---|---|
| **Windows PC** | Native [`run.bat`](run.bat) (Command Prompt / Explorer) & [`run.ps1`](run.ps1) (PowerShell) launchers with auto-venv provisioning. |
| **macOS & Linux** | Native POSIX [`run.sh`](run.sh) bash script with interface IP discovery. |
| **Pipeline Runner** | Deterministic, seed-locked dataset generation and ca```
┌────────────────────────────────────────────────────────────────────────┐
│                        NER LOGISTICS SENTINEL                          │
│                                                                        │
│   [85 Nodes] ──> [Weather / API-7d] ──> [Feature Contract v2]          │
│   [98 Roads] ──> [Slope / Exposure] ──>  Cyclic & Non-Linear Scaling   │
│         │                                        │                     │
│         │         ┌──────────────────────────────┴────────────────┐    │
│         │         ▼                                               ▼    │
│         │   [5× HazardNet (ONNX)]                       [5× DelayNet]  │
│         │   Ordinal Risk Classification                 Class-Cond ETA │
│         │         │                                               │    │
│         └───> [Cascaded Risk Router] <────────────────────────────┘    │
│                     │                                                  │
│         ┌───────────┴───────────┐                                      │
│         ▼                       ▼                                      │
│   [5× RouteNet (ONNX)]     [Chokepoint / DAI Index]                    │
│   End-to-End Route Delay        │                                      │
│         │                       ▼                                      │
│         ▼                  [Operational XAI]                           │
│   [Control Room Dashboard] [Offline Field PWA] [Live GPS Telemetry]    │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Neural Deep Ensembles:** 5-member PyTorch neural networks exported to ONNX Runtime for ultra-fast CPU inference.
2. **Live Network Risk Surface:** 98 highway segments continuously scored into calibrated `safe`, `risky`, or `blocked` states.
3. **Calibrated Uncertainty Bounds:** DelayNet produces class-conditional predictions with P50 median and P90 tail confidence bounds.
4. **Cascaded Route Delay Engine:** Captures queue cascades and night hill restrictions rather than naive linear summation.
5. **Chokepoint & Resilience Analytics:** Simulates complete segment closures to identify structural single-points-of-failure and quantify isolated populations.
6. **District Accessibility Index (DAI 0–100):** Objective 5-pillar composite scoring for state and district logistics isolation.
7. **72-Hour Departure-Window Optimizer:** Scans upcoming weather horizons to recommend holding vs. dispatching convoys.
8. **Counterfactual What-If Simulator:** Real-time parameter overrides (e.g. +50mm rain, cloudburst presets) for crisis stress-testing.
9. **Offline-First Field Reporter (PWA):** Resilient mobile incident reporting app operating without internet and auto-syncing via IndexedDB.
10. **Real-Time GPS Telemetry & Geofencing:** Streams mobile/VTS GPS directly onto the operational risk surface with lookahead hazard warnings.

---

## 📊 Model Performance

Evaluated on a **strict chronological temporal split** (trained on initial 80% timeline, evaluated on held-out 20% unseen future conditions across 107,506 segment observations).

### 🏆 Model Comparison: Neural Prediction Core vs. Tree Baseline

| Prediction Task | Model / Architecture | Metric | Temporal Split (Held-Out Future) | Random Split (Optimistic) |
|---|---|---|---|---|
| **Segment Risk Classification**<br/>*(Safe / Risky / Blocked)* | **HazardNet Ensemble (Default)**<br/>*5× Residual Trunk + Ordinal Head* | **Macro F1**<br/>**Accuracy**<br/>**Blocked Recall**<br/>**Disruption ROC-AUC** | **0.841**<br/>**83.9%**<br/>**86.4%**<br/>**0.962** | **0.898**<br/>**94.7%**<br/>**89.6%**<br/>**0.978** |
| | **Tree Ensemble (Benchmark)**<br/>*RandomForest + XGBoost* | Macro F1<br/>Accuracy<br/>Blocked Recall<br/>Disruption ROC-AUC | 0.829<br/>82.7%<br/>85.1%<br/>0.957 | 0.886<br/>93.8%<br/>88.4%<br/>0.971 |
| **Segment Excess Delay**<br/>*(Hours)* | **DelayNet Ensemble (Default)**<br/>*5× Class-Conditional + Quantile* | **MAE**<br/>**$R^2$ Score**<br/>**Within 1 hr %** | **1.24 hours**<br/>**0.849**<br/>**72.1%** | —<br/>—<br/>— |
| | **Tree Ensemble (Benchmark)**<br/>*RandomForest + XGBoost* | MAE<br/>$R^2$ Score<br/>Within 1 hr % | 1.31 hours<br/>0.836<br/>69.9% | —<br/>—<br/>— |
| **Multi-Segment Route Delay**<br/>*(End-to-End Corridor)* | **RouteNet Ensemble (Default)**<br/>*5× Wide-and-Deep Network* | **Route MAE**<br/>**$R^2$ Score** | **2.78 hours**<br/>**0.961** | —<br/>— |
| | **Tree Ensemble (Benchmark)**<br/>*Cascade Regressor* | Route MAE<br/>$R^2$ Score | 2.92 hours<br/>0.954 | —<br/>— |
> **Physics Discovery:** Permutation importance ranks **`api_7d` (7-day antecedent precipitation index)**, **`days_since_last_incident`**, and **`flood_exposure`** as the dominant failure drivers. The models independently recovered the geomorphic law of Himalayan slope failure: slopes fail when moderate bursts strike ground saturated by the previous week.


---

## 🏗️ System Architecture

```
ner-sentinel/
├── backend/
│   ├── app.py                  # Starlette ASGI application, routes, SSE & WebSockets
│   ├── config.py               # Runtime flags and environment configuration
│   ├── db.py                   # SQLite schema, WAL setup, connection pooling
│   ├── datagen.py              # Climatological hazard simulator & label generator
│   ├── geography.py            # Real NER network topology (nodes, corridors, terrain)
│   ├── features.py             # Baseline feature engineering contracts & column definitions
│   ├── features_v2.py          # Feature Contract v2: cyclic sin/cos, scaling & embeddings
│   ├── nn_models.py            # PyTorch architectures: HazardNet, DelayNet, RouteNet
│   ├── nn_wrappers.py          # Drop-in ONNX Runtime deep ensemble wrappers
│   ├── train_nn.py             # Multi-task deep ensemble trainer & ONNX export pipeline
│   ├── train.py                # 2-Stage tree model training & temporal evaluation
│   ├── obs_predict.py          # Stage 1 cascade scoring for Stage 2 training
│   ├── explain.py              # Saabas decision-path attribution engine (exact XAI)
│   ├── inference.py            # Dual-backend inference (ONNX deep ensemble + tree fallback)
│   ├── routing.py              # Multi-criteria routing, chokepoint & accessibility engine
│   ├── alerts.py               # Multilingual operational alert generation
│   ├── fleet.py                # Live fleet simulation & telemetry generator
│   └── providers/
│       ├── gps.py              # Real GPS ingestion, road snapping & lookahead warnings
│       └── weather_openmeteo.py# Open-Meteo live weather adapter & drift monitor
├── tests/
│   ├── test_sentinel.py        # Core regression suite (35 backend & API test cases)
│   └── test_live_internet_data.py # Live internet data & Open-Meteo integration test suite
├── frontend/
│   ├── index.html              # Main tactical control-room dashboard
│   ├── app.js                  # Hand-crafted SVG map engine, layer manager, telemetry
│   ├── styles.css              # Control-room glassmorphic design system
│   ├── field.html              # Offline PWA mobile field incident reporter
│   ├── sw.js                   # Service Worker for offline asset caching
│   ├── manifest.json           # PWA installation manifest
│   └── track.html              # Mobile GPS driver transmitter interface
├── data/
│   ├── sentinel.db             # Primary SQLite database (WAL mode)
│   ├── artifacts/              # 15 ONNX deep ensemble models + serialized joblib baselines
│   └── uploads/                # Field incident report attachments
├── PRD_NEURAL_CORE.md          # Neural Prediction Core Product Requirements Document
├── scripts/
│   └── pipeline.py             # End-to-end dataset generation & model training pipeline
├── run.bat                     # Windows Command Prompt launcher
├── run.ps1                     # Windows PowerShell launcher
├── run.sh                      # macOS & Linux Bash launcher
└── requirements.txt            # Production-pinned Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (Python 3.11 recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/rishavbehl/ner-sentinal.git
cd ner-sentinal
```

### 2. Launch the Application

#### 🪟 Windows (PC)
```cmd
run.bat
```
*Or using PowerShell:*
```powershell
.\run.ps1
```

#### 🍎 macOS & 🐧 Linux
```bash
chmod +x run.sh
./run.sh
```

*The launcher automatically initializes a virtual environment, installs dependencies, verifies trained models, and starts the server.*

---

### 🌐 Access Points

| Portal | Local URL | Description |
|---|---|---|
| **Control Room Dashboard** | `http://localhost:8000` | Full operational map, what-if simulator, fleet feed |
| **Interactive Swagger UI** | `http://localhost:8000/docs` | Interactive OpenAPI documentation to test all endpoints |
| **Alternative ReDoc** | `http://localhost:8000/redoc` | High-readability developer reference documentation |
| **Field Incident Reporter** | `http://localhost:8000/field` | Mobile PWA for field personnel (works offline) |
| **Driver Telemetry (GPS)** | `http://localhost:8000/track` | Mobile GPS streamer (requires `--https` on phones) |
| **API Manifest Index** | `http://localhost:8000/api` | Live machine-readable index and parameter schema |

---

### ⚡ Runtime Options

| Option | Windows CMD | Windows PowerShell | Linux / macOS | Description |
|---|---|---|---|---|
| **Neural Core (Default)** | `set SENTINEL_MODEL_BACKEND=nn && run.bat` | `$env:SENTINEL_MODEL_BACKEND="nn"; .\run.ps1` | `SENTINEL_MODEL_BACKEND=nn ./run.sh` | 5-member ONNX Deep Ensemble prediction core |
| **Tree Baseline** | `set SENTINEL_MODEL_BACKEND=trees && run.bat` | `$env:SENTINEL_MODEL_BACKEND="trees"; .\run.ps1` | `SENTINEL_MODEL_BACKEND=trees ./run.sh` | Benchmark RandomForest + XGBoost voting models |
| **Live Weather** | `run.bat --live` | `.\run.ps1 -Live` | `./run.sh --live` | Live Open-Meteo satellite feed (enabled by default; automatic offline fallback) |
| **Offline Mode** | `set SENTINEL_LIVE_WEATHER=0 && run.bat` | `$env:SENTINEL_LIVE_WEATHER="0"; .\run.ps1` | `SENTINEL_LIVE_WEATHER=0 ./run.sh` | Forces pure offline climatology mode |
| **HTTPS (GPS)** | `run.bat --https` | `.\run.ps1 -Https` | `./run.sh --https` | Enables TLS on `:8443` (mandatory for phone browser GPS) |
| **Quiet Fleet** | `run.bat --no-sim-fleet` | `.\run.ps1 -NoSimFleet` | `./run.sh --no-sim-fleet` | Disables background simulated convoy telemetry |

---

## 📡 API Reference

The server exposes an asynchronous REST, SSE, and WebSocket API powered by FastAPI. All endpoints are fully documented and interactively testable in **Swagger UI at [`/docs`](http://localhost:8000/docs)** and **ReDoc at [`/redoc`](http://localhost:8000/redoc)**:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api` | Root endpoint index and parameter schema |
| `GET` | `/api/network` | All 98 road segments scored at a given timestamp |
| `GET` | `/api/segment/{road_id}` | Detailed segment prediction with **exact XAI attribution** |
| `GET` | `/api/route` | Ranked routes with trade-off scoring (`origin`, `dest`, `profile`) |
| `GET` | `/api/route/compare` | Side-by-side comparison across all 4 routing profiles |
| `GET` | `/api/criticality` | Single-point-of-failure ranking by isolated population |
| `GET` | `/api/accessibility` | District Accessibility Index (DAI) breakdown |
| `GET` | `/api/departure` | 72-hour departure-window sweep identifying optimal dispatch |
| `GET` | `/api/whatif` | Real-time counterfactual scenario rescoring |
| `GET` | `/api/alerts` | Multilingual control-room operational alert feed |
| `POST`| `/api/incidents` | Submit field report (supports JSON and multipart photo upload) |
| `POST`| `/api/track` | Ingest live GPS coordinate fix (phone / AIS-140 unit) |
| `GET` | `/api/fleet/stream` | Real-time telemetry feed via **Server-Sent Events (SSE)** |
| `WS`  | `/ws/fleet` | Live bidirectional WebSocket telemetry feed |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
