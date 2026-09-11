# NER Logistics Sentinel
### AI Route-Risk & Accessibility Intelligence for the North Eastern Region

Predicts road disruption, estimates delivery delay, recommends safer alternates,
and — the part nobody else builds — tells you **which single roads the whole
region hangs on**.

### Windows (PC)
```cmd
run.bat               # Command Prompt / Double-click
# or in PowerShell:
.\run.ps1
```

### macOS & Linux
```bash
./run.sh              # Bash launcher
```
*First run takes ~2 minutes (builds dataset and trains models). After that, the server launches in ~1 second.*

| View | Local URL | Notes |
|---|---|---|
| **Control Room Dashboard** | http://localhost:8000 | Hand-crafted SVG map, risk layers, what-if simulator |
| **Field Reporter (Mobile PWA)** | http://localhost:8000/field | Offline-first incident reporting with IndexedDB queue |
| **Driver Telemetry (Live GPS)** | http://localhost:8000/track | Streams device GPS to live risk surface (requires HTTPS) |
| **API Index** | http://localhost:8000/api | Complete interactive endpoint documentation & schemas |

**Zero external dependencies at runtime.** No external CDN, no map tiles, no mandatory internet connection.
The map is hand-rendered SVG. This runs entirely on a laptop with Wi-Fi switched off —
which is the point, both for a venue demo and for the remote North Eastern districts this serves.

---

## Technology Stack

### 🧠 Machine Learning & Data Science
- **scikit-learn (`>=1.3`)**:
  - **Model 1 (Segment Risk Classifier)**: Balanced `RandomForestClassifier` predicting segment status (`safe`, `risky`, `blocked`) with calibrated class probabilities.
  - **Model 2 (Segment Delay Regressor)**: `RandomForestRegressor` estimating excess transit delay hours on disrupted corridors.
  - **Model 3 (Cascaded Route Delay Regressor)**: 2-stage ensemble learning non-linear route-level delay, queue cascades, and checkpost dwell times.
  - **Evaluation & Validation**: Strict `TimeSeriesSplit` temporal evaluation (80% train / 20% test forward in time), Brier score calibration, ROC-AUC (`0.954`), and permutation importance.
- **pandas (`>=2.0`) & NumPy (`>=1.24`)**: Vectorized feature generation, climatological decay index computations, and antecedent precipitation index (API-7d) soil saturation modeling.
- **SciPy (`>=1.10`)**: Statistical distribution modelling for synthetic hazard models, gamma-tailed cloudburst simulations, and wet-spell autocorrelation.
- **Joblib (`>=1.3`)**: Memory-mapped model serialization and high-throughput model persistence.

### 🔍 Explainable AI (XAI)
- **Saabas Decision-Tree Path Decomposition**: Exact additive feature attribution algorithm developed from first principles (same family as TreeSHAP).
- **Zero-Residual Mathematical Verification**: On every prediction, the API verifies:
  $$\text{Baseline} + \sum \text{Feature Contributions} \equiv \text{Model Output} \quad (\text{error} < 10^{-12})$$
  Returning an `additivity_check: { abs_error: 0.0, exact: true }` guarantee.
- **Natural Language Reason Synthesis**: Dynamic domain-aware translation of SHAP-like attribution vectors into actionable operational sentences for control-room operators.

### 🗺️ Network Graph Theory & Routing Engine
- **NetworkX (`>=3.0`)**: Multi-attribute geographic graph representation of the entire North Eastern Region (85 nodes, 98 segments, 43 National Highway corridors, 6,858 km).
- **Risk-Aware Multi-Criteria Shortest Path Routing**: Customized Dijkstra and Yen's $k$-shortest path algorithm variants with multi-objective trade-off scoring across 4 profiles:
  - `fastest`: Minimizes expected arrival time.
  - `balanced`: Commercial haulage optimizing time vs. risk penalty.
  - `safest`: Exponential risk penalty minimizing landslide exposure.
  - `emergency`: Strict failure-probability thresholds for critical supplies (blood, vaccines, oxygen).
- **Single-Point-of-Failure & Chokepoint Analysis**: Graph edge-cut and bridge identification measuring population isolation and district accessibility loss.
- **District Accessibility Index (DAI 0–100)**: Composite accessibility metric computing reach, reliability, redundancy, terrain burden, and live disruption across all 8 states.

### ⚡ Backend & API Architecture
- **Python 3.9+ / 3.11**: Cross-platform asynchronous server core.
- **Starlette (`>=0.37`)**: High-performance ASGI framework powering async route dispatch, middleware, and streaming endpoints.
- **Uvicorn (`>=0.27`)**: Production ASGI web server running with native `asyncio` event loop.
- **Pydantic (`>=2.0`)**: Rigorous schema validation, settings management, and API contract enforcement.
- **Server-Sent Events (SSE)**: Unidirectional real-time event streaming (`/api/fleet/stream`) pushing live vehicle coordinates, alerts, and risk recalculations to dashboards without WebSocket proxy overhead.
- **WebSockets (`/ws/fleet`)**: Dual real-time bidirectional protocol option for fleet GPS & control room events.
- **Python-Multipart**: Asynchronous binary and multipart form processing for field incident photo uploads.

### 💾 Database & Storage
- **SQLite 3**: Embedded relational database configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), foreign key constraints, and multi-threaded connection management.
- **Normalized Schema (12 Tables & Indexes)**:
  - Infrastructure: `roads`, `districts`, `corridors`
  - Dynamic Environment: `weather`, `observations`, `disruptions`
  - Operations & Incidents: `incidents`, `alerts`, `shipments`
  - Real-Time Telemetry: `live_vehicles`, `track_points`, `meta`

### 🛰️ Live Telemetry & External Providers
- **Open-Meteo REST API**: Free, keyless live meteorological telemetry ingestion, hourly forecasts, and rolling antecedent soil moisture calculation with automatic distribution-drift checks.
- **W3C Geolocation API**: HTML5 `navigator.geolocation` live browser GPS streaming from mobile devices.
- **AIS-140 / VTS Standard Schema**: Vehicle tracking standard schema compatibility (`/api/track`).
- **OpenSSL / TLS 1.3**: Automatic self-signed SSL/TLS certificate generation for secure context HTTPS required by mobile GPS.

### 🖥️ Frontend & UI/UX
- **HTML5 & Vanilla ES6+ JavaScript**: 100% dependency-free, zero build step, zero npm packages, zero external font or CDN downloads.
- **Custom Hand-Crafted SVG Map Engine**:
  - Dynamic vector rendering of all 8 NER states with pan/zoom matrix transforms.
  - 8 distinct visual layers: Risk choropleth, National Highway corridors, real-time fleet, live GPS breadcrumbs, district hubs, weather overlays, terrain contours, and incident flags.
  - Custom pulsating keyframe status indicators.
- **CSS3 Design System**: Custom glassmorphism, responsive CSS Grid and Flexbox layouts, tactical dark-mode color scheme tailored for operations control centers.
- **Progressive Web App (PWA)**:
  - **Service Workers (`sw.js`)**: Offline caching of application assets.
  - **IndexedDB**: Client-side resilient offline queue storing field incident reports with UUID deduplication for automatic sync upon network reconnection.
  - **Web App Manifest (`manifest.json`)**: Mobile installable app experience on iOS and Android.

### 🛠️ Cross-Platform DevOps & Tooling
- **Windows (PC)**: `run.bat` (Command Prompt) & `run.ps1` (PowerShell) native launchers.
- **macOS & Linux**: `run.sh` Bash launcher.
- **Automated Virtual Environment Management**: Automated detection, `.venv` isolation, dependency installation, and health checks.
- **Deterministic Pipeline**: Seed-locked reproducible data synthesis and model training pipeline (`scripts/pipeline.py`).

---

## 1. The core loop

1. Pick an origin, destination and routing profile.
2. The system pulls live weather + soil saturation + incident history for every
   road segment.
3. **Model 1** classifies each segment `safe / risky / blocked` with a probability.
4. **Model 2** estimates excess delay on each segment.
5. **Model 3** estimates end-to-end route delay (not a sum — see §4).
6. The router returns ranked alternates, scored on risk *and* time.
7. Alerts go out in English + the local language, with the reason attached.

---

## 2. What is real and what is simulated

Stated plainly, because a judge will ask and the answer is a strength:

| Layer | Status |
|---|---|
| Road network — 85 nodes, 98 segments, 43 NH corridors, 6,858 km | **Real.** Real towns, real coordinates, real elevations, real National Highway alignments. Verifiable on any map. |
| Terrain, gradient, flood exposure, landslide susceptibility | **Derived** from real elevations and documented hotspot corridors. |
| Weather | **Simulated** from real NER climatology — monthly monsoon profiles, per-district orographic multipliers (Meghalaya's southern slopes vs Tawang's rain shadow), wet-spell autocorrelation, gamma-tailed cloudbursts. |
| Failure labels | **Simulated** from a hidden physical hazard model the ML never sees. |
| The ML | **Real.** Trained on 44,296 segment observations + 4,427 route samples, evaluated on a held-out *temporal* split. |

The hazard model combines independent failure channels with a noisy-OR:
rainfall-triggered slope failure, floodplain inundation, snow/ice on high
passes, storm damage, structural memory (a slope that has failed will fail
again), blockades/bandhs, and hill fog. The ML has to recover that from noisy
observations. **That is a legitimate supervised setup, not a tautology** — the
model sees features and outcomes, never the generating function.

Swapping in real data is a data-source change, not a rewrite: replace the
`weather` table with IMD / Open-Meteo rows and `incidents` with state PWD +
Bhuvan landslide-inventory records. Everything downstream is unchanged.

---

## 3. Model performance (honest numbers)

Evaluated on a **temporal split** — trained on the first 80% of the timeline,
tested on the last 20%. In production you always predict forward in time, so
this is the number that means something. The random split is reported alongside
so the optimism gap is visible instead of hidden.

| Model | Metric | Temporal | Random |
|---|---|---|---|
| Risk classifier | Accuracy | **82.5%** | 89.6% |
| | Macro F1 | 0.829 | 0.882 |
| | Balanced accuracy | 0.843 | — |
| | **Recall on `blocked`** | **87.2%** | 88.1% |
| | AUC (will-disrupt) | 0.954 | — |
| | Brier score | 0.100 | — |
| | test rows | 8,918 | — |
| Segment delay | MAE | **0.98 h** | — |
| | R² | 0.833 | — |
| | within 1 h | 78.2% | — |
| Route delay | MAE | **5.05 h** | — |
| | R² | 0.952 | — |
| | mean-baseline MAE | 32.5 h | — |

The build is **deterministic** — fixed seeds throughout, so `scripts/pipeline.py`
reproduces these numbers exactly on any machine. The live figures are always in
the dashboard's **Model** tab, read straight from `metrics.json`.

**Recall on `blocked` is the metric that matters.** A missed closure sends a
convoy into a landslide; a false alarm costs a phone call. Classes are weighted
to protect it.

A HistGradientBoosting challenger scores ~3 points higher (85.7%). **We ship
the RandomForest anyway**, because the forest supports exact additive
attribution (§5) and the boosted model does not. For a tool a district officer
has to sign off on, explainability is worth more than a point of accuracy. That
trade is a decision, stated — not an accident.

### What the model learned by itself

Permutation importance on the held-out fold ranks **`api_7d` — seven-day
antecedent rainfall, i.e. soil saturation — as the strongest single driver**,
ahead of today's rainfall.

Nobody told it that. It recovered the actual physics of Himalayan slope failure
from the data: **slopes do not fail because of today's rain, they fail because
today's rain lands on ground already saturated by last week's.** That is exactly
why the feature exists, and the model independently agreeing is the best
evidence the pipeline is sound.

---

## 4. Architecture

```
frontend/                 zero-dependency dashboard
  index.html  app.js      hand-rolled SVG map engine, pan/zoom, 8 layers
  styles.css              dark control-room theme
  field.html  sw.js       offline-first PWA field reporter (IndexedDB queue)

backend/
  geography.py            REAL NER network: nodes, corridors, terrain derivation
  datagen.py              climatology + hidden hazard model + label generation
  features.py             single source of truth for the feature contract
  train.py                stage 1 + stage 2 training, honest evaluation
  obs_predict.py          scores history with stage-1 models (cascade input)
  explain.py              exact additive decision-path attribution
  inference.py            batch network scoring + what-if overrides + caching
  routing.py              risk-aware routing, criticality, accessibility, optimiser
  alerts.py               multilingual alert generation
  fleet.py                live fleet simulation
  app.py                  Starlette ASGI API + SSE stream
  db.py                   SQLite schema and access

scripts/pipeline.py       one command: data → stage 1 → cascade → stage 2
```

### The two-stage cascade (and the bug it fixes)

The route-delay model consumes "risk on this route" as an input. Train it on
*true* hazard but serve it *predicted* risk and the input distributions differ —
the model is silently wrong.

We hit exactly that: a Siliguri→Gangtok route whose segments summed to 3 h of
delay was predicted at **27 h**, because predicted failure probability saturates
near 1.0 where true hazard sat at 0.45.

The fix is a proper cascade — stage 2 trains on **stage 1's own outputs**, so the
distribution at training time matches production. `obs_predict.py` exists solely
for this. It is the single most important engineering decision in the stack.

### Why route delay is not a sum of segment delays

Because queues cascade, a blocked segment forces re-planning and detour, every
inter-state checkpost adds fixed dwell time, and hill sections effectively close
after dark. The route model learns these interactions; a sum cannot represent
them. The UI shows both numbers side by side so you can see the difference.

### Severity vs failure probability

`risk_score` = P(risky) + P(blocked) saturates near 1.0 for *any* hill segment in
the monsoon, so it cannot separate "slow going" from "impassable". We publish a
second graded score, `severity` = 0.4·P(risky) + 1.0·P(blocked), and route on
that. It is what drives the map colour ramp too.

---

## 5. Explainable AI — exact, not decorative

`shap` isn't installable in this environment, so rather than fall back to a
hand-wavy global importance chart, we implemented the real thing.

For a decision tree, the prediction equals the value at the root plus the sum of
value-changes along the sample's decision path; each change is attributed to the
feature that produced that split. Averaged over the forest this is **exactly
additive**:

```
baseline + Σ contributions  ==  model output     (to ~1e-12)
```

This is the Saabas decision-path method — same family as SHAP's TreeSHAP,
differing in how credit is split when features interact (TreeSHAP averages over
all orderings; this follows the single realised path). **The API returns the
additivity residual on every call**, so the explanation can be verified rather
than trusted:

```json
"additivity_check": { "abs_error": 0.0, "exact": true }
```

Each driver is rendered as an operational sentence, not a number:

> *"Ground is already saturated from 214 mm of antecedent rainfall — slopes fail
> on the next moderate burst, not the first one."*

That is what turns a score into an order someone can sign.

---

## 6. LIVE MODE — real weather and real GPS

The project runs fully offline by default. Two switches make it live.

### Real weather (Open-Meteo)

```bash
python3 -m backend.providers.weather_openmeteo --test   # verify + see drift check
./run.sh --live                                          # run with live weather
```

`--live` pulls real observations and forecasts for all 85 nodes at startup and
every 30 minutes after (`SENTINEL_WEATHER_INTERVAL`). You can also trigger a
pull at any time with `POST /api/weather/refresh`, and check provenance with
`GET /api/weather/status`.

**No API key.** Open-Meteo is free and keyless — nothing to configure and
nothing to expire the night before a demo. IMD is the right authoritative
source for production; it has no clean public JSON API, and `fetch_raw()` is
the single function that would change.

**The hard part is feature parity, not the HTTP call.** The models were trained
on features computed a specific way. The adapter recomputes `rainfall_24h_mm`,
`rainfall_72h_mm` and the 7-day antecedent index using *the same decay constant
and the same recurrence* as `datagen.py` — there is a unit test asserting the
two implementations agree. Get this wrong and the model degrades silently.

Two things the adapter handles that a naive integration gets wrong:

1. **Accumulation can escalate the reported condition.** Open-Meteo gives the
   weather code at one instant, so a segment that has taken 120 mm over the day
   can be labelled "light drizzle" because that is what it happens to be doing
   right now. We take the more severe of the code and the 24-hour accumulation.
2. **The antecedent index is rolled forward across forecast days**, so the
   departure-window optimiser reasons about soil saturation *as it will be*,
   not as it is now.

**Drift check.** `--test` prints every live feature next to the 5th–95th
percentile range the models were trained on and warns if anything sits outside
it. Tree models do not extrapolate — they saturate at the edge of what they
saw — so knowing you are off-distribution matters more than a pretty number.

**Failure is non-fatal by design.** If the network is down, the refresh logs,
returns `ok: false`, and the previously stored weather keeps serving. The
dashboard never goes down because a weather API did.

### Real GPS tracking

```bash
./run.sh --https          # then open https://<your-ip>:8443/track on a phone
```

`/track` streams real phone GPS to `POST /api/track`. Every fix is snapped to
the nearest highway segment (accurate to a few metres against the real
alignment) and scored by the live model, so a vehicle is not tracked on a
separate map — **it is tracked against the risk surface the control room is
watching.** The driver sees the risk of the road they are on, and:

- a **geofence-breach** alert the moment they enter a risky or blocked segment,
- a **look-ahead** warning when a blocked segment is coming up on the same
  corridor ("blocked in 118 km — re-route before the next junction"), fired
  once per hazard rather than on every fix,
- their device appears on the control-room map as a hexagon with a breadcrumb
  trail, visually distinct from the simulated fleet.

`POST /api/track` takes plain JSON with `lat`/`lon`, so a real **AIS-140 / VTS**
unit or a VAHAN feed pushes into exactly the same path with no code change. A
phone is simply the cheapest tracker for a demo.

**Why `--https` is not optional for phone GPS:** browsers only expose
`navigator.geolocation` in a secure context. Over plain `http://192.168.x.x` a
phone returns nothing, silently. `--https` generates a self-signed certificate
and serves on :8443; the phone warns once and you accept it. The page detects an
insecure context and says exactly this rather than appearing broken.

### What is live and what is not

| | Default | `--live` | `--https` + `/track` |
|---|---|---|---|
| Road network | real | real | real |
| Weather | simulated | **real (Open-Meteo)** | — |
| Incident history | simulated | simulated | — |
| Field reports | real (yours) | real | — |
| Vehicle positions | simulated fleet | simulated fleet | **real GPS** |

Incident history stays simulated in every mode — there is no public feed of NER
road closures to replace it with. That is the honest gap, and it is the first
thing we would wire to a state PWD feed.

---

## 7. Features

**Core**
- Interactive network map, 98 segments colour-coded by predicted severity
- Risk classification (safe / risky / blocked) with calibrated probabilities
- Segment and route delay prediction
- Ranked alternate routes with diversity (not three variants of one road)
- Four routing profiles: fastest / balanced / safest / **emergency**
- Multilingual alert feed
- Geo-tagged incident reporting, snapped to the nearest segment

**The differentiators**

1. **Choke-point analysis.** Every segment is deleted from the graph and the
   network re-solved from the supply hub. Output: *"if this 38 km fails, 11
   districts and 1.2 M people lose their only road link."* The system
   independently identifies the Siliguri Corridor, NH-306 into Mizoram, NH-10
   into Sikkim and Sela Pass — the real chokepoints of NER logistics. Ranked
   both structurally and by **live exposure** (criticality × today's failure
   probability) — that second list is the watch-list a control room opens the
   morning with.

2. **District Accessibility Index (0–100).** The problem statement asks for
   accessibility intelligence, not just routing. A composite of reach,
   reliability, redundancy, terrain burden and live disruption — **every
   sub-score published**, so an official can see why a district scores 32 and
   what would move it. Correctly ranks Mizoram worst (34.1), then Tripura (42.9), then
   Sikkim (44.7) — the three states each served by a single arterial road.

3. **Departure-window optimiser.** Sweeps the 72-hour forecast, re-scoring the
   entire network at every 3-hour window, and answers the real dispatch
   question: *leave now, or hold nine hours and save six?* Holding a convoy
   costs money; driving into a failing slope costs the load, the vehicle and
   sometimes the crew.

4. **Counterfactual what-if simulator.** Drag rainfall and soil saturation,
   scope it to states, and watch the network degrade live — models re-score,
   routes re-plan, the accessibility index recomputes. Nothing is hard-coded.
   The Meghalaya cloudburst preset takes blocked segments from 3 → 35 and cut
   road length from 236 km → 2,636 km.

5. **Emergency mode with an escalation ladder.** Critical-supply routing starts
   strict (avoid anything above 0.75 failure probability) and relaxes only as
   far as it must, **reporting exactly which constraint it had to give up**. An
   operator is never shown a "safe" route that quietly crossed a failing slope.

6. **Critical-cargo viability.** Blood has an 8-hour window, vaccines 36. The
   system checks predicted ETA against the cargo's window and says plainly when
   road transport is not viable and an air lift is needed.

7. **Live fleet tracking** over Server-Sent Events — trucks on real route
   geometry, speed modulated by segment risk, with geofence-breach alerts,
   look-ahead warnings (*"blocked segment 40 km ahead, re-route before the next
   junction"*) and cargo-window alarms.

8. **Offline-first field PWA.** Large parts of NER have no usable mobile data
   exactly where roads fail. Service worker + IndexedDB queue; reports carry a
   client ID so a retry can never create a duplicate. Reports snap to the
   nearest segment and feed the incident-memory features on the next inference —
   the loop closes.

9. **"No alternative exists" detection.** When all four profiles return the same
   road, that is reported as a *finding*, not a bug: safety cannot be traded for
   time because there is nothing to trade with. It is the single most important
   fact about NER logistics.

---

## 8. API

`GET /api` returns the full index. Highlights:

| Endpoint | Purpose |
|---|---|
| `/api/network` | all segments scored at a timestamp |
| `/api/segment/{road_id}` | one segment + **explained** prediction |
| `/api/route` | ranked routes (`origin`, `dest`, `profile`, `k`, `cargo`) |
| `/api/route/compare` | the same OD under all four profiles |
| `/api/departure` | 72-hour departure-window sweep |
| `/api/criticality` | single-point-of-failure ranking |
| `/api/accessibility` | district accessibility index |
| `/api/whatif` | counterfactual vs baseline |
| `/api/alerts` | multilingual control-room feed |
| `POST /api/incidents` | field report ingestion (JSON or multipart) |
| `/api/fleet/stream` | live telemetry (Server-Sent Events) |

Any read endpoint accepts what-if parameters: `rain_24h_set`, `rain_multiplier`,
`api_set`, `api_multiplier`, `temperature_delta`, `condition_set`, `states`.

**Why SSE rather than WebSockets** for fleet telemetry: it is strictly one-way
(server → control room), runs over plain HTTP/1.1 with no extra dependency,
reconnects automatically in the browser, and survives proxies that drop upgrade
requests. A `/ws/fleet` WebSocket route is also present and activates when a
WebSocket library is installed.

---

## 9. Known limitations

Worth stating before a judge finds them:

- **Weather is simulated unless you pass `--live`.** The Open-Meteo adapter is
  written and unit-tested against a synthetic payload, but it could not be run
  against the live API from the machine it was built on (egress blocked), so
  run `--test` once on your own machine before relying on it.
- **Incident history is simulated in every mode.** There is no public feed of
  NER road closures; a state PWD / NHAI integration is the first real-data gap
  to close.
- **Translations need review.** English, Hindi, Assamese, Bengali and Nepali are
  reliable. Mizo, Khasi and Meitei are structurally correct placeholders written
  without a native speaker — **the system flags them `review_pending` in the API
  and the UI rather than pretending otherwise.** Shipping unreviewed text in a
  safety-critical alert is a real harm.
- **State extents on the map are schematic** (convex hulls of network nodes),
  not survey boundaries.
- **Road geometry is straight-line between nodes**, with length corrected by a
  terrain-dependent sinuosity factor. Real OSM polylines would be a drop-in
  improvement.
- **No live traffic**, no vehicle-specific constraints (axle load, height), no
  toll or fuel costing yet.
- The fleet is **simulated**, not connected to real GPS/VTS feeds.

---

## 10. Where this goes next

- Ingest IMD nowcast + Open-Meteo forecast; Bhuvan landslide inventory; NHAI /
  state PWD closure feeds; VAHAN VTS for real fleet positions.
- Replace synthetic labels with actual closure records and retrain — the
  pipeline is unchanged.
- Push alerts over SMS/IVR gateways in the local language (the alert layer
  already emits per-language payloads).
- Publish the accessibility index as a planning instrument: it ranks where a
  bypass, tunnel or second bridge buys the most resilience per rupee — the Sela
  Tunnel is precisely this argument, made retrospectively.
