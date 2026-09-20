# Real-World Dataset Integration Guide: Google Earth Engine & Satellite Feeds

This document details how the four recommended datasets are accessed, mapped, and integrated into the **NER Logistics Sentinel** ML pipeline (Random Forest + XGBoost ensemble).

---

## 1. Dataset Matrix & Direct Access

| Dataset | Real-world Source | Google Earth Engine / API Endpoint | Pipeline Mapping (`backend/features.py`) |
| :--- | :--- | :--- | :--- |
| **ERA5-Land Hourly** | ECMWF / Copernicus | `ECMWF/ERA5_LAND/HOURLY`<br>or REST via Open-Meteo ERA5-Land Reanalysis API | `api_7d`, `temperature_c`, soil saturation layer |
| **NASA GPM IMERG** | NASA Goddard / JAXA | `NASA/GPM_L3/IMERG_V06`<br>or Live Open-Meteo GPM Ensemble | `rainfall_24h_mm`, `rainfall_72h_mm`, `weather_condition` |
| **NASA Global Landslide Catalog (GLC) & ISRO Bhuvan** | NASA / ISRO | NASA Open Data / Bhuvan ISRO GeoPortal | `landslide_base`, `hist_incidents_90d`, `days_since_last_incident` |
| **SRTM Digital Elevation Model (30m)** | NASA / USGS | `USGS/SRTMGL1_003`<br>or Open-Meteo SRTM Elevation API | `max_elev_m`, `avg_slope_pct`, `terrain_enc` |

---

## 2. Ingestion Methods

### Method A: REST Ingestion (Immediate, No Google Cloud Auth / Billing Required)
The project now includes live satellite & reanalysis ingestion through high-throughput open endpoints backed by ECMWF and NASA models:
- **Real-Time Weather & GPM Precipitation**: Built into `backend/providers/weather_openmeteo.py` with `SENTINEL_LIVE_WEATHER=1`.
- **ERA5-Land Historical & Forecast Reanalysis**: `https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&models=era5_land&hourly=precipitation,soil_moisture_0_to_7cm`
- **SRTM 30m Topography**: `https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}`

### Method B: Google Earth Engine (GEE) Python API Ingestion
For batch geospatial extraction directly using your Google Cloud Account:
1. Install GEE: `pip install earthengine-api`
2. Authenticate: `earthengine authenticate`
3. Sample GEE Extraction script:
```python
import ee
ee.Initialize(project="your-gcp-project-id")

# 1. SRTM 30m Elevation & Slope
srtm = ee.Image("USGS/SRTMGL1_003")
slope = ee.Terrain.slope(srtm)

# 2. ERA5-Land Soil Moisture and 2m Temperature
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
         .filterDate("2024-06-01", "2024-06-30") \
         .select(["volumetric_soil_water_layer_1", "temperature_2m"])

# 3. NASA GPM IMERG Half-Hourly Precipitation
gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V06") \
        .filterDate("2024-06-01", "2024-06-30") \
        .select("precipitationCal")
```

---

## 3. Retraining with Live/Updated Data
Whenever fresh observations are synced into SQLite:
```powershell
.venv\Scripts\python -m scripts.pipeline
```
This automatically updates:
1. Stage 1 `risk_clf.joblib` (RandomForest + XGBoost soft-voting ensemble)
2. Stage 1 `delay_reg.joblib` (RandomForest + XGBoost voting regressor)
3. Stage 2 `route_delay_reg.joblib` (RandomForest + XGBoost cascade delay regressor)
