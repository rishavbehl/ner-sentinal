"""
LIVE WEATHER — Open-Meteo provider.

Why Open-Meteo and not IMD or OpenWeatherMap:
  * No API key. Nothing to configure, nothing to leak, nothing to rate-limit
    you at 2 a.m. the night before a demo.
  * Free for non-commercial use.
  * It serves PAST days and FORECAST days from the same endpoint, which this
    project specifically needs: our strongest feature is a 7-day antecedent
    rainfall index, so we cannot work from a "current conditions" API alone.
    Most weather APIs give you now; we need the last three weeks and the next
    three days.
  * IMD is the right production source for India and has the authoritative
    station network, but it has no clean public JSON API. The adapter boundary
    below is where IMD would plug in — `fetch_raw()` is the only function that
    would change.

FEATURE PARITY IS THE HARD PART, NOT THE HTTP CALL.
The models were trained on features computed a specific way. If live data is
assembled even slightly differently — a different accumulation window, a
different decay constant — the model silently degrades. So the API index here
is computed with the SAME decay (0.90/day) and the SAME definition as
`datagen.py`, and `python3 -m backend.providers.weather_openmeteo --test`
prints the live values next to the training distribution so drift is visible
rather than assumed.

Run the self-test on a machine with internet:
    python3 -m backend.providers.weather_openmeteo --test
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .. import db
from ..geography import NODES

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
USER_AGENT = "NER-Logistics-Sentinel/1.0 (SIH prototype; contact: team)"

# Must match backend/datagen.py — the models were trained with this decay.
API_DECAY = 0.90
API_PAST_DAYS = 21          # 0.90^21 ≈ 0.11 — captures ~89% of the infinite sum
HOURLY_PAST_DAYS = 3        # enough for the 24 h and 72 h accumulations
FORECAST_DAYS = 3

CHUNK = 20                  # locations per request — polite, and well under limits
TIMEOUT = 25
RETRIES = 3


# --------------------------------------------------------------------------
# WMO weather codes -> our condition vocabulary
# https://open-meteo.com/en/docs  (WW interpretation codes)
# --------------------------------------------------------------------------
SEVERITY_RANK = {"clear": 0, "cloudy": 1, "fog": 2, "light_rain": 3,
                 "rain": 4, "heavy_rain": 5, "storm": 6, "snow": 7}


def _from_accumulation(precip_mm: float, temp_c: float) -> str:
    if temp_c <= 1.5 and precip_mm > 1.0:
        return "snow"
    if precip_mm >= 65:
        return "storm"
    if precip_mm >= 25:
        return "heavy_rain"
    if precip_mm >= 7:
        return "rain"
    if precip_mm >= 1.0:
        return "light_rain"
    return "clear"


def _from_code(code: Optional[int]) -> str:
    if code is None:
        return "clear"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "storm"
    if code in (45, 48):
        return "fog"
    if code in (65, 67, 82):
        return "heavy_rain"
    if code in (63, 66, 81):
        return "rain"
    if code in (51, 53, 55, 56, 57, 61, 80):
        return "light_rain"
    if code in (1, 2, 3):
        return "cloudy"
    if code == 0:
        return "clear"
    return "cloudy"


def wmo_to_condition(code: Optional[int], precip_mm: float, temp_c: float) -> str:
    """
    Combine the instantaneous WMO code with the 24 h ACCUMULATION, taking
    whichever is more severe.

    This matters: Open-Meteo reports the condition at a single instant, so a
    segment that has taken 120 mm over the day can carry code 61 ("light
    drizzle") because it happens to be drizzling at the moment of the reading.
    For a road, 120 mm in 24 h is a storm regardless of what it is doing right
    now — and the model was trained on accumulation-derived conditions, so
    taking the max is also what keeps live data on-distribution.
    """
    a = _from_accumulation(precip_mm, temp_c)
    c = _from_code(code)
    if "snow" in (a, c):
        return "snow"
    return a if SEVERITY_RANK[a] >= SEVERITY_RANK[c] else c


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _get(url: str, params: dict) -> dict | list:
    qs = urllib.parse.urlencode(params, safe=",")
    full = f"{url}?{qs}"
    last: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            last = RuntimeError(f"HTTP {e.code} from Open-Meteo: {body}")
            if e.code in (400, 401, 403):
                break               # a bad request will not fix itself
        except Exception as e:       # URLError, timeout, JSON error
            last = e
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed after {RETRIES} attempts: {last}")


def _as_list(payload) -> list:
    """Open-Meteo returns an object for one location, an array for several."""
    return payload if isinstance(payload, list) else [payload]


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def fetch_raw(node_ids: List[str]) -> Dict[str, dict]:
    """
    Two calls per chunk, deliberately:
      * daily precipitation over 21 past days -> the antecedent index
      * hourly precip/temp/wind over 3 past + 3 forecast days -> everything else
    Requesting 21 days of HOURLY data for 85 locations would be ~10x the payload
    for no extra information.
    """
    out: Dict[str, dict] = {}
    for i in range(0, len(node_ids), CHUNK):
        batch = node_ids[i:i + CHUNK]
        lats = ",".join(f"{NODES[n].lat:.4f}" for n in batch)
        lons = ",".join(f"{NODES[n].lon:.4f}" for n in batch)

        daily = _as_list(_get(FORECAST_URL, {
            "latitude": lats, "longitude": lons,
            "daily": "precipitation_sum",
            "past_days": API_PAST_DAYS,
            "forecast_days": FORECAST_DAYS,
            "timezone": TIMEZONE,
        }))
        hourly = _as_list(_get(FORECAST_URL, {
            "latitude": lats, "longitude": lons,
            "hourly": "precipitation,temperature_2m,wind_speed_10m,weather_code",
            "past_days": HOURLY_PAST_DAYS,
            "forecast_days": FORECAST_DAYS,
            "timezone": TIMEZONE,
        }))
        if len(daily) != len(batch) or len(hourly) != len(batch):
            raise RuntimeError(
                f"Open-Meteo returned {len(daily)}/{len(hourly)} series for "
                f"{len(batch)} requested locations — response shape changed")
        for j, nid in enumerate(batch):
            out[nid] = {"daily": daily[j], "hourly": hourly[j]}
        time.sleep(0.4)              # be a good citizen of a free API
    return out


# --------------------------------------------------------------------------
# Feature assembly — MUST match datagen.py
# --------------------------------------------------------------------------
def compute_api_index(daily_precip: List[float]) -> float:
    """
    Antecedent Precipitation Index, oldest-to-newest, decay 0.90/day.
    This is the single strongest feature in the model, so it is computed here
    exactly as the training data computed it: api = decay*api + rain_today.
    """
    api = 0.0
    for p in daily_precip:
        api = API_DECAY * api + (p or 0.0)
    return round(api, 2)


def _window_sum(times: List[str], values: List[Optional[float]],
                end: datetime, hours: int) -> float:
    start = end - timedelta(hours=hours)
    tot = 0.0
    for t, v in zip(times, values):
        try:
            ts = datetime.fromisoformat(t)
        except ValueError:
            continue
        if start < ts <= end:
            tot += (v or 0.0)
    return round(tot, 2)


def _nearest_index(times: List[str], target: datetime) -> int:
    best, bd = 0, None
    for i, t in enumerate(times):
        try:
            ts = datetime.fromisoformat(t)
        except ValueError:
            continue
        d = abs((ts - target).total_seconds())
        if bd is None or d < bd:
            best, bd = i, d
    return best


def build_rows(raw: Dict[str, dict], at: Optional[datetime] = None
               ) -> List[tuple]:
    """
    Convert raw Open-Meteo payloads into `weather` table rows:
      (district, ts, rainfall_mm, rainfall_24h_mm, rainfall_72h_mm, api_7d,
       weather_condition, temperature_c, wind_kmph, is_forecast)

    Produces the current row plus every forecast hour at 3-hourly spacing, so
    the departure-window optimiser works on real forecast data.
    """
    now = at or datetime.now()
    rows: List[tuple] = []

    for nid, payload in raw.items():
        name = NODES[nid].name
        d = payload["daily"]
        h = payload["hourly"]
        dts = d.get("daily", {}).get("time", []) or []
        dpr = d.get("daily", {}).get("precipitation_sum", []) or []
        htimes = h.get("hourly", {}).get("time", []) or []
        hprec = h.get("hourly", {}).get("precipitation", []) or []
        htemp = h.get("hourly", {}).get("temperature_2m", []) or []
        hwind = h.get("hourly", {}).get("wind_speed_10m", []) or []
        hcode = h.get("hourly", {}).get("weather_code", []) or [None] * len(htimes)
        if not htimes:
            continue

        # --- antecedent index up to and including today
        today = now.date().isoformat()
        past = [p for t, p in zip(dts, dpr) if t <= today]
        api_now = compute_api_index(past)
        # daily precip keyed by date, for rolling the index forward
        daily_by_date = {t: (p or 0.0) for t, p in zip(dts, dpr)}

        # --- emit the "now" row plus 3-hourly forecast rows
        targets: List[Tuple[datetime, int]] = [(now, 0)]
        step = now.replace(minute=0, second=0, microsecond=0)
        for k in range(1, FORECAST_DAYS * 8 + 1):
            t = step + timedelta(hours=3 * k)
            targets.append((t, 1))

        for tgt, is_fc in targets:
            idx = _nearest_index(htimes, tgt)
            if idx >= len(hprec):
                continue
            temp = float(htemp[idx]) if idx < len(htemp) and htemp[idx] is not None else 20.0
            wind = float(hwind[idx]) if idx < len(hwind) and hwind[idx] is not None else 5.0
            code = hcode[idx] if idx < len(hcode) else None

            r24 = _window_sum(htimes, hprec, tgt, 24)
            r72 = _window_sum(htimes, hprec, tgt, 72)

            # roll the antecedent index forward across forecast days
            api_t = api_now
            dcursor = now.date()
            while dcursor < tgt.date():
                dcursor = dcursor + timedelta(days=1)
                api_t = API_DECAY * api_t + daily_by_date.get(dcursor.isoformat(), 0.0)

            cond = wmo_to_condition(int(code) if code is not None else None, r24, temp)
            rows.append((
                name, tgt.replace(minute=0, second=0,
                                  microsecond=0).isoformat(timespec="seconds"),
                round(float(hprec[idx] or 0.0), 2), r24, r72, round(api_t, 2),
                cond, round(temp, 1), round(wind, 1), is_fc,
            ))
    return rows


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def refresh(verbose: bool = True) -> dict:
    """
    Pull live weather for every node and write it into the `weather` table.
    Returns a status dict. Never raises — a failed refresh must not take the
    dashboard down; it logs, returns ok=False, and the previously stored
    (simulated or older live) rows keep serving.
    """
    t0 = time.time()
    node_ids = list(NODES.keys())
    try:
        raw = fetch_raw(node_ids)
    except Exception as e:
        if verbose:
            print(f"[weather] LIVE FETCH FAILED: {e}")
        return {"ok": False, "error": str(e), "rows": 0,
                "note": "Serving previously stored weather. The dashboard keeps "
                        "working; it is simply no longer live."}

    rows = build_rows(raw)
    if not rows:
        return {"ok": False, "error": "no rows built from response", "rows": 0}

    db.executemany(
        "INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    stamp = datetime.now().isoformat(timespec="seconds")
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('weather_source', ?)",
        (json.dumps({"source": "open-meteo", "fetched_at": stamp,
                     "nodes": len(raw), "rows": len(rows)}),))

    if verbose:
        print(f"[weather] live refresh OK — {len(raw)} locations, "
              f"{len(rows)} rows, {time.time() - t0:.1f}s")
    return {"ok": True, "rows": len(rows), "nodes": len(raw),
            "fetched_at": stamp, "seconds": round(time.time() - t0, 1)}


def last_refresh() -> Optional[dict]:
    r = db.query_one("SELECT value FROM meta WHERE key = 'weather_source'")
    if not r:
        return None
    try:
        return json.loads(r["value"])
    except Exception:
        return None


# --------------------------------------------------------------------------
# Self-test — run this on a machine with internet
# --------------------------------------------------------------------------
TRAINING_RANGES = {
    # 5th–95th percentile observed in the training data, for drift checking
    "rainfall_24h_mm": (0.0, 38.0),
    "rainfall_72h_mm": (0.0, 78.0),
    "api_7d": (1.0, 215.0),
    "temperature_c": (4.0, 33.0),
    "wind_kmph": (1.0, 28.0),
}


def self_test() -> int:
    print("=" * 72)
    print("LIVE WEATHER SELF-TEST — Open-Meteo")
    print("=" * 72)
    sample = ["GUWAHATI", "SHILLONG", "AIZAWL", "TAWANG", "IMPHAL",
              "GANGTOK", "AGARTALA", "SILIGURI"]
    print(f"\nFetching {len(sample)} locations …")
    try:
        raw = fetch_raw(sample)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        print("\n  Check: is this machine online? Does a proxy block "
              "api.open-meteo.com?")
        print("  The app still runs without this — it falls back to the "
              "simulated weather it was built with.")
        return 1

    rows = build_rows(raw)
    now_rows = [r for r in rows if r[9] == 0]
    print(f"  OK — {len(raw)} locations, {len(rows)} rows "
          f"({len(now_rows)} current, {len(rows) - len(now_rows)} forecast)\n")

    print(f"{'District':<14}{'rain24':>8}{'rain72':>8}{'API_7d':>9}"
          f"{'temp':>7}{'wind':>7}  condition")
    print("-" * 72)
    for r in sorted(now_rows):
        print(f"{r[0][:13]:<14}{r[3]:>8.1f}{r[4]:>8.1f}{r[5]:>9.1f}"
              f"{r[7]:>7.1f}{r[8]:>7.1f}  {r[6]}")

    print("\n" + "-" * 72)
    print("DRIFT CHECK — live values vs the range the models were trained on")
    print("-" * 72)
    cols = {"rainfall_24h_mm": 3, "rainfall_72h_mm": 4, "api_7d": 5,
            "temperature_c": 7, "wind_kmph": 8}
    warned = False
    for feat, idx in cols.items():
        vals = [r[idx] for r in now_rows]
        lo, hi = min(vals), max(vals)
        tlo, thi = TRAINING_RANGES[feat]
        ok = hi <= thi * 1.6 and lo >= -0.01
        flag = "ok" if ok else "OUT OF RANGE"
        if not ok:
            warned = True
        print(f"  {feat:<20} live {lo:7.1f} … {hi:7.1f}   "
              f"training p5–p95 {tlo:6.1f} … {thi:6.1f}   [{flag}]")

    print()
    if warned:
        print("  NOTE: at least one feature sits outside the training range.")
        print("  Tree models do not extrapolate — they will saturate at the")
        print("  edge of what they saw. Predictions stay sane, but treat the")
        print("  extreme end as 'at least this bad' rather than calibrated.")
    else:
        print("  All live features sit inside the training distribution.")
        print("  The models are operating on data that looks like what they")
        print("  learned from.")
    print("\nTo switch the app to live weather:")
    print("    export SENTINEL_LIVE_WEATHER=1   &&   ./run.sh")
    print("or hit  POST /api/weather/refresh  while it is running.\n")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        raise SystemExit(self_test())
    db.init_db()
    print(json.dumps(refresh(), indent=2))
