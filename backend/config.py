"""
Runtime configuration.

Everything is an environment variable with a safe default, because the one
thing you cannot afford on demo day is a config file you forgot to edit.

    SENTINEL_LIVE_WEATHER=1     pull real weather from Open-Meteo at startup
                                and every SENTINEL_WEATHER_INTERVAL minutes
    SENTINEL_WEATHER_INTERVAL=30
    SENTINEL_LIVE_GPS=1         show real GPS-tracked devices (default on)
    SENTINEL_SIM_FLEET=1        also run the simulated demo fleet (default on)
    PORT=8000
"""
from __future__ import annotations

import os


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


LIVE_WEATHER = _flag("SENTINEL_LIVE_WEATHER", False)
WEATHER_INTERVAL_MIN = max(5, _int("SENTINEL_WEATHER_INTERVAL", 30))
LIVE_GPS = _flag("SENTINEL_LIVE_GPS", True)
SIM_FLEET = _flag("SENTINEL_SIM_FLEET", True)


def describe() -> dict:
    return {
        "live_weather": LIVE_WEATHER,
        "weather_interval_minutes": WEATHER_INTERVAL_MIN,
        "live_gps": LIVE_GPS,
        "simulated_fleet": SIM_FLEET,
        "note": ("Live weather is OFF by default so the project always runs "
                 "offline. Set SENTINEL_LIVE_WEATHER=1 to pull real data from "
                 "Open-Meteo. GPS tracking is always available — it simply has "
                 "no vehicles until a device starts reporting to /api/track."),
    }
