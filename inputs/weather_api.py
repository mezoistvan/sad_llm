"""Open-Meteo client for current conditions.

No auth, no rate limits at this scale. Single GET per call. 10-minute in-process
cache to avoid hammering the API when run.py is invoked repeatedly.

If the network fails and we have a cached value (even a stale one), return it
with `is_stale=True`. This keeps run.py functional during transient outages.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 600  # 10 minutes
REQUEST_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class CurrentWeather:
    latitude: float
    longitude: float
    temperature_c: float
    cloud_cover_pct: float       # 0..100
    precipitation_mm: float      # mm in the last hour
    wind_kph: float              # km/h
    is_daytime: bool
    timezone: str
    fetched_at: float            # unix seconds at the time of the API call
    is_stale: bool = False       # True if returned from cache after a failed refresh

    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


# Module-level cache keyed by (rounded_lat, rounded_lon).
_CACHE: dict[tuple[float, float], CurrentWeather] = {}


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    # Round to 2 decimal places (~1km precision) so calls with sub-km jitter
    # share a cache entry. More than enough for SAD-LLM.
    return (round(lat, 2), round(lon, 2))


def fetch_current_weather(
    lat: float,
    lon: float,
    *,
    use_cache: bool = True,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> CurrentWeather:
    """Fetch current weather for (lat, lon) from Open-Meteo.

    Returns the cached value if it's < CACHE_TTL_SECONDS old. On network failure
    with no cached value, raises RuntimeError. On network failure with a stale
    cached value, returns the stale value with is_stale=True.
    """
    key = _cache_key(lat, lon)
    cached = _CACHE.get(key) if use_cache else None
    if cached is not None and cached.age_seconds() < CACHE_TTL_SECONDS:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m",
            "cloud_cover",
            "precipitation",
            "wind_speed_10m",
            "is_day",
        ]),
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        if cached is not None:
            return CurrentWeather(
                **{**cached.__dict__, "is_stale": True},
            )
        raise RuntimeError(f"Open-Meteo request failed and no cache available: {e}") from e

    current = data["current"]
    weather = CurrentWeather(
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
        temperature_c=float(current["temperature_2m"]),
        cloud_cover_pct=float(current["cloud_cover"]),
        precipitation_mm=float(current["precipitation"]),
        wind_kph=float(current["wind_speed_10m"]),
        is_daytime=bool(current["is_day"]),
        timezone=str(data.get("timezone", "UTC")),
        fetched_at=time.time(),
        is_stale=False,
    )
    _CACHE[key] = weather
    return weather


def fake_weather(season: str, lat: float = 47.5, lon: float = 19.0) -> CurrentWeather:
    """Synthetic weather for `--force-season` debugging in run.py.
    Produces niceness near +1 (summer) or near -1 (winter) under default weights."""
    if season == "summer":
        return CurrentWeather(
            latitude=lat, longitude=lon,
            temperature_c=22.0, cloud_cover_pct=10.0, precipitation_mm=0.0,
            wind_kph=5.0, is_daytime=True, timezone="forced",
            fetched_at=time.time(),
        )
    if season == "winter":
        return CurrentWeather(
            latitude=lat, longitude=lon,
            temperature_c=2.0, cloud_cover_pct=95.0, precipitation_mm=1.5,
            wind_kph=28.0, is_daytime=False, timezone="forced",
            fetched_at=time.time(),
        )
    raise ValueError(f"Unknown season: {season!r}; expected 'summer' or 'winter'.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=47.4979, help="Latitude")
    parser.add_argument("--lon", type=float, default=19.0402, help="Longitude")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    weather = fetch_current_weather(args.lat, args.lon, use_cache=not args.no_cache)
    print(f"Location:        ({weather.latitude}, {weather.longitude}) {weather.timezone}")
    print(f"Temperature:     {weather.temperature_c:.1f} C")
    print(f"Cloud cover:     {weather.cloud_cover_pct:.0f}%")
    print(f"Precipitation:   {weather.precipitation_mm:.1f} mm/h")
    print(f"Wind:            {weather.wind_kph:.1f} kph")
    print(f"Daytime:         {weather.is_daytime}")
    print(f"Fetched at:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(weather.fetched_at))}")
    if weather.is_stale:
        print("(cached, stale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
