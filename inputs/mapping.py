"""Weather -> steering coefficients.

Pure, side-effect-free, easy to unit-test. The whole module is deliberately
tiny so the mapping logic stays inspectable. Tunables live at the top.

Pipeline:
    weather  ->  niceness in [-1, +1]  ->  {happy: {layer: coef, ...}, sad: {...}}
                                           coefficients in fraction-of-norm units

Each emotion can be steered at one or more layers simultaneously (per-emotion
list configured in config.yaml). Every layer belonging to an emotion shares
the same niceness scalar but has its own calibrated `max`, so a single
niceness value drives all layers of that emotion proportionally to their
per-layer maxes.

`happy` and `sad` coefficients are non-negative and at most one emotion group
is non-zero at a time — niceness > 0 activates happy layers only, niceness < 0
activates sad layers only. Avoids the muddle of both emotions being half-active
simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inputs.weather_api import CurrentWeather


# Tunables. Each component contributes to niceness in [-1, +1].
# Final niceness = clamp(weighted_sum, -1, +1).

@dataclass(frozen=True)
class NicenessWeights:
    """Default weights sum to 1.0 so the weighted sum naturally lives in [-1, +1].
    Temperature is weighted highest because human emotional response to weather
    is dominated by temperature comfort more than any other single factor."""
    cloud: float = 0.25
    precipitation: float = 0.20
    daylight: float = 0.10
    temperature: float = 0.35
    wind: float = 0.10

    def total(self) -> float:
        return self.cloud + self.precipitation + self.daylight + self.temperature + self.wind


# Temperature niceness peaks at TEMP_PEAK_C, linearly decays to 0 at the bounds,
# becomes negative beyond them. (e.g. 35C is unpleasant, -5C is unpleasant.)
TEMP_PEAK_C = 22.0
TEMP_PLEASANT_RADIUS_C = 8.0     # +1 at peak; 0 at ±this from peak
TEMP_UNPLEASANT_RADIUS_C = 18.0  # -1 at ±this from peak (and beyond, clamped)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cloud_niceness(cloud_pct: float) -> float:
    """0% cloud -> +1, 100% cloud -> -1, linear."""
    return 1.0 - (cloud_pct / 50.0)


def _precip_niceness(precip_mm: float) -> float:
    """Dry weather is mildly nice; rain is strongly unpleasant.
    0 mm/h -> +0.3, 1 mm/h -> -0.3, >=2 mm/h -> -1.0 (linear in between)."""
    if precip_mm <= 0.0:
        return 0.3
    if precip_mm >= 2.0:
        return -1.0
    # Linear from (0, +0.3) to (2, -1.0): slope = -0.65 per mm
    return 0.3 - 0.65 * precip_mm


def _daylight_niceness(is_daytime: bool) -> float:
    """Daytime is mildly +; nighttime is mildly -. Time-of-day matters but
    isn't the dominant signal — bad daytime weather should still be sad."""
    return 0.5 if is_daytime else -0.5


def _temperature_niceness(temp_c: float) -> float:
    """Peaks at TEMP_PEAK_C, falls off symmetrically. Piecewise linear."""
    delta = abs(temp_c - TEMP_PEAK_C)
    if delta <= TEMP_PLEASANT_RADIUS_C:
        return 1.0 - (delta / TEMP_PLEASANT_RADIUS_C)
    if delta <= TEMP_UNPLEASANT_RADIUS_C:
        out = (delta - TEMP_PLEASANT_RADIUS_C) / (TEMP_UNPLEASANT_RADIUS_C - TEMP_PLEASANT_RADIUS_C)
        return -out
    return -1.0


def _wind_niceness(wind_kph: float) -> float:
    """Calm to light breeze is mildly nice; gale-force is fully unpleasant.
    <=5 kph -> +0.3, 15 kph -> 0, >=40 kph -> -1.0 (piecewise linear)."""
    if wind_kph <= 5.0:
        return 0.3
    if wind_kph >= 40.0:
        return -1.0
    if wind_kph <= 15.0:
        # 5..15 kph: linear from +0.3 to 0
        return 0.3 - 0.03 * (wind_kph - 5.0)
    # 15..40 kph: linear from 0 to -1
    return -(wind_kph - 15.0) / 25.0


def compute_niceness(
    weather: "CurrentWeather",
    *,
    weights: NicenessWeights = NicenessWeights(),
) -> float:
    """Compose a single signed niceness score in [-1, +1] from raw weather."""
    components = {
        "cloud":         weights.cloud         * _cloud_niceness(weather.cloud_cover_pct),
        "precipitation": weights.precipitation * _precip_niceness(weather.precipitation_mm),
        "daylight":      weights.daylight      * _daylight_niceness(weather.is_daytime),
        "temperature":   weights.temperature   * _temperature_niceness(weather.temperature_c),
        "wind":          weights.wind          * _wind_niceness(weather.wind_kph),
    }
    raw = sum(components.values()) / max(weights.total(), 1e-6)
    return _clamp(raw)


def niceness_to_coefficients(
    niceness: float,
    *,
    happy_maxes: dict[int, float],
    sad_maxes: dict[int, float],
) -> dict[str, dict[int, float]]:
    """One-sided activation with per-layer maxes.

    Every happy/sad layer scales by the same niceness-derived factor but with
    its own calibrated `max`. At most one emotion group is non-zero at a time.

    niceness = +1.0 -> happy[L] = happy_maxes[L] for each L, sad[L] = 0
    niceness =  0.0 -> all zero (baseline behavior)
    niceness = -1.0 -> happy[L] = 0, sad[L] = sad_maxes[L] for each L
    """
    pos = max(0.0, niceness)
    neg = max(0.0, -niceness)
    return {
        "happy": {int(L): pos * float(m) for L, m in happy_maxes.items()},
        "sad":   {int(L): neg * float(m) for L, m in sad_maxes.items()},
    }


def weather_to_coefficients(
    weather: "CurrentWeather",
    *,
    happy_maxes: dict[int, float],
    sad_maxes: dict[int, float],
    weights: NicenessWeights = NicenessWeights(),
) -> dict[str, dict[int, float]]:
    """Convenience: weather -> niceness -> per-layer coefficients in one call."""
    n = compute_niceness(weather, weights=weights)
    return niceness_to_coefficients(n, happy_maxes=happy_maxes, sad_maxes=sad_maxes)
