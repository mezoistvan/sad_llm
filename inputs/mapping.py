"""Weather -> steering coefficients.

Pure, side-effect-free, easy to unit-test. The whole module is deliberately
tiny so the mapping logic stays inspectable. Tunables live at the top.

Pipeline:
    weather  ->  niceness in [-1, +1]  ->  {happy, sad} coefficients in fraction-of-norm units

`happy` and `sad` are non-negative and one of them is always zero — niceness > 0
activates only happy, niceness < 0 activates only sad. Avoids the muddle of
both vectors being half-active simultaneously.

Design note — Gaussian temperature + adaptive headroom:
    Temperature is the backbone of the score. It's mapped through a Gaussian
    bell centred on TEMP_PEAK_C and contributes to niceness unconditionally.
    The other components (cloud, precipitation, daylight, wind) contribute
    only within a *headroom* budget `max(0, 1 - |temp_score|)`, which collapses
    toward 0 as temperature approaches either extreme. This prevents a sunny,
    calm, dry day at -40°C from scoring "pleasant" on the strength of the
    non-temperature factors alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inputs.weather_api import CurrentWeather


# Tunables. Each component's niceness lives in [-1, +1].

@dataclass(frozen=True)
class NicenessWeights:
    """Weights for the non-temperature components. Each multiplies its
    component's niceness (in [-1, +1]); the weighted sum is then scaled by
    the temperature-derived headroom and added to the temperature score.

    Temperature has no explicit weight — it contributes unconditionally with
    an implicit weight of 1.0. Defaults here sum to 0.65, meaning in the
    best case (neutral temperature, everything else perfect) the other
    components can swing the score by up to ±0.65 from the temperature
    backbone.
    """
    cloud: float = 0.25
    precipitation: float = 0.20
    daylight: float = 0.10
    wind: float = 0.10

    def total(self) -> float:
        return self.cloud + self.precipitation + self.daylight + self.wind


# Temperature is a Gaussian bell: +1 at the peak, asymptotes to -1 at extremes.
# TEMP_SIGMA_C controls the width: at |T - peak| = sigma, niceness ≈ -0.26;
# at 2*sigma, ≈ -0.96. sigma ≈ 12°C roughly matches the "22°F" sigma the
# design note was drafted against.
TEMP_PEAK_C = 22.0
TEMP_SIGMA_C = 12.0


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
    """Gaussian bell centred at TEMP_PEAK_C.

    +1.0 at the peak, smoothly decaying to an asymptote at -1.0 for extreme
    hot or cold. Formula: ``2 * exp(-((T - peak) / sigma)^2) - 1``. Smooth
    everywhere, so the headroom term downstream transitions gently as
    temperature moves through its sweet spot.
    """
    delta = temp_c - TEMP_PEAK_C
    return 2.0 * math.exp(-((delta / TEMP_SIGMA_C) ** 2)) - 1.0


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
    """Compose a single signed niceness score in [-1, +1] from raw weather.

    Temperature forms the backbone (unconditional contribution). The remaining
    components contribute only within the ``headroom = max(0, 1 - |t|)`` budget,
    so extreme temperatures saturate the score regardless of how nice the sky
    or wind happens to be. At the peak temperature (22°C) ``headroom`` is also
    zero: temperature alone pins the score to +1.
    """
    t = _temperature_niceness(weather.temperature_c)
    others = (
        weights.cloud         * _cloud_niceness(weather.cloud_cover_pct)
        + weights.precipitation * _precip_niceness(weather.precipitation_mm)
        + weights.daylight      * _daylight_niceness(weather.is_daytime)
        + weights.wind          * _wind_niceness(weather.wind_kph)
    )
    headroom = max(0.0, 1.0 - abs(t))
    return _clamp(t + headroom * others)


def niceness_to_coefficients(
    niceness: float,
    *,
    happy_max: float,
    sad_max: float,
) -> dict[str, float]:
    """One-sided activation: only one vector is on at a time.

    niceness = +1.0 -> happy = happy_max, sad = 0
    niceness =  0.0 -> happy = 0,         sad = 0       (baseline)
    niceness = -1.0 -> happy = 0,         sad = sad_max
    """
    return {
        "happy": max(0.0, niceness) * happy_max,
        "sad":   max(0.0, -niceness) * sad_max,
    }


def weather_to_coefficients(
    weather: "CurrentWeather",
    *,
    happy_max: float,
    sad_max: float,
    weights: NicenessWeights = NicenessWeights(),
) -> dict[str, float]:
    """Convenience: weather -> niceness -> coefficients in one call."""
    n = compute_niceness(weather, weights=weights)
    return niceness_to_coefficients(n, happy_max=happy_max, sad_max=sad_max)
