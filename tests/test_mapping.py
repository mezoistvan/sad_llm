"""Tests for inputs/mapping.py.

The mapping module is pure: weather -> niceness -> coefficients. We test the
component functions and the integrated weather_to_coefficients with a few
realistic synthetic weather scenarios from the project design table.
"""

from __future__ import annotations

import time

import pytest

from inputs.mapping import (
    NicenessWeights,
    _cloud_niceness,
    _daylight_niceness,
    _precip_niceness,
    _temperature_niceness,
    _wind_niceness,
    compute_niceness,
    niceness_to_coefficients,
    weather_to_coefficients,
)
from inputs.weather_api import CurrentWeather


def make_weather(
    *,
    temperature_c: float = 18.0,
    cloud_cover_pct: float = 50.0,
    precipitation_mm: float = 0.0,
    wind_kph: float = 10.0,
    is_daytime: bool = True,
) -> CurrentWeather:
    return CurrentWeather(
        latitude=47.0, longitude=19.0,
        temperature_c=temperature_c,
        cloud_cover_pct=cloud_cover_pct,
        precipitation_mm=precipitation_mm,
        wind_kph=wind_kph,
        is_daytime=is_daytime,
        timezone="UTC",
        fetched_at=time.time(),
    )


# ---------- Component functions ----------

class TestCloudNiceness:
    def test_clear_sky_max(self):
        assert _cloud_niceness(0.0) == pytest.approx(1.0)

    def test_overcast_min(self):
        assert _cloud_niceness(100.0) == pytest.approx(-1.0)

    def test_half_cloudy_zero(self):
        assert _cloud_niceness(50.0) == pytest.approx(0.0)


class TestPrecipNiceness:
    def test_dry_mildly_positive(self):
        assert _precip_niceness(0.0) == pytest.approx(0.3)

    def test_drizzle_negative(self):
        # 0.5mm/h: 0.3 - 0.65*0.5 = -0.025
        assert _precip_niceness(0.5) == pytest.approx(-0.025)

    def test_one_mm_clearly_negative(self):
        # 1mm/h: 0.3 - 0.65 = -0.35
        assert _precip_niceness(1.0) == pytest.approx(-0.35)

    def test_steady_rain_floor(self):
        assert _precip_niceness(2.0) == pytest.approx(-1.0)

    def test_heavy_rain_clamped(self):
        assert _precip_niceness(10.0) == pytest.approx(-1.0)


class TestDaylightNiceness:
    def test_day_positive(self):
        assert _daylight_niceness(True) > 0

    def test_night_negative(self):
        assert _daylight_niceness(False) < 0


class TestTemperatureNiceness:
    def test_peak_at_22c(self):
        assert _temperature_niceness(22.0) == pytest.approx(1.0)

    def test_pleasant_at_18c(self):
        assert _temperature_niceness(18.0) > 0.4

    def test_pleasant_at_26c(self):
        assert _temperature_niceness(26.0) > 0.4

    def test_unpleasant_at_freezing(self):
        assert _temperature_niceness(0.0) < -0.5

    def test_unpleasant_at_hot(self):
        assert _temperature_niceness(38.0) < -0.5

    def test_extreme_asymptotes_toward_minus_one(self):
        # Gaussian asymptotes but never reaches -1 exactly; tolerate a small gap.
        assert _temperature_niceness(-30.0) == pytest.approx(-1.0, abs=1e-3)
        assert _temperature_niceness(60.0) == pytest.approx(-1.0, abs=1e-3)
        # Monotone decay past the unpleasant region.
        assert _temperature_niceness(-30.0) < _temperature_niceness(-10.0)
        assert _temperature_niceness(60.0) < _temperature_niceness(40.0)

    def test_symmetric(self):
        # Symmetric around 22°C peak
        assert _temperature_niceness(15.0) == pytest.approx(_temperature_niceness(29.0))

    def test_smooth_no_corners(self):
        # Gaussian should be strictly decreasing as we move away from the peak
        # in either direction — no piecewise kinks.
        below = [_temperature_niceness(t) for t in (22.0, 20.0, 15.0, 10.0, 5.0, 0.0)]
        above = [_temperature_niceness(t) for t in (22.0, 24.0, 29.0, 34.0, 39.0, 44.0)]
        assert all(a > b for a, b in zip(below, below[1:]))
        assert all(a > b for a, b in zip(above, above[1:]))


class TestWindNiceness:
    def test_calm_mildly_positive(self):
        assert _wind_niceness(0.0) == pytest.approx(0.3)
        assert _wind_niceness(5.0) == pytest.approx(0.3)

    def test_breeze_neutral_at_15(self):
        assert _wind_niceness(15.0) == pytest.approx(0.0)

    def test_breeze_decays_smoothly(self):
        # 10 kph: 0.3 - 0.03 * 5 = 0.15
        assert _wind_niceness(10.0) == pytest.approx(0.15)

    def test_strong_wind_negative(self):
        assert _wind_niceness(40.0) == pytest.approx(-1.0)

    def test_extreme_wind_clamped(self):
        assert _wind_niceness(80.0) == pytest.approx(-1.0)


# ---------- Integrated niceness ----------

class TestComputeNiceness:
    def test_perfect_summer_day(self):
        # Sunny noon at 22C -> strongly positive niceness (max achievable ~0.69)
        weather = make_weather(temperature_c=22.0, cloud_cover_pct=10.0,
                               precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        n = compute_niceness(weather)
        assert n > 0.6, f"sunny day niceness={n}, expected > 0.6"

    def test_winter_storm(self):
        # Cold, overcast, wind, raining, night
        weather = make_weather(temperature_c=-2.0, cloud_cover_pct=100.0,
                               precipitation_mm=2.0, wind_kph=35.0, is_daytime=False)
        n = compute_niceness(weather)
        assert n < -0.7, f"winter storm niceness={n}, expected < -0.7"

    def test_meh_overcast_10c(self):
        # 10C overcast still day, no precip -> mildly negative. Under the
        # Gaussian, 15°C is still noticeably pleasant (~0.4); 10°C lands us
        # in honest "meh" territory where the overcast sky tugs it negative.
        weather = make_weather(temperature_c=10.0, cloud_cover_pct=80.0,
                               precipitation_mm=0.0, wind_kph=12.0, is_daytime=True)
        n = compute_niceness(weather)
        assert -0.5 < n < 0.0, f"meh weather niceness={n}, expected mild negative"

    def test_clamped_to_unit_interval(self):
        # Hammered by every negative signal
        weather = make_weather(temperature_c=-30.0, cloud_cover_pct=100.0,
                               precipitation_mm=20.0, wind_kph=80.0, is_daytime=False)
        assert compute_niceness(weather) >= -1.0


# ---------- Coefficient mapping ----------

class TestNicenessToCoefficients:
    def test_full_summer_only_happy(self):
        c = niceness_to_coefficients(1.0, happy_max=0.5, sad_max=0.5)
        assert c["happy"] == pytest.approx(0.5)
        assert c["sad"] == pytest.approx(0.0)

    def test_full_winter_only_sad(self):
        c = niceness_to_coefficients(-1.0, happy_max=0.5, sad_max=0.5)
        assert c["happy"] == pytest.approx(0.0)
        assert c["sad"] == pytest.approx(0.5)

    def test_neutral_both_zero(self):
        c = niceness_to_coefficients(0.0, happy_max=0.5, sad_max=0.5)
        assert c["happy"] == pytest.approx(0.0)
        assert c["sad"] == pytest.approx(0.0)

    def test_one_sided_no_simultaneous(self):
        # For any niceness value, at most one of {happy, sad} should be > 0
        for niceness in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            c = niceness_to_coefficients(niceness, happy_max=0.5, sad_max=0.5)
            assert min(c["happy"], c["sad"]) == pytest.approx(0.0)
            assert c["happy"] >= 0 and c["sad"] >= 0

    def test_partial_summer_scaled(self):
        c = niceness_to_coefficients(0.5, happy_max=0.4, sad_max=0.4)
        assert c["happy"] == pytest.approx(0.2)
        assert c["sad"] == pytest.approx(0.0)


# ---------- End-to-end ----------

class TestWeatherToCoefficients:
    """Test the full pipeline against the synthetic weather scenarios from §9."""

    HAPPY_MAX = 0.5
    SAD_MAX = 0.5

    def test_sunny_noon_22c(self):
        w = make_weather(temperature_c=22.0, cloud_cover_pct=10.0,
                         precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_max=self.HAPPY_MAX, sad_max=self.SAD_MAX)
        # niceness ~0.69 * happy_max 0.5 = ~0.34
        assert c["happy"] > 0.25
        assert c["sad"] == pytest.approx(0.0)

    def test_overcast_drizzle_8c(self):
        w = make_weather(temperature_c=8.0, cloud_cover_pct=90.0,
                         precipitation_mm=0.5, wind_kph=15.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_max=self.HAPPY_MAX, sad_max=self.SAD_MAX)
        assert c["happy"] == pytest.approx(0.0)
        assert 0.05 < c["sad"] < 0.4

    def test_midnight_thunderstorm(self):
        w = make_weather(temperature_c=10.0, cloud_cover_pct=100.0,
                         precipitation_mm=4.0, wind_kph=40.0, is_daytime=False)
        c = weather_to_coefficients(w, happy_max=self.HAPPY_MAX, sad_max=self.SAD_MAX)
        assert c["happy"] == pytest.approx(0.0)
        assert c["sad"] > 0.3

    def test_clear_winter_night_minus5(self):
        w = make_weather(temperature_c=-5.0, cloud_cover_pct=10.0,
                         precipitation_mm=0.0, wind_kph=8.0, is_daytime=False)
        c = weather_to_coefficients(w, happy_max=self.HAPPY_MAX, sad_max=self.SAD_MAX)
        # Cold night but clear sky -> some sad
        assert c["happy"] == pytest.approx(0.0)
        assert c["sad"] > 0.0

    def test_mild_overcast_10c_baseline(self):
        # Mediocre weather: both should be near zero, model behaves baseline
        w = make_weather(temperature_c=10.0, cloud_cover_pct=80.0,
                         precipitation_mm=0.0, wind_kph=10.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_max=self.HAPPY_MAX, sad_max=self.SAD_MAX)
        assert max(c["happy"], c["sad"]) < 0.2  # both near baseline


class TestNicenessWeightsCustom:
    def test_custom_weights_change_outcome(self):
        # 28°C is far enough off-peak that headroom > 0, so the cloud weight
        # actually bites. (At the 22°C peak, headroom collapses to 0 and no
        # non-temperature weights can move the score.)
        weather = make_weather(temperature_c=28.0, cloud_cover_pct=100.0,
                               precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        n_default = compute_niceness(weather)
        n_no_cloud = compute_niceness(
            weather, weights=NicenessWeights(cloud=0.0)
        )
        assert n_no_cloud > n_default

    def test_peak_temperature_saturates(self):
        # At the Gaussian peak, headroom = 0: the temperature signal alone
        # pins the score to +1 regardless of how bad the other factors are.
        awful_sky = make_weather(temperature_c=22.0, cloud_cover_pct=100.0,
                                 precipitation_mm=2.0, wind_kph=40.0, is_daytime=False)
        perfect_sky = make_weather(temperature_c=22.0, cloud_cover_pct=0.0,
                                   precipitation_mm=0.0, wind_kph=3.0, is_daytime=True)
        assert compute_niceness(awful_sky) == pytest.approx(1.0)
        assert compute_niceness(perfect_sky) == pytest.approx(1.0)

    def test_headroom_suppresses_at_cold_extreme(self):
        # A sunny, dry, calm day at -30°C should still be extremely unpleasant:
        # the "niceness" of the sky shouldn't be able to rescue it.
        arctic_sunshine = make_weather(temperature_c=-30.0, cloud_cover_pct=0.0,
                                       precipitation_mm=0.0, wind_kph=2.0, is_daytime=True)
        assert compute_niceness(arctic_sunshine) < -0.95
