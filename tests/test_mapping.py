"""Tests for inputs/mapping.py.

The mapping module is pure: weather -> niceness -> coefficients. We test the
component functions and the integrated weather_to_coefficients with a few
realistic synthetic weather scenarios from the SAD-LLM design table.
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

    def test_extreme_clamped(self):
        assert _temperature_niceness(-30.0) == pytest.approx(-1.0)
        assert _temperature_niceness(60.0) == pytest.approx(-1.0)

    def test_symmetric(self):
        # Symmetric around 22°C peak
        assert _temperature_niceness(15.0) == pytest.approx(_temperature_niceness(29.0))


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

    def test_meh_overcast_15c(self):
        # 15C overcast still day, no precip -> mildly negative
        weather = make_weather(temperature_c=15.0, cloud_cover_pct=80.0,
                               precipitation_mm=0.0, wind_kph=12.0, is_daytime=True)
        n = compute_niceness(weather)
        assert -0.4 < n < 0.1, f"meh weather niceness={n}, expected mild negative"

    def test_clamped_to_unit_interval(self):
        # Hammered by every negative signal
        weather = make_weather(temperature_c=-30.0, cloud_cover_pct=100.0,
                               precipitation_mm=20.0, wind_kph=80.0, is_daytime=False)
        assert compute_niceness(weather) >= -1.0


# ---------- Coefficient mapping ----------

# Single-layer configurations preserve the production (happy@L19, sad@L17) shape.
SINGLE_HAPPY = {19: 0.5}
SINGLE_SAD = {17: 0.5}

# Two-layer configurations exercise the per-layer max semantic: each layer
# scales by the same niceness-derived factor but with its own max.
PAIR_HAPPY = {19: 0.5, 20: 0.3}
PAIR_SAD = {17: 0.5, 20: 0.4}


class TestNicenessToCoefficientsSingleLayer:
    def test_full_summer_only_happy(self):
        c = niceness_to_coefficients(1.0, happy_maxes=SINGLE_HAPPY, sad_maxes=SINGLE_SAD)
        assert c["happy"] == {19: pytest.approx(0.5)}
        assert c["sad"] == {17: pytest.approx(0.0)}

    def test_full_winter_only_sad(self):
        c = niceness_to_coefficients(-1.0, happy_maxes=SINGLE_HAPPY, sad_maxes=SINGLE_SAD)
        assert c["happy"] == {19: pytest.approx(0.0)}
        assert c["sad"] == {17: pytest.approx(0.5)}

    def test_neutral_both_zero(self):
        c = niceness_to_coefficients(0.0, happy_maxes=SINGLE_HAPPY, sad_maxes=SINGLE_SAD)
        assert c["happy"] == {19: pytest.approx(0.0)}
        assert c["sad"] == {17: pytest.approx(0.0)}

    def test_one_sided_no_simultaneous(self):
        # For any niceness value, at most one emotion group should have non-zero
        # entries. The other group's entries must all be exactly 0.
        for niceness in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            c = niceness_to_coefficients(niceness, happy_maxes=SINGLE_HAPPY, sad_maxes=SINGLE_SAD)
            max_happy = max(c["happy"].values())
            max_sad = max(c["sad"].values())
            assert min(max_happy, max_sad) == pytest.approx(0.0)
            assert all(v >= 0 for v in c["happy"].values())
            assert all(v >= 0 for v in c["sad"].values())

    def test_partial_summer_scaled(self):
        c = niceness_to_coefficients(0.5, happy_maxes={19: 0.4}, sad_maxes={17: 0.4})
        assert c["happy"][19] == pytest.approx(0.2)
        assert c["sad"][17] == pytest.approx(0.0)


class TestNicenessToCoefficientsTwoLayer:
    def test_full_summer_both_happy_layers_at_max(self):
        c = niceness_to_coefficients(1.0, happy_maxes=PAIR_HAPPY, sad_maxes=SINGLE_SAD)
        assert c["happy"][19] == pytest.approx(0.5)
        assert c["happy"][20] == pytest.approx(0.3)
        assert c["sad"][17] == pytest.approx(0.0)

    def test_full_winter_both_sad_layers_at_max(self):
        c = niceness_to_coefficients(-1.0, happy_maxes=SINGLE_HAPPY, sad_maxes=PAIR_SAD)
        assert c["happy"][19] == pytest.approx(0.0)
        assert c["sad"][17] == pytest.approx(0.5)
        assert c["sad"][20] == pytest.approx(0.4)

    def test_partial_niceness_scales_every_layer(self):
        # niceness=+0.5 should scale every happy layer to half its max,
        # and leave all sad layers at zero.
        c = niceness_to_coefficients(0.5, happy_maxes=PAIR_HAPPY, sad_maxes=PAIR_SAD)
        assert c["happy"][19] == pytest.approx(0.25)
        assert c["happy"][20] == pytest.approx(0.15)
        assert c["sad"][17] == pytest.approx(0.0)
        assert c["sad"][20] == pytest.approx(0.0)

    def test_partial_winter_scales_every_sad_layer(self):
        c = niceness_to_coefficients(-0.5, happy_maxes=PAIR_HAPPY, sad_maxes=PAIR_SAD)
        assert c["happy"][19] == pytest.approx(0.0)
        assert c["happy"][20] == pytest.approx(0.0)
        assert c["sad"][17] == pytest.approx(0.25)
        assert c["sad"][20] == pytest.approx(0.20)

    def test_neutral_all_layers_zero(self):
        c = niceness_to_coefficients(0.0, happy_maxes=PAIR_HAPPY, sad_maxes=PAIR_SAD)
        assert all(v == pytest.approx(0.0) for v in c["happy"].values())
        assert all(v == pytest.approx(0.0) for v in c["sad"].values())

    def test_one_sided_preserved_across_all_layers(self):
        # With two layers per emotion, the one-sided property must still hold:
        # for any niceness, either every happy layer is zero or every sad layer is zero.
        for niceness in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            c = niceness_to_coefficients(niceness, happy_maxes=PAIR_HAPPY, sad_maxes=PAIR_SAD)
            max_happy = max(c["happy"].values())
            max_sad = max(c["sad"].values())
            assert min(max_happy, max_sad) == pytest.approx(0.0)

    def test_returned_dicts_include_every_configured_layer(self):
        c = niceness_to_coefficients(0.7, happy_maxes=PAIR_HAPPY, sad_maxes=PAIR_SAD)
        assert set(c["happy"].keys()) == {19, 20}
        assert set(c["sad"].keys()) == {17, 20}


# ---------- End-to-end ----------

class TestWeatherToCoefficients:
    """Test the full pipeline against the synthetic weather scenarios from §9."""

    HAPPY_MAXES = {19: 0.5}
    SAD_MAXES = {17: 0.5}

    def test_sunny_noon_22c(self):
        w = make_weather(temperature_c=22.0, cloud_cover_pct=10.0,
                         precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_maxes=self.HAPPY_MAXES, sad_maxes=self.SAD_MAXES)
        # niceness ~0.69 * happy_max 0.5 = ~0.34
        assert c["happy"][19] > 0.25
        assert c["sad"][17] == pytest.approx(0.0)

    def test_overcast_drizzle_8c(self):
        w = make_weather(temperature_c=8.0, cloud_cover_pct=90.0,
                         precipitation_mm=0.5, wind_kph=15.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_maxes=self.HAPPY_MAXES, sad_maxes=self.SAD_MAXES)
        assert c["happy"][19] == pytest.approx(0.0)
        assert 0.05 < c["sad"][17] < 0.4

    def test_midnight_thunderstorm(self):
        w = make_weather(temperature_c=10.0, cloud_cover_pct=100.0,
                         precipitation_mm=4.0, wind_kph=40.0, is_daytime=False)
        c = weather_to_coefficients(w, happy_maxes=self.HAPPY_MAXES, sad_maxes=self.SAD_MAXES)
        assert c["happy"][19] == pytest.approx(0.0)
        assert c["sad"][17] > 0.3

    def test_clear_winter_night_minus5(self):
        w = make_weather(temperature_c=-5.0, cloud_cover_pct=10.0,
                         precipitation_mm=0.0, wind_kph=8.0, is_daytime=False)
        c = weather_to_coefficients(w, happy_maxes=self.HAPPY_MAXES, sad_maxes=self.SAD_MAXES)
        # Cold night but clear sky -> some sad
        assert c["happy"][19] == pytest.approx(0.0)
        assert c["sad"][17] > 0.0

    def test_mild_overcast_15c_baseline(self):
        # Mediocre weather: both should be near zero, model behaves baseline
        w = make_weather(temperature_c=15.0, cloud_cover_pct=80.0,
                         precipitation_mm=0.0, wind_kph=10.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_maxes=self.HAPPY_MAXES, sad_maxes=self.SAD_MAXES)
        assert max(c["happy"][19], c["sad"][17]) < 0.2  # both near baseline

    def test_two_layer_sunny_day_scales_all_happy_layers(self):
        # With two happy layers, a sunny day should drive both to a
        # proportional fraction of their own max.
        w = make_weather(temperature_c=22.0, cloud_cover_pct=10.0,
                         precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        c = weather_to_coefficients(w, happy_maxes={19: 0.5, 20: 0.3}, sad_maxes={17: 0.5})
        # Both happy layers should be scaled by the same niceness, so their
        # coefficients should preserve the max ratio 0.5 / 0.3.
        assert c["sad"][17] == pytest.approx(0.0)
        assert c["happy"][19] > 0
        assert c["happy"][20] > 0
        assert c["happy"][19] / c["happy"][20] == pytest.approx(0.5 / 0.3, rel=1e-6)


class TestNicenessWeightsCustom:
    def test_custom_weights_change_outcome(self):
        weather = make_weather(temperature_c=22.0, cloud_cover_pct=100.0,
                               precipitation_mm=0.0, wind_kph=5.0, is_daytime=True)
        # Default weights: cloud is dominant (0.30) so 100% cloud should drag
        # niceness substantially negative even with perfect temp
        n_default = compute_niceness(weather)
        # If we zero out cloud weight, the niceness should improve
        n_no_cloud = compute_niceness(
            weather, weights=NicenessWeights(cloud=0.0)
        )
        assert n_no_cloud > n_default
