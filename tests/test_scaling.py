"""Scaler arithmetic, invertibility, and that fitting never sees evaluation data."""

import numpy as np
import pytest

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.scaling import (
    SCALERS,
    AffineScaler,
    fit_mean_abs,
    fit_scaler,
    fit_scalers,
    fit_standard,
)
from tsfm_peft.data.windows import BacktestProtocol, make_split, plan_split


def protocol(**overrides: object) -> BacktestProtocol:
    defaults = {"horizon": 10, "context_length": 20, "n_test_windows": 3}
    return BacktestProtocol(**{**defaults, **overrides})


class TestAffineScaler:
    def test_hand_computed_transform(self):
        scaler = AffineScaler(center=2.0, scale=4.0)
        assert scaler.transform([2.0, 6.0, -2.0]).tolist() == [0.0, 1.0, -1.0]

    def test_round_trips(self):
        scaler = AffineScaler(center=-3.5, scale=0.25)
        values = np.linspace(-10, 10, 21)
        assert scaler.inverse_transform(scaler.transform(values)) == pytest.approx(values)

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="strictly positive"):
            AffineScaler(center=0.0, scale=0.0)

    def test_rejects_non_finite_parameters(self):
        with pytest.raises(ValueError, match="finite"):
            AffineScaler(center=float("nan"), scale=1.0)

    def test_to_dict_records_the_fitting_rule(self):
        assert fit_standard([0.0, 2.0]).to_dict() == {
            "kind": "standard",
            "center": 1.0,
            "scale": 1.0,
        }


class TestFittingRules:
    def test_standard_matches_mean_and_population_std(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        scaler = fit_standard(values)
        assert scaler.center == pytest.approx(2.5)
        # Population std: sqrt(mean([2.25, 0.25, 0.25, 2.25])) = sqrt(1.25)
        assert scaler.scale == pytest.approx(np.sqrt(1.25))

    def test_standard_falls_back_on_a_constant_region(self):
        scaler = fit_standard(np.full(10, 7.0))
        assert scaler.scale > 0
        assert scaler.transform(np.full(3, 7.0)).tolist() == [0.0, 0.0, 0.0]

    def test_mean_abs_does_not_shift_zero(self):
        # Count-like series: zero must stay zero after scaling.
        scaler = fit_mean_abs([0.0, 0.0, 4.0, 8.0])
        assert scaler.center == 0.0
        assert scaler.scale == pytest.approx(3.0)
        assert scaler.transform([0.0]).tolist() == [0.0]

    def test_identity_leaves_values_untouched(self):
        values = [3.0, -1.0, 0.5]
        assert fit_scaler(values, "identity").transform(values).tolist() == values

    def test_every_registered_rule_is_reachable_by_name(self):
        values = np.arange(1.0, 11.0)
        for kind in SCALERS:
            assert fit_scaler(values, kind).kind == kind

    def test_unknown_rule_lists_the_available_ones(self):
        with pytest.raises(ValueError, match="identity"):
            fit_scaler([1.0, 2.0], "minmax")

    def test_rejects_empty_and_non_finite_input(self):
        with pytest.raises(ValueError, match="empty"):
            fit_standard([])
        with pytest.raises(ValueError, match="NaN"):
            fit_standard([1.0, np.nan])

    def test_rejects_two_dimensional_input(self):
        with pytest.raises(ValueError, match="1-D"):
            fit_standard([[1.0, 2.0], [3.0, 4.0]])


class TestNoLeakage:
    def test_fit_scalers_matches_fitting_on_the_fit_region_alone(self, dataset):
        split = make_split(dataset, protocol(n_val_windows=2))
        scalers = fit_scalers(split, "standard")
        for series in dataset:
            fit_end = split.plans[series.series_id].fit_end
            expected = fit_standard(series.values[:fit_end])
            assert scalers[series.series_id].center == pytest.approx(expected.center)
            assert scalers[series.series_id].scale == pytest.approx(expected.scale)

    def test_scalers_differ_from_a_full_series_fit(self, dataset):
        # If these agreed, the test above could not distinguish a leak-free fit from a
        # leaky one; the fixture has a trend, so the two genuinely differ.
        split = make_split(dataset, protocol())
        scalers = fit_scalers(split, "standard")
        for series in dataset:
            leaky = fit_standard(series.values)
            assert scalers[series.series_id].center != pytest.approx(leaky.center)

    def test_poisoning_everything_past_fit_end_does_not_move_the_scaler(self, make_dataset):
        base = make_dataset(n_series=1, length=200, seasonality=4)
        proto = protocol(n_val_windows=2)
        fit_end = plan_split("s0", 200, proto).fit_end
        poisoned = base["s0"].values.copy()
        poisoned[fit_end:] = 1e9
        contaminated = TimeSeriesDataset.from_arrays(
            "poisoned", [("s0", poisoned)], freq="h", seasonality=4
        )

        clean_scaler = fit_scalers(make_split(base, proto))["s0"]
        poisoned_scaler = fit_scalers(make_split(contaminated, proto))["s0"]

        assert clean_scaler.center == pytest.approx(poisoned_scaler.center)
        assert clean_scaler.scale == pytest.approx(poisoned_scaler.scale)

    def test_forecasts_must_be_inverted_before_scoring(self, dataset):
        # Guards the intended usage: metrics are scale-dependent, so a forecast produced in
        # scaled space and scored without inversion would be measuring something else.
        split = make_split(dataset, protocol())
        scaler = fit_scalers(split)["s0"]
        window = split.test_windows[0]
        scaled_target = scaler.transform(window.target)
        assert scaler.inverse_transform(scaled_target) == pytest.approx(window.target)
        assert not np.allclose(scaled_target, window.target)
