"""Metric correctness against values computed by hand.

Every expected number in this file is derived on paper from the definition in the
docstring of the function under test, not from a previous run of that function.
"""

import json

import numpy as np
import pytest

from tsfm_peft.metrics import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastMetrics,
    evaluate_forecasts,
    mase,
    quantile_losses,
    seasonal_naive_scale,
    smape,
    weighted_quantile_loss,
    wmape,
)


class TestSeasonalNaiveScale:
    def test_lag_one_on_unit_ramp(self):
        # |y_t - y_{t-1}| = 1 everywhere.
        assert seasonal_naive_scale([1, 2, 3, 4, 5, 6], seasonality=1) == pytest.approx(1.0)

    def test_seasonal_lag_uses_the_requested_period(self):
        # |y_t - y_{t-2}| = 2 for every one of the four valid pairs.
        assert seasonal_naive_scale([1, 2, 3, 4, 5, 6], seasonality=2) == pytest.approx(2.0)

    def test_hand_computed_irregular_series(self):
        # pairs: |3-1|=2, |7-3|=4, |4-7|=3, |10-4|=6 -> mean 15/4
        values = [1, 3, 7, 4, 10]
        assert seasonal_naive_scale(values, seasonality=1) == pytest.approx(3.75)

    def test_constant_series_is_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            seasonal_naive_scale([2.0] * 10, seasonality=1)

    def test_seasonally_constant_series_is_rejected(self):
        # Non-constant overall, but exactly periodic at lag 2.
        with pytest.raises(ValueError, match="undefined"):
            seasonal_naive_scale([1, 5, 1, 5, 1, 5], seasonality=2)

    def test_series_shorter_than_the_season_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            seasonal_naive_scale([1, 2, 3], seasonality=3)

    def test_non_positive_seasonality_is_rejected(self):
        with pytest.raises(ValueError, match="seasonality"):
            seasonal_naive_scale([1, 2, 3], seasonality=0)


class TestMase:
    def test_hand_computed(self):
        # abs errors 1, 0, 1 -> MAE 2/3; scale 2 -> MASE 1/3.
        result = mase([[1.0, 2.0, 3.0]], [[2.0, 2.0, 2.0]], [2.0])
        assert result == pytest.approx([1 / 3])

    def test_perfect_forecast_is_zero(self):
        assert mase([[1.0, 2.0]], [[1.0, 2.0]], [0.5]) == pytest.approx([0.0])

    def test_equals_one_when_the_error_matches_the_naive_scale(self):
        # A forecast whose MAE equals the seasonal-naive MAE scores exactly 1.0 -- the
        # interpretive anchor of MASE.
        train = [1, 2, 3, 4, 5, 6]
        scale = seasonal_naive_scale(train, seasonality=1)
        result = mase([[10.0, 10.0]], [[11.0, 9.0]], [scale])
        assert result == pytest.approx([1.0])

    def test_rows_are_scaled_independently(self):
        result = mase([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]], [1.0, 4.0])
        assert result == pytest.approx([1.0, 0.25])

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="strictly positive"):
            mase([[1.0]], [[1.0]], [0.0])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            mase([[1.0, 2.0]], [[1.0]], [1.0])

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            mase([[1.0, np.nan]], [[1.0, 1.0]], [1.0])


class TestSmape:
    def test_hand_computed(self):
        # 200 * |100 - 110| / (100 + 110) = 2000 / 210
        assert smape([[100.0]], [[110.0]]) == pytest.approx([2000 / 210])

    def test_perfect_forecast_is_zero(self):
        assert smape([[3.0, 4.0]], [[3.0, 4.0]]) == pytest.approx([0.0])

    def test_zero_over_zero_contributes_zero_not_nan(self):
        # A perfect prediction of zero is not an infinite percentage error.
        result = smape([[0.0, 100.0]], [[0.0, 100.0]])
        assert result == pytest.approx([0.0])

    def test_upper_bound_is_two_hundred(self):
        # Sign-flipped forecast of equal magnitude is the worst case.
        assert smape([[1.0, 5.0]], [[-1.0, -5.0]]) == pytest.approx([200.0])

    def test_is_symmetric_in_its_arguments(self):
        a, b = [[3.0, 9.0]], [[7.0, 2.0]]
        assert smape(a, b) == pytest.approx(smape(b, a))

    def test_averages_over_the_horizon_not_over_rows(self):
        # Row 0 steps: 200*0/200 = 0 and 200*10/210; row 1 is perfect.
        result = smape([[100.0, 100.0], [1.0, 1.0]], [[100.0, 110.0], [1.0, 1.0]])
        assert result == pytest.approx([(0 + 2000 / 210) / 2, 0.0])


class TestWmape:
    def test_hand_computed(self):
        # sum|e| = 10, sum|y| = 100 + 200 -> 100 * 10 / 300
        assert wmape([[100.0, 200.0]], [[110.0, 200.0]]) == pytest.approx(1000 / 300)

    def test_perfect_forecast_is_zero(self):
        assert wmape([[3.0, 4.0]], [[3.0, 4.0]]) == pytest.approx(0.0)

    def test_pools_rather_than_averaging_per_row(self):
        # A small-magnitude row with a proportionally large error would dominate a mean of
        # per-row ratios (100% on row 1); pooled, it is 1 part in 101.
        pooled = wmape([[100.0], [1.0]], [[100.0], [2.0]])
        assert pooled == pytest.approx(100 * 1 / 101)

    def test_a_near_zero_observation_cannot_blow_it_up(self):
        # The MAPE failure mode: dividing a fixed error by an arbitrarily small observation.
        # Here the denominator is the total, so the metric stays finite as y -> 0.
        assert wmape([[1000.0, 1e-9]], [[1000.0, 1.0]]) < 1.0

    def test_is_scale_invariant(self):
        assert wmape([[10.0, 20.0]], [[11.0, 19.0]]) == pytest.approx(
            wmape([[1000.0, 2000.0]], [[1100.0, 1900.0]])
        )

    def test_exactly_predicted_zeros_contribute_zero_not_an_error(self):
        assert wmape([[0.0, 0.0]], [[0.0, 0.0]]) == pytest.approx(0.0)

    def test_rejects_an_all_zero_target_that_was_not_predicted(self):
        # A relative error measured against nothing. Better to refuse than to publish a
        # stand-in that reads like a measurement.
        with pytest.raises(ValueError, match="undefined"):
            wmape([[0.0, 0.0]], [[0.0, 1.0]])

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            wmape([[1.0, 2.0]], [[1.0]])


class TestQuantileLoss:
    def test_median_level_reduces_to_normalised_mae(self):
        # 2 * sum(0.5 * |e|) / sum(|y|) = sum|e| / sum|y| = 1 / 3
        per_level, denom = quantile_losses([[1.0, 2.0]], [[[2.0], [2.0]]], levels=[0.5])
        assert denom == pytest.approx(3.0)
        assert per_level == pytest.approx([1 / 3])

    def test_high_quantile_penalises_under_forecasting_more(self):
        # q=0.9, y=10: under-forecast by 2 -> 2*0.9*2/10 = 0.36
        under, _ = quantile_losses([[10.0]], [[[8.0]]], levels=[0.9])
        # over-forecast by 2 -> 2*0.1*2/10 = 0.04
        over, _ = quantile_losses([[10.0]], [[[12.0]]], levels=[0.9])
        assert under == pytest.approx([0.36])
        assert over == pytest.approx([0.04])
        assert under[0] > over[0]

    def test_low_quantile_penalises_over_forecasting_more(self):
        under, _ = quantile_losses([[10.0]], [[[8.0]]], levels=[0.1])
        over, _ = quantile_losses([[10.0]], [[[12.0]]], levels=[0.1])
        assert under == pytest.approx([0.04])
        assert over == pytest.approx([0.36])

    def test_perfect_quantiles_give_zero(self):
        preds = np.zeros((1, 2, 9)) + np.array([[1.0], [2.0]])
        assert weighted_quantile_loss([[1.0, 2.0]], preds) == pytest.approx(0.0)

    def test_wql_is_the_mean_over_levels(self):
        preds = np.stack([np.full((1, 2), 3.0), np.full((1, 2), 5.0)], axis=-1)
        per_level, _ = quantile_losses([[4.0, 4.0]], preds, levels=[0.25, 0.75])
        # level 0.25, e=+1 twice: 2*0.25*2/8 = 0.125
        # level 0.75, e=-1 twice: 2*0.25*2/8 = 0.125
        assert per_level == pytest.approx([0.125, 0.125])
        assert weighted_quantile_loss([[4.0, 4.0]], preds, levels=[0.25, 0.75]) == pytest.approx(
            0.125
        )

    def test_rejects_unsorted_levels(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            quantile_losses([[1.0]], np.ones((1, 1, 2)), levels=[0.9, 0.1])

    def test_rejects_levels_outside_the_open_unit_interval(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            quantile_losses([[1.0]], np.ones((1, 1, 2)), levels=[0.0, 0.5])

    def test_rejects_prediction_shape_mismatch(self):
        with pytest.raises(ValueError, match="must have shape"):
            quantile_losses([[1.0, 2.0]], np.ones((1, 2, 3)), levels=[0.25, 0.75])

    def test_all_zero_targets_are_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            quantile_losses([[0.0, 0.0]], np.zeros((1, 2, 1)), levels=[0.5])


class TestEvaluateForecasts:
    @staticmethod
    def _flat_quantiles(point: np.ndarray, n_levels: int) -> np.ndarray:
        """Broadcast a point forecast across every quantile level."""
        return np.repeat(point[..., None], n_levels, axis=-1)

    def test_aggregates_and_per_series_breakdown(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0], [10.0, 10.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 5.0], [11.0, 11.0], [9.0, 9.0]])
        preds_q = self._flat_quantiles(y_pred, len(DEFAULT_QUANTILE_LEVELS))
        series_ids = ["a", "a", "b", "b"]
        scales = np.array([1.0, 1.0, 2.0, 2.0])

        result = evaluate_forecasts(y_true, y_pred, preds_q, series_ids, scales)

        assert isinstance(result, ForecastMetrics)
        assert result.n_series == 2
        assert result.n_rows == 4
        assert result.horizon == 2
        # Row MASEs: 0, 0.5/1, 1/2, 1/2 -> mean 0.375
        assert result.mase == pytest.approx(0.375)
        # Series a rows: 0 and 0.5; series b rows: 0.5 and 0.5.
        assert result.per_series["a"]["mase"] == pytest.approx(0.25)
        assert result.per_series["b"]["mase"] == pytest.approx(0.5)
        assert result.per_series["a"]["n_windows"] == 2

    def test_pooled_wql_is_dominated_by_the_large_series_but_macro_is_not(self):
        # Small series is forecast perfectly; large series is off by 10%.
        y_true = np.array([[1.0, 1.0], [1000.0, 1000.0]])
        y_pred = np.array([[1.0, 1.0], [1100.0, 1100.0]])
        preds_q = self._flat_quantiles(y_pred, 1)
        result = evaluate_forecasts(
            y_true, y_pred, preds_q, ["small", "large"], [1.0, 1.0], levels=[0.5]
        )
        # Pooled: sum|e| / sum|y| = 200 / 2002
        assert result.wql == pytest.approx(200 / 2002)
        # Macro: (0 + 200/2000) / 2 = 0.05
        assert result.wql_macro == pytest.approx(0.05)

    def test_pooled_wmape_is_dominated_by_the_large_series_but_macro_is_not(self):
        # Same shape of argument as the WQL case above, and the same reason for reporting
        # both: over channels of different scale the pooled number is the large channel.
        y_true = np.array([[1.0, 1.0], [1000.0, 1000.0]])
        y_pred = np.array([[2.0, 2.0], [1100.0, 1100.0]])
        preds_q = self._flat_quantiles(y_pred, 1)
        result = evaluate_forecasts(
            y_true, y_pred, preds_q, ["small", "large"], [1.0, 1.0], levels=[0.5]
        )
        # Pooled: 100 * (2 + 200) / 2002 -- the small series is 1% of the denominator.
        assert result.wmape == pytest.approx(100 * 202 / 2002)
        # Macro: (100% + 10%) / 2, where the small series' own error is 100% of its size.
        assert result.per_series["small"]["wmape"] == pytest.approx(100.0)
        assert result.per_series["large"]["wmape"] == pytest.approx(10.0)
        assert result.wmape_macro == pytest.approx(55.0)

    def test_wmape_survives_a_round_trip_through_the_artifact_dict(self):
        y_true = np.array([[10.0, 20.0]])
        y_pred = np.array([[11.0, 20.0]])
        preds_q = self._flat_quantiles(y_pred, 1)
        payload = evaluate_forecasts(y_true, y_pred, preds_q, ["s"], [1.0], levels=[0.5]).to_dict()
        assert payload["wmape"] == pytest.approx(100 / 30)
        assert payload["wmape_macro"] == pytest.approx(100 / 30)
        assert payload["per_series"]["s"]["wmape"] == pytest.approx(100 / 30)

    def test_per_quantile_losses_are_keyed_by_level(self):
        y_true = np.array([[4.0]])
        preds_q = np.array([[[3.0, 5.0]]])
        result = evaluate_forecasts(
            y_true, np.array([[4.0]]), preds_q, ["s"], [1.0], levels=[0.25, 0.75]
        )
        assert set(result.per_quantile_wql) == {"0.25", "0.75"}
        assert result.per_quantile_wql["0.25"] == pytest.approx(0.125)

    def test_to_dict_is_json_serialisable(self):
        y_true = np.array([[1.0, 2.0]])
        preds_q = self._flat_quantiles(y_true, len(DEFAULT_QUANTILE_LEVELS))
        result = evaluate_forecasts(y_true, y_true, preds_q, ["s"], [1.0])
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["n_series"] == 1
        assert payload["quantile_levels"] == list(DEFAULT_QUANTILE_LEVELS)
        assert payload["per_series"]["s"]["mase"] == pytest.approx(0.0)

    def test_rejects_series_id_length_mismatch(self):
        y_true = np.array([[1.0], [2.0]])
        preds_q = self._flat_quantiles(y_true, 1)
        with pytest.raises(ValueError, match="series_ids"):
            evaluate_forecasts(y_true, y_true, preds_q, ["only-one"], [1.0, 1.0], levels=[0.5])
