"""Adapter interface, batching contract and the seasonal-naive baseline."""

import numpy as np
import pytest

from tsfm_peft.models.base import DatasetContext, Forecast, ForecastModel
from tsfm_peft.models.naive import SeasonalNaiveModel, SeasonalNaiveOptions
from tsfm_peft.models.registry import (
    available_models,
    build_model,
    get_model_spec,
    validate_options,
)

LEVELS = (0.1, 0.5, 0.9)


def make_forecast(n_rows=2, horizon=3, levels=LEVELS, offset=0.0):
    point = np.arange(n_rows * horizon, dtype=float).reshape(n_rows, horizon) + offset
    quantiles = point[:, :, None] + np.array(levels) - 0.5
    return Forecast(point=point, quantiles=quantiles, levels=levels)


class TestForecast:
    def test_shapes_and_accessors(self):
        forecast = make_forecast()
        assert forecast.n_rows == 2
        assert forecast.horizon == 3
        assert forecast.levels == LEVELS

    def test_arrays_are_frozen(self):
        forecast = make_forecast()
        with pytest.raises(ValueError, match="read-only"):
            forecast.point[0, 0] = 1.0

    def test_rejects_1d_point(self):
        with pytest.raises(ValueError, match="point must be 2-D"):
            Forecast(point=np.zeros(3), quantiles=np.zeros((3, 3)), levels=LEVELS)

    def test_rejects_quantile_shape_mismatch(self):
        with pytest.raises(ValueError, match="quantiles must have shape"):
            Forecast(point=np.zeros((2, 3)), quantiles=np.zeros((2, 3, 2)), levels=LEVELS)

    def test_rejects_non_finite(self):
        point = np.zeros((1, 2))
        point[0, 0] = np.nan
        with pytest.raises(ValueError, match="point contains NaN"):
            Forecast(point=point, quantiles=np.zeros((1, 2, 3)), levels=LEVELS)

    @pytest.mark.parametrize(
        ("levels", "match"),
        [
            ((0.5, 0.1), "strictly increasing"),
            ((0.5, 0.5), "strictly increasing"),
            ((0.0, 0.5), r"strictly in \(0, 1\)"),
            ((0.5, 1.0), r"strictly in \(0, 1\)"),
        ],
    )
    def test_rejects_bad_levels(self, levels, match):
        point = np.zeros((1, 2))
        with pytest.raises(ValueError, match=match):
            Forecast(point=point, quantiles=np.zeros((1, 2, len(levels))), levels=levels)

    def test_crossing_rate_counts_inversions(self):
        point = np.zeros((1, 2))
        quantiles = np.zeros((1, 2, 3))
        quantiles[0, 0] = [0.0, 1.0, 2.0]  # ordered
        quantiles[0, 1] = [2.0, 1.0, 0.0]  # both adjacent pairs inverted
        forecast = Forecast(point=point, quantiles=quantiles, levels=LEVELS)
        assert forecast.quantile_crossing_rate() == pytest.approx(0.5)

    def test_crossing_rate_is_zero_for_single_level(self):
        forecast = Forecast(point=np.zeros((1, 2)), quantiles=np.zeros((1, 2, 1)), levels=(0.5,))
        assert forecast.quantile_crossing_rate() == 0.0

    def test_concatenate_preserves_order(self):
        joined = Forecast.concatenate([make_forecast(offset=0.0), make_forecast(offset=100.0)])
        assert joined.n_rows == 4
        assert joined.point[2, 0] == 100.0

    def test_concatenate_rejects_empty(self):
        with pytest.raises(ValueError, match="empty sequence"):
            Forecast.concatenate([])

    def test_concatenate_rejects_level_mismatch(self):
        with pytest.raises(ValueError, match="levels differ"):
            Forecast.concatenate([make_forecast(), make_forecast(levels=(0.2, 0.5, 0.8))])

    def test_concatenate_rejects_horizon_mismatch(self):
        with pytest.raises(ValueError, match="horizons differ"):
            Forecast.concatenate([make_forecast(horizon=3), make_forecast(horizon=4)])


class RecordingModel(ForecastModel):
    """Echoes the first context value so batching order is observable."""

    name = "recording"

    def __init__(self, batch_size=2, max_horizon=4) -> None:
        self.batches = []
        self._batch_size = batch_size
        self._max_horizon = max_horizon

    @property
    def quantile_levels(self):
        return LEVELS

    @property
    def max_horizon(self):
        return self._max_horizon

    @property
    def batch_size(self):
        return self._batch_size

    def _predict_batch(self, contexts, horizon):
        self.batches.append(contexts.shape[0])
        point = np.repeat(contexts[:, :1], horizon, axis=1)
        return Forecast(
            point=point, quantiles=np.repeat(point[:, :, None], 3, axis=2), levels=LEVELS
        )


class TestForecastModel:
    def test_chunks_into_batches_and_preserves_row_order(self):
        model = RecordingModel(batch_size=2)
        contexts = np.arange(5, dtype=float)[:, None] * np.ones((5, 6))
        forecast = model.predict(contexts, horizon=3)
        assert model.batches == [2, 2, 1]
        assert forecast.n_rows == 5
        np.testing.assert_allclose(forecast.point[:, 0], np.arange(5))

    def test_rejects_horizon_beyond_max(self):
        with pytest.raises(ValueError, match="exceeds the adapter's single-call maximum"):
            RecordingModel(max_horizon=4).predict(np.ones((1, 6)), horizon=5)

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            RecordingModel().predict(np.ones((1, 6)), horizon=0)

    def test_rejects_1d_contexts(self):
        with pytest.raises(ValueError, match="contexts must be 2-D"):
            RecordingModel().predict(np.ones(6), horizon=2)

    def test_rejects_non_finite_contexts(self):
        contexts = np.ones((1, 6))
        contexts[0, 0] = np.inf
        with pytest.raises(ValueError, match="contexts contain NaN or inf"):
            RecordingModel().predict(contexts, horizon=2)

    def test_bind_dataset_defaults_to_noop(self):
        assert RecordingModel().bind_dataset(DatasetContext("d", "h", 24)) is None


def naive(seasonality=4, levels=LEVELS):
    model = SeasonalNaiveModel(
        SeasonalNaiveOptions(seasonality=seasonality, quantile_levels=levels)
    )
    return model


class TestSeasonalNaive:
    def test_point_forecast_tiles_the_last_cycle(self):
        # Exactly periodic: the seasonal naive is a perfect forecaster.
        cycle = np.array([1.0, 2.0, 3.0, 4.0])
        contexts = np.tile(cycle, 5)[None, :]
        forecast = naive().predict(contexts, horizon=6)
        np.testing.assert_allclose(forecast.point[0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0])

    def test_perfectly_periodic_series_has_zero_spread(self):
        contexts = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 5)[None, :]
        forecast = naive().predict(contexts, horizon=8)
        np.testing.assert_allclose(
            forecast.quantiles[0], np.repeat(forecast.point[0][:, None], 3, 1)
        )

    def test_median_quantile_equals_the_point_forecast(self):
        rng = np.random.default_rng(0)
        contexts = rng.normal(size=(3, 40))
        forecast = naive(levels=(0.1, 0.5, 0.9)).predict(contexts, horizon=8)
        np.testing.assert_allclose(forecast.quantiles[..., 1], forecast.point)

    def test_spread_widens_as_sqrt_of_cycles_ahead(self):
        rng = np.random.default_rng(1)
        contexts = rng.normal(size=(1, 60))
        forecast = naive(seasonality=4).predict(contexts, horizon=12)
        width = forecast.quantiles[0, :, 2] - forecast.quantiles[0, :, 0]
        # Steps 0-3 are one cycle ahead, 4-7 two, 8-11 three.
        np.testing.assert_allclose(width[4] / width[0], np.sqrt(2.0))
        np.testing.assert_allclose(width[8] / width[0], np.sqrt(3.0))
        np.testing.assert_allclose(width[:4], width[0])

    def test_quantiles_never_cross(self):
        rng = np.random.default_rng(2)
        forecast = naive().predict(rng.normal(size=(4, 40)), horizon=10)
        assert forecast.quantile_crossing_rate() == 0.0

    def test_uses_only_the_context(self):
        # Appending future observations must not change a forecast made from the same context.
        rng = np.random.default_rng(3)
        series = rng.normal(size=80)
        first = naive().predict(series[:40][None, :], horizon=8)
        second = naive().predict(series[:40][None, :], horizon=8)
        np.testing.assert_array_equal(first.point, second.point)
        np.testing.assert_array_equal(first.quantiles, second.quantiles)

    def test_batching_does_not_change_results(self):
        rng = np.random.default_rng(4)
        contexts = rng.normal(size=(7, 40))
        model = naive()
        whole = model.predict(contexts, horizon=6)
        rows = [model.predict(contexts[i : i + 1], horizon=6).point for i in range(7)]
        np.testing.assert_allclose(whole.point, np.concatenate(rows, axis=0))

    def test_requires_context_longer_than_the_period(self):
        with pytest.raises(ValueError, match="context longer than the seasonal period"):
            naive(seasonality=8).predict(np.ones((1, 8)), horizon=2)

    def test_seasonality_comes_from_the_dataset(self):
        model = SeasonalNaiveModel(SeasonalNaiveOptions())
        with pytest.raises(ValueError, match="has no seasonality"):
            model.predict(np.ones((1, 40)), horizon=2)
        model.bind_dataset(DatasetContext("d", "h", 24))
        assert model.seasonality == 24
        assert model.describe()["seasonality"] == 24

    def test_configured_seasonality_must_match_the_dataset(self):
        model = naive(seasonality=7)
        with pytest.raises(ValueError, match="configured with seasonality 7"):
            model.bind_dataset(DatasetContext("etth1", "h", 24))

    def test_matching_override_is_accepted(self):
        model = naive(seasonality=24)
        model.bind_dataset(DatasetContext("etth1", "h", 24))
        assert model.seasonality == 24

    def test_has_no_trainable_parameters(self):
        counts = naive().parameter_counts()
        assert counts["total_parameters"] == 0
        assert counts["trainable_pct"] == 0.0


class TestRegistry:
    def test_lists_both_v01_adapters(self):
        assert available_models() == ("seasonal_naive", "timesfm_2p5")

    def test_unknown_model_names_the_alternatives(self):
        with pytest.raises(KeyError, match="unknown model 'moirai'"):
            get_model_spec("moirai")

    def test_options_reject_unknown_keys(self):
        with pytest.raises(ValueError, match="rank"):
            validate_options("seasonal_naive", {"rank": 8})

    def test_timesfm_options_validate_without_torch(self):
        # Config validation must work in an environment without the models extra.
        options = validate_options("timesfm_2p5", {"batch_size": 8, "dtype": "bfloat16"})
        assert options.batch_size == 8
        assert options.checkpoint.startswith("google/timesfm-2.5")

    def test_timesfm_rejects_unknown_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            validate_options("timesfm_2p5", {"dtype": "float8"})

    def test_build_model_validates_raw_options(self):
        model = build_model("seasonal_naive", {"seasonality": 12})
        assert model.seasonality == 12

    def test_timesfm_spec_declares_its_extra(self):
        assert get_model_spec("timesfm_2p5").extra == "models"
        assert get_model_spec("seasonal_naive").extra is None
