"""The evaluation driver: scoring, scaling round-trips, determinism and leakage."""

import numpy as np
import pytest

from tsfm_peft.data.scaling import fit_scaler
from tsfm_peft.data.windows import BacktestProtocol, make_split
from tsfm_peft.evaluate import EvaluationResult, evaluate_model, select_windows
from tsfm_peft.models.base import Forecast, ForecastModel
from tsfm_peft.models.naive import SeasonalNaiveModel, SeasonalNaiveOptions
from tsfm_peft.runtime import ResourceLog

LEVELS = (0.1, 0.5, 0.9)


@pytest.fixture
def protocol():
    return BacktestProtocol(horizon=24, context_length=96, n_test_windows=3, n_val_windows=1)


@pytest.fixture
def split(dataset, protocol):
    return make_split(dataset, protocol)


class LastValueModel(ForecastModel):
    """Repeats the final context value. Optionally demands external scaling."""

    name = "last_value"

    def __init__(self, requires_scaling=False) -> None:
        self._requires_scaling = requires_scaling
        self.seen_contexts = []

    @property
    def quantile_levels(self):
        return LEVELS

    @property
    def max_horizon(self):
        return None

    @property
    def requires_external_scaling(self):
        return self._requires_scaling

    def _predict_batch(self, contexts, horizon):
        self.seen_contexts.append(np.array(contexts))
        point = np.repeat(contexts[:, -1:], horizon, axis=1)
        quantiles = point[:, :, None] + np.array(LEVELS) - 0.5
        return Forecast(point=point, quantiles=quantiles, levels=LEVELS)


def naive_model():
    return SeasonalNaiveModel(SeasonalNaiveOptions())


class TestSelectWindows:
    def test_returns_the_named_set(self, split):
        assert select_windows(split, "test") is split.test_windows
        assert select_windows(split, "val") is split.val_windows

    def test_rejects_an_unknown_name(self, split):
        with pytest.raises(ValueError, match="must be 'test' or 'val'"):
            select_windows(split, "train")

    def test_rejects_an_empty_set(self, dataset, protocol):
        no_val = make_split(dataset, protocol.model_copy(update={"n_val_windows": 0}))
        with pytest.raises(ValueError, match="no val windows"):
            select_windows(no_val, "val")


class TestEvaluateModel:
    def test_scores_every_test_window(self, split):
        result = evaluate_model(naive_model(), split)
        assert isinstance(result, EvaluationResult)
        assert result.metrics.n_rows == len(split.test_windows)
        assert result.metrics.horizon == 24
        assert result.metrics.n_series == len(split.dataset)
        assert result.forecast.point.shape == (len(split.test_windows), 24)

    def test_out_of_sample_seasonal_naive_scores_near_one(self, split):
        # MASE divides by the in-sample seasonal-naive error, so the out-of-sample seasonal
        # naive should land near 1.0. A number far from it means the scale or the windowing
        # is wrong, not that the baseline got good.
        result = evaluate_model(naive_model(), split)
        assert 0.5 < result.metrics.mase < 2.0

    def test_can_score_the_validation_windows(self, split):
        result = evaluate_model(naive_model(), split, window_set="val")
        assert result.window_set == "val"
        assert result.metrics.n_rows == len(split.val_windows)

    def test_binds_the_datasets_seasonality(self, split):
        model = naive_model()
        evaluate_model(model, split)
        assert model.seasonality == split.dataset.seasonality

    def test_is_deterministic(self, split):
        first = evaluate_model(naive_model(), split)
        second = evaluate_model(naive_model(), split)
        assert first.to_dict() == second.to_dict()
        np.testing.assert_array_equal(first.forecast.point, second.forecast.point)

    def test_records_the_prediction_phase(self, split):
        log = ResourceLog()
        evaluate_model(naive_model(), split, resources=log)
        assert [p["label"] for p in log.to_dict()["phases"]] == ["predict_test"]

    def test_reports_quantile_crossings(self, split):
        result = evaluate_model(naive_model(), split)
        assert result.quantile_crossing_rate == 0.0

    def test_per_series_breakdown_covers_the_dataset(self, split):
        result = evaluate_model(naive_model(), split)
        assert set(result.metrics.per_series) == set(split.dataset.series_ids)
        for entry in result.metrics.per_series.values():
            assert entry["n_windows"] == split.protocol.n_test_windows

    def test_dict_form_is_json_serialisable(self, split):
        import json

        payload = json.loads(json.dumps(evaluate_model(naive_model(), split).to_dict()))
        assert payload["window_set"] == "test"
        assert payload["scaler_kind"] is None


class TestScaling:
    def test_internally_normalising_models_get_raw_contexts(self, split):
        model = LastValueModel(requires_scaling=False)
        result = evaluate_model(model, split)
        assert result.scaler_kind is None
        raw_last = np.array([w.context[-1] for w in split.test_windows])
        np.testing.assert_allclose(model.seen_contexts[0][:, -1], raw_last)

    def test_rejects_a_scaler_the_model_did_not_ask_for(self, split):
        with pytest.raises(ValueError, match="double-normalise"):
            evaluate_model(LastValueModel(), split, scaler_kind="standard")

    def test_identity_scaler_is_allowed_as_a_no_op(self, split):
        assert evaluate_model(LastValueModel(), split, scaler_kind="identity") is not None

    def test_external_scaling_round_trips_to_raw_units(self, split):
        model = LastValueModel(requires_scaling=True)
        result = evaluate_model(model, split, scaler_kind="standard")
        assert result.scaler_kind == "standard"
        # The adapter saw scaled values but the scored forecast is in raw units.
        raw_last = np.array([w.context[-1] for w in split.test_windows])
        np.testing.assert_allclose(result.forecast.point[:, 0], raw_last)
        assert abs(model.seen_contexts[0]).max() < abs(raw_last).max()

    def test_external_scaling_defaults_to_standard(self, split):
        assert evaluate_model(LastValueModel(True), split).scaler_kind == "standard"

    def test_scalers_are_fitted_on_the_fit_region_only(self, split):
        model = LastValueModel(requires_scaling=True)
        evaluate_model(model, split, scaler_kind="standard")
        # Reconstruct the scaler the driver must have used for the first window's series and
        # confirm it matches one fitted on values[:fit_end], not on the whole series.
        window = split.test_windows[0]
        expected = fit_scaler(split.train[window.series_id], "standard")
        scaled_last = model.seen_contexts[0][0, -1]
        np.testing.assert_allclose(scaled_last, expected.transform(window.context[-1]))

    def test_scaling_is_metric_neutral_for_a_scale_equivariant_model(self, split):
        # Guards the inverse-transform. This model repeats its last input, so scaling it and
        # unscaling the forecast must land on exactly the raw-path answer. If the driver
        # forgot to invert, MASE would be computed in scaled units and these would diverge.
        raw = evaluate_model(LastValueModel(False), split)
        scaled = evaluate_model(LastValueModel(True), split, scaler_kind="standard")
        assert raw.metrics.mase == pytest.approx(scaled.metrics.mase)


class TestLeakage:
    def test_first_window_forecast_ignores_the_evaluation_region(self, make_dataset, protocol):
        # Corrupt every observation from the first test origin onwards. The first test
        # window's context is entirely before that point, so its forecast -- and the MASE
        # denominator, which is fitted on the training region -- must not move. If either
        # changes, something is reading past the origin.
        dataset = make_dataset()
        clean = make_split(dataset, protocol)
        first_origin = min(plan.test_origins[0] for plan in clean.plans.values())

        corrupted_series = []
        for series in dataset:
            values = series.values.copy()
            values[first_origin:] += 1000.0
            corrupted_series.append((series.series_id, values))
        corrupted = make_split(
            type(dataset).from_arrays(
                dataset.name,
                corrupted_series,
                freq=dataset.freq,
                seasonality=dataset.seasonality,
            ),
            protocol,
        )

        assert clean.mase_scales == corrupted.mase_scales
        before = evaluate_model(naive_model(), clean)
        after = evaluate_model(naive_model(), corrupted)

        n_windows = protocol.n_test_windows
        first_rows = slice(0, None, n_windows)  # series-major, so row 0 of each series
        np.testing.assert_allclose(
            before.forecast.point[first_rows], after.forecast.point[first_rows]
        )
        # The metrics themselves must move: the targets changed.
        assert before.metrics.mase != pytest.approx(after.metrics.mase)
