"""Rolling-origin split arithmetic, window boundaries and leakage containment."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.windows import (
    BacktestProtocol,
    cut_window,
    make_split,
    plan_split,
    stack_windows,
)
from tsfm_peft.metrics import seasonal_naive_scale


def protocol(**overrides: object) -> BacktestProtocol:
    defaults = {"horizon": 10, "context_length": 20, "n_test_windows": 3}
    return BacktestProtocol(**{**defaults, **overrides})


class TestProtocol:
    def test_stride_defaults_to_the_horizon(self):
        assert protocol().effective_stride == 10

    def test_eval_span_without_validation(self):
        # 3 non-overlapping windows of 10 steps.
        assert protocol().eval_span == 30

    def test_eval_span_with_validation(self):
        # 3 test + 2 validation windows, all 10 steps, no overlap.
        assert protocol(n_val_windows=2).eval_span == 50

    def test_eval_span_with_overlapping_windows(self):
        # Origins 5 apart: the span is one horizon plus two strides.
        assert protocol(stride=5).eval_span == 20

    def test_min_series_length_reserves_a_full_context(self):
        assert protocol(context_length=20).min_series_length == 50

    def test_stride_longer_than_horizon_is_rejected(self):
        with pytest.raises(ValidationError, match="exceeds horizon"):
            protocol(stride=11)

    def test_non_positive_parameters_are_rejected(self):
        with pytest.raises(ValidationError):
            protocol(horizon=0)

    def test_unknown_fields_are_rejected(self):
        # extra="forbid" turns a typo in a YAML config into an error, not a silent default.
        with pytest.raises(ValidationError):
            BacktestProtocol(horizon=10, context_length=20, n_test_windows=3, hrizon=5)

    def test_is_hashable_and_frozen(self):
        assert hash(protocol()) == hash(protocol())


class TestPlanSplit:
    def test_last_origin_is_pinned_to_the_end_of_the_series(self):
        plan = plan_split("s", 100, protocol())
        assert plan.test_origins == (70, 80, 90)
        assert plan.test_origins[-1] + 10 == 100

    def test_fit_end_is_the_first_test_origin_without_validation(self):
        assert plan_split("s", 100, protocol()).fit_end == 70

    def test_validation_windows_sit_between_the_fit_region_and_the_test_region(self):
        plan = plan_split("s", 100, protocol(n_val_windows=2))
        assert plan.val_origins == (50, 60)
        assert plan.test_origins == (70, 80, 90)
        # The last validation target ends exactly where the test region starts.
        assert plan.val_origins[-1] + 10 == plan.test_origins[0]
        assert plan.fit_end == 50

    def test_overlapping_stride_packs_origins_closer(self):
        plan = plan_split("s", 100, protocol(stride=5))
        assert plan.test_origins == (80, 85, 90)
        assert plan.fit_end == 80

    def test_single_window_reduces_to_a_holdout_split(self):
        plan = plan_split("s", 100, protocol(n_test_windows=1))
        assert plan.test_origins == (90,)
        assert plan.fit_end == 90

    def test_exactly_minimum_length_is_accepted(self):
        proto = protocol()
        plan = plan_split("s", proto.min_series_length, proto)
        assert plan.fit_end == proto.context_length

    def test_one_observation_short_is_rejected(self):
        proto = protocol()
        with pytest.raises(ValueError, match=r"too short|at least"):
            plan_split("s", proto.min_series_length - 1, proto)

    def test_error_message_names_the_series_and_the_requirement(self):
        with pytest.raises(ValueError, match=r"'sensor-7'.*at least 50"):
            plan_split("sensor-7", 20, protocol())


class TestCutWindow:
    def test_context_and_target_are_adjacent_and_correct(self):
        values = np.arange(100, dtype=np.float64)
        window = cut_window("s", values, 70, horizon=10, context_length=20)
        assert window.context.tolist() == list(range(50, 70))
        assert window.target.tolist() == list(range(70, 80))

    def test_context_never_includes_the_origin(self):
        values = np.arange(100, dtype=np.float64)
        window = cut_window("s", values, 70, horizon=10, context_length=20)
        assert window.context.max() < window.origin

    def test_insufficient_history_is_rejected(self):
        values = np.arange(100, dtype=np.float64)
        with pytest.raises(ValueError, match="context_length"):
            cut_window("s", values, 10, horizon=10, context_length=20)

    def test_running_past_the_end_is_rejected(self):
        values = np.arange(100, dtype=np.float64)
        with pytest.raises(ValueError, match="past the end"):
            cut_window("s", values, 95, horizon=10, context_length=20)


class TestStackWindows:
    def test_shapes_and_ordering(self, dataset):
        split = make_split(dataset, protocol())
        contexts, targets, ids = stack_windows(split.test_windows)
        assert contexts.shape == (6, 20)
        assert targets.shape == (6, 10)
        assert ids == ("s0", "s0", "s0", "s1", "s1", "s1")

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            stack_windows([])


class TestMakeSplit:
    def test_window_counts(self, dataset):
        split = make_split(dataset, protocol(n_val_windows=2))
        assert len(split.test_windows) == 3 * len(dataset)
        assert len(split.val_windows) == 2 * len(dataset)

    def test_train_region_stops_at_fit_end(self, dataset):
        split = make_split(dataset, protocol(n_val_windows=2))
        for sid, values in split.train.items():
            assert values.size == split.plans[sid].fit_end

    def test_scales_for_aligns_with_the_window_order(self, dataset):
        split = make_split(dataset, protocol())
        scales = split.scales_for(split.test_windows)
        assert scales.shape == (len(split.test_windows),)
        expected = [split.mase_scales[w.series_id] for w in split.test_windows]
        assert scales.tolist() == expected

    def test_describe_is_json_serialisable(self, dataset):
        payload = json.loads(json.dumps(make_split(dataset, protocol()).describe()))
        assert payload["protocol"]["horizon"] == 10
        assert payload["dataset"]["license"] == "CC0-1.0"
        assert payload["n_test_windows_total"] == 6

    def test_is_deterministic(self, dataset):
        a = make_split(dataset, protocol())
        b = make_split(dataset, protocol())
        assert a.mase_scales == b.mase_scales
        assert [w.origin for w in a.test_windows] == [w.origin for w in b.test_windows]

    def test_rejects_a_dataset_whose_shortest_series_is_too_short(self):
        short = TimeSeriesDataset.from_arrays(
            "short",
            [("ok", np.arange(400.0)), ("tiny", np.arange(30.0))],
            freq="h",
            seasonality=24,
        )
        with pytest.raises(ValueError, match="'tiny'"):
            make_split(short, protocol())

    def test_rejects_a_fit_region_that_cannot_define_the_mase_scale(self):
        # Constant fit region, varying tail: the seasonal-naive MAE would be zero.
        values = np.concatenate([np.full(70, 5.0), np.arange(30.0)])
        constant = TimeSeriesDataset.from_arrays(
            "constant", [("flat", values)], freq="h", seasonality=1
        )
        with pytest.raises(ValueError, match="MASE scale"):
            make_split(constant, protocol())


class TestNoLeakage:
    """The fit region must not contain, or be influenced by, any evaluated observation."""

    def test_training_data_ends_before_the_first_evaluated_observation(self, dataset):
        split = make_split(dataset, protocol(n_val_windows=2))
        for series in dataset:
            plan = split.plans[series.series_id]
            first_evaluated = min((*plan.val_origins, *plan.test_origins))
            assert split.train[series.series_id].size == first_evaluated
            assert plan.fit_end == first_evaluated

    def test_no_window_context_reaches_its_own_origin_or_beyond(self, dataset):
        split = make_split(dataset, protocol(n_val_windows=2))
        for window in (*split.val_windows, *split.test_windows):
            values = dataset[window.series_id].values
            expected = values[window.origin - 20 : window.origin]
            assert np.array_equal(window.context, expected)
            assert np.array_equal(window.target, values[window.origin : window.origin + 10])

    def test_a_sentinel_written_past_fit_end_never_reaches_the_fit_region(self, make_dataset):
        # Every observation from fit_end onwards is replaced with an unmistakable value.
        # If any of it were to reach the training region or the MASE scale, these
        # assertions would catch it regardless of how the leak happened.
        base = make_dataset(n_series=1, length=200, seasonality=4)
        proto = protocol(context_length=20, n_test_windows=3, n_val_windows=2)
        fit_end = plan_split("s0", 200, proto).fit_end
        clean = base["s0"].values.copy()
        poisoned = clean.copy()
        poisoned[fit_end:] = 1e9
        contaminated = TimeSeriesDataset.from_arrays(
            "poisoned", [("s0", poisoned)], freq="h", seasonality=4
        )

        clean_split = make_split(base, proto)
        poisoned_split = make_split(contaminated, proto)

        assert np.array_equal(clean_split.train["s0"], poisoned_split.train["s0"])
        assert clean_split.mase_scales["s0"] == poisoned_split.mase_scales["s0"]
        assert poisoned_split.train["s0"].max() < 1e9

    def test_mase_scale_comes_from_the_fit_region_not_the_full_series(self):
        # A tail with a much larger seasonal swing than the fit region: if the scale were
        # computed over the whole series it would be visibly larger, which would deflate
        # MASE for every arm.
        values = np.concatenate([np.tile([0.0, 1.0], 100), np.tile([0.0, 100.0], 50)])
        data = TimeSeriesDataset.from_arrays(
            "tail-heavy", [("s0", values)], freq="h", seasonality=1
        )
        proto = BacktestProtocol(horizon=10, context_length=20, n_test_windows=3)
        split = make_split(data, proto)
        fit_end = split.plans["s0"].fit_end

        assert split.mase_scales["s0"] == pytest.approx(seasonal_naive_scale(values[:fit_end], 1))
        assert split.mase_scales["s0"] < seasonal_naive_scale(values, 1)

    def test_evaluation_windows_tile_the_tail_without_overlap(self, dataset):
        # With stride == horizon every evaluated observation is predicted exactly once, so
        # aggregate metrics cannot double-count part of the series.
        split = make_split(dataset, protocol())
        for series in dataset:
            windows = [w for w in split.test_windows if w.series_id == series.series_id]
            stitched = np.concatenate([w.target for w in windows])
            first_origin = split.plans[series.series_id].test_origins[0]
            assert np.array_equal(stitched, series.values[first_origin:])
