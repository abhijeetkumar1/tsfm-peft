"""The fine-tuning loop: schedule, batching, selection, and what it is allowed to read.

The config, schedule and batching tests are torch-free and run in the CPU CI job. The loop
tests need the ``models`` extra and drive the tiny fixture model on CPU.
"""

import numpy as np
import pytest

from tsfm_peft.config import ExperimentConfig
from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.registry import load_dataset
from tsfm_peft.data.windows import BacktestProtocol, make_split
from tsfm_peft.evaluate import evaluate_model
from tsfm_peft.models.registry import build_model
from tsfm_peft.runtime import set_seed
from tsfm_peft.training import (
    TrainingConfig,
    iter_batches,
    learning_rate_multiplier,
    train_model,
)

PROTOCOL = BacktestProtocol(horizon=24, context_length=96, n_test_windows=2, n_val_windows=1)
TINY_PEFT = {"device": "cpu", "batch_size": 8, "peft": {"rank": 4, "alpha": 8}}


def training_config(**overrides: object) -> TrainingConfig:
    defaults = {
        "max_steps": 6,
        "batch_size": 8,
        "learning_rate": 0.01,
        "eval_every": 3,
        "log_every": 3,
        "save_adapter": False,
    }
    return TrainingConfig(**{**defaults, **overrides})


def experiment_dict(**overrides: object):
    payload = {
        "name": "unit-test",
        "data": {
            "dataset": "synthetic",
            "protocol": PROTOCOL.model_dump(),
            "scaler": "identity",
        },
        "model": {"name": "timesfm_2p5_tiny", "options": {"peft": {"rank": 4}}},
        "training": {"max_steps": 4, "eval_every": 2},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def split():
    return make_split(load_dataset("synthetic"), PROTOCOL)


@pytest.fixture
def model():
    pytest.importorskip("peft")
    set_seed(0)
    return build_model("timesfm_2p5_tiny", TINY_PEFT)


class TestTrainingConfig:
    def test_rejects_a_warmup_that_never_ends(self):
        with pytest.raises(ValueError, match="never reaches its peak"):
            TrainingConfig(max_steps=10, warmup_steps=10)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig.model_validate({"lr": 0.001})

    def test_rejects_a_non_positive_step_count(self):
        with pytest.raises(ValueError, match="greater than 0"):
            TrainingConfig(max_steps=0)


class TestLearningRateSchedule:
    def test_warmup_ramps_from_a_fraction_to_one(self):
        multipliers = [
            learning_rate_multiplier(s, max_steps=100, warmup_steps=10) for s in range(10)
        ]
        assert multipliers[0] == pytest.approx(0.1)
        assert multipliers[-1] == pytest.approx(1.0)
        assert multipliers == sorted(multipliers)

    def test_cosine_decays_to_zero_at_the_last_step(self):
        assert learning_rate_multiplier(0, max_steps=100) == pytest.approx(1.0)
        assert learning_rate_multiplier(50, max_steps=100) == pytest.approx(0.5)
        assert learning_rate_multiplier(100, max_steps=100) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_does_not_go_negative_past_the_end(self):
        assert learning_rate_multiplier(150, max_steps=100) == pytest.approx(0.0, abs=1e-12)

    def test_constant_holds_after_warmup(self):
        for step in (10, 50, 99):
            assert (
                learning_rate_multiplier(step, max_steps=100, warmup_steps=10, schedule="constant")
                == 1.0
            )


class TestIterBatches:
    def test_yields_the_requested_number_of_batches(self):
        rng = np.random.default_rng(0)
        assert len(list(iter_batches(10, 3, 7, rng=rng))) == 7

    def test_one_pass_covers_every_window_exactly_once(self):
        # Sampling with replacement would weight some windows more than others within an
        # epoch; a permutation cannot.
        rng = np.random.default_rng(0)
        seen = np.concatenate(list(iter_batches(10, 5, 2, rng=rng)))
        assert sorted(seen) == list(range(10))

    def test_the_same_seed_gives_the_same_order(self):
        first = list(iter_batches(20, 4, 8, rng=np.random.default_rng(7)))
        second = list(iter_batches(20, 4, 8, rng=np.random.default_rng(7)))
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_a_different_seed_gives_a_different_order(self):
        first = np.concatenate(list(iter_batches(20, 4, 5, rng=np.random.default_rng(1))))
        second = np.concatenate(list(iter_batches(20, 4, 5, rng=np.random.default_rng(2))))
        assert not np.array_equal(first, second)

    def test_reshuffles_on_the_next_pass(self):
        rng = np.random.default_rng(0)
        batches = list(iter_batches(6, 6, 2, rng=rng))
        assert sorted(batches[0]) == sorted(batches[1]) == list(range(6))
        assert not np.array_equal(batches[0], batches[1])

    def test_rejects_an_empty_window_set(self):
        with pytest.raises(ValueError, match="empty set"):
            list(iter_batches(0, 4, 1, rng=np.random.default_rng(0)))


class TestConfigValidation:
    def test_accepts_a_finetunable_model_with_adapters(self):
        assert ExperimentConfig.model_validate(experiment_dict()).training.max_steps == 4

    def test_rejects_training_a_model_that_cannot_be_finetuned(self):
        payload = experiment_dict(model={"name": "seasonal_naive"})
        with pytest.raises(ValueError, match="cannot be fine-tuned"):
            ExperimentConfig.model_validate(payload)

    def test_rejects_training_with_no_peft_block(self):
        # The expensive silent failure: a full training run that updates nothing.
        payload = experiment_dict(model={"name": "timesfm_2p5_tiny", "options": {}})
        with pytest.raises(ValueError, match="every weight would stay frozen"):
            ExperimentConfig.model_validate(payload)

    def test_rejects_validation_without_validation_windows(self):
        protocol = PROTOCOL.model_dump() | {"n_val_windows": 0}
        payload = experiment_dict(
            data={"dataset": "synthetic", "protocol": protocol, "scaler": "identity"}
        )
        with pytest.raises(ValueError, match="reserves no validation windows"):
            ExperimentConfig.model_validate(payload)

    def test_allows_a_fixed_step_count_without_validation_windows(self):
        protocol = PROTOCOL.model_dump() | {"n_val_windows": 0}
        payload = experiment_dict(
            data={"dataset": "synthetic", "protocol": protocol, "scaler": "identity"},
            training={"max_steps": 4, "eval_every": 0},
        )
        assert ExperimentConfig.model_validate(payload).training.eval_every == 0


class TestTrainModel:
    def test_reduces_the_training_loss(self, model, split):
        record = train_model(model, split, training_config(max_steps=12, log_every=6), seed=0)
        assert record.loss_curve[-1]["loss"] < record.loss_curve[0]["loss"]

    def test_reports_windows_steps_and_epochs(self, model, split):
        record = train_model(model, split, training_config(max_steps=6, batch_size=8), seed=0)
        assert record.n_train_windows == 54
        assert record.steps == 6
        # 6 steps of 8 windows over 54 available windows.
        assert record.epochs == pytest.approx(48 / 54)

    def test_restores_the_selected_checkpoint(self, model, split):
        # The claim the artifact makes is that the reported numbers come from the selected
        # step. Re-scoring validation after training has to reproduce the selected value; if
        # the restore were a no-op this would return the last step's score instead.
        record = train_model(model, split, training_config(max_steps=6, eval_every=3), seed=0)
        assert record.selection == "best_val"
        assert record.best_step in [point["step"] for point in record.val_curve]
        rescored = evaluate_model(model, split, window_set="val").metrics.mase
        assert rescored == pytest.approx(record.best_metric, rel=1e-9)

    def test_keeps_the_last_step_when_validation_is_disabled(self, model, split):
        record = train_model(model, split, training_config(eval_every=0), seed=0)
        assert record.selection == "last_step"
        assert record.val_curve == []
        assert record.best_step is None

    def test_records_the_selection_metric(self, model, split):
        record = train_model(model, split, training_config(selection_metric="wql"), seed=0)
        assert record.to_dict()["selection_metric"] == "wql"

    def test_leaves_the_model_in_eval_mode(self, model, split):
        # A test score taken with dropout still active is not the score the weights produce
        # at inference; the loop owes the caller an evaluation-ready model.
        train_model(model, split, training_config(), seed=0)
        assert not model.module.training

    def test_is_deterministic_under_the_same_seed(self, split):
        def run():
            set_seed(0)
            model = build_model("timesfm_2p5_tiny", TINY_PEFT)
            return train_model(model, split, training_config(), seed=0).loss_curve

        assert run() == run()

    def test_the_batch_seed_changes_the_result(self, split):
        def run(seed_offset):
            set_seed(0)
            model = build_model("timesfm_2p5_tiny", TINY_PEFT)
            config = training_config(seed_offset=seed_offset)
            return train_model(model, split, config, seed=0).loss_curve[-1]["loss"]

        assert run(0) != run(1)

    def test_writes_the_adapter_when_asked(self, model, split, tmp_path):
        record = train_model(
            model,
            split,
            training_config(save_adapter=True),
            seed=0,
            adapter_dir=tmp_path / "adapter",
        )
        written = {p.name for p in (tmp_path / "adapter").iterdir()}
        assert "adapter_config.json" in written
        assert record.adapter_path == str(tmp_path / "adapter")

    def test_rejects_a_model_that_cannot_be_finetuned(self, split):
        with pytest.raises(TypeError, match="does not support fine-tuning"):
            train_model(build_model("seasonal_naive"), split, training_config(), seed=0)

    def test_rejects_a_model_without_adapters(self, split):
        # Every base weight is trainable on an unwrapped model, so "has trainable
        # parameters" is not the check that catches this; having something checkpointable is.
        pytest.importorskip("torch")
        zero_shot = build_model("timesfm_2p5_tiny", {"device": "cpu", "batch_size": 8})
        with pytest.raises(ValueError, match="no adapters attached"):
            train_model(zero_shot, split, training_config(), seed=0)

    def test_rejects_validation_without_validation_windows(self, model):
        protocol = PROTOCOL.model_copy(update={"n_val_windows": 0})
        no_val = make_split(load_dataset("synthetic"), protocol)
        with pytest.raises(ValueError, match="no validation windows"):
            train_model(model, no_val, training_config(eval_every=2), seed=0)


class TestNoLeakage:
    def test_training_batches_come_only_from_the_fit_region(self, model):
        # A series whose fit region is bounded well below the values that follow it. Every
        # observation the loop is handed must come from the low region: one index past
        # fit_end and the assertion below fails by three orders of magnitude.
        fit_values = np.linspace(0.0, 1.0, 228)
        evaluation_values = np.full(72, 1000.0)
        data = TimeSeriesDataset.from_arrays(
            "bounded",
            [("s0", np.concatenate([fit_values, evaluation_values]))],
            freq="h",
            seasonality=24,
        )
        split = make_split(data, PROTOCOL)
        assert split.plans["s0"].fit_end == 228

        seen = []
        original = model.training_loss

        def spy(contexts, targets):
            seen.append((np.max(contexts), np.max(targets)))
            return original(contexts, targets)

        model.training_loss = spy
        train_model(model, split, training_config(eval_every=0), seed=0)

        assert seen
        assert max(max(pair) for pair in seen) <= 1.0
