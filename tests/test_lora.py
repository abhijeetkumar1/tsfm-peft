"""LoRA/DoRA options, adapter attachment and the aligned training loss.

The options tests are pure pydantic and run in the torch-free CI job, so a bad ``peft``
block in a config is caught there. The rest need the ``models`` extra and are skipped
without it; the models CI job installs it and runs them on CPU.
"""

import numpy as np
import pytest

from tsfm_peft.models.lora import DEFAULT_TARGET_MODULES, PeftOptions
from tsfm_peft.models.registry import build_model, validate_options
from tsfm_peft.runtime import set_seed

TINY = {"device": "cpu", "batch_size": 4}


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    set_seed(0)
    return build_model("timesfm_2p5_tiny", TINY)


def tiny_with_peft(**peft: object):
    pytest.importorskip("peft")
    set_seed(0)
    return build_model("timesfm_2p5_tiny", {**TINY, "peft": {"rank": 4, "alpha": 8, **peft}})


def scalar(loss) -> float:
    """Read a loss tensor as a float without tripping torch's autograd warning."""
    return float(loss.detach())


def contexts(n=4, length=96, seed=0):
    steps = np.random.default_rng(seed).normal(size=(n, length))
    return 10.0 + steps.cumsum(axis=1)


class TestPeftOptions:
    def test_defaults_target_attention_and_mlp(self):
        assert PeftOptions().target_modules == DEFAULT_TARGET_MODULES
        assert PeftOptions().method == "lora"
        assert not PeftOptions().use_dora

    def test_dora_is_the_same_options_block(self):
        assert PeftOptions(method="dora").use_dora

    def test_rejects_an_empty_target_set(self):
        # A silent no-op fine-tune is the most expensive possible failure: hours of GPU time
        # producing a model identical to the base one.
        with pytest.raises(ValueError, match="would train nothing"):
            PeftOptions(target_modules=())

    def test_rejects_duplicate_targets(self):
        with pytest.raises(ValueError, match="duplicates"):
            PeftOptions(target_modules=("q_proj", "q_proj"))

    def test_rejects_a_non_positive_rank(self):
        with pytest.raises(ValueError, match="greater than 0"):
            PeftOptions(rank=0)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="lora_rank"):
            PeftOptions.model_validate({"lora_rank": 8})

    def test_describe_records_the_effective_scaling(self):
        described = PeftOptions(rank=8, alpha=16).describe()
        assert described["scaling"] == 2.0
        assert described["target_modules"] == list(DEFAULT_TARGET_MODULES)

    def test_validates_through_the_model_options_without_torch(self):
        options = validate_options("timesfm_2p5", {"peft": {"method": "dora", "rank": 4}})
        assert options.peft.use_dora
        assert options.peft.rank == 4

    def test_a_typo_in_the_peft_block_fails_at_config_time(self):
        with pytest.raises(ValueError, match="target_moduls"):
            validate_options("timesfm_2p5", {"peft": {"target_moduls": ["q_proj"]}})


class TestApplyPeft:
    def test_zero_shot_has_no_adapters(self, tiny):
        counts = tiny.parameter_counts()
        assert counts["trainable_pct"] == 100.0
        assert tiny.peft_options is None

    def test_lora_trains_a_small_fraction_of_the_weights(self):
        counts = tiny_with_peft().parameter_counts()
        assert 0 < counts["trainable_pct"] < 10.0
        assert counts["frozen_parameters"] > counts["trainable_parameters"]

    def test_dora_trains_more_than_lora_at_the_same_rank(self):
        # DoRA adds a magnitude vector per adapted output channel, so it is strictly the
        # larger arm; if these ever match, use_dora is not reaching peft.
        lora = tiny_with_peft(method="lora").parameter_counts()
        dora = tiny_with_peft(method="dora").parameter_counts()
        assert dora["trainable_parameters"] > lora["trainable_parameters"]

    def test_attaching_adapters_does_not_change_the_forecast(self, tiny):
        # lora_B is initialised to zero, so a freshly wrapped model must be numerically
        # identical to the base one. If this drifts, every fine-tuned arm starts from a
        # different model than the zero-shot row it is compared against.
        batch = contexts()
        wrapped = tiny_with_peft()
        assert np.allclose(tiny.predict(batch, 24).point, wrapped.predict(batch, 24).point)

    def test_an_unmatched_target_names_the_model(self):
        with pytest.raises(ValueError, match="could not attach"):
            tiny_with_peft(target_modules=("not_a_module",))

    def test_describe_carries_the_peft_block(self):
        described = tiny_with_peft(method="dora", rank=4).describe()
        assert described["peft"]["method"] == "dora"
        assert described["peft"]["rank"] == 4
        assert described["trainable_parameters"] < described["total_parameters"]


class TestAdapterState:
    def test_round_trips_the_trained_weights(self):
        model = tiny_with_peft()
        state = model.checkpoint_state()
        assert state
        model.restore_checkpoint(state)

    def test_a_restored_checkpoint_reproduces_its_forecast(self, torch):
        model = tiny_with_peft()
        saved = model.checkpoint_state()
        batch = contexts()
        before = model.predict(batch, 24).point

        # Move the adapters somewhere else, then restore.
        with torch.no_grad():
            for name, parameter in model.module.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.5)
        assert not np.allclose(model.predict(batch, 24).point, before)

        model.restore_checkpoint(saved)
        assert np.allclose(model.predict(batch, 24).point, before)

    def test_round_trips_dora_magnitude_vectors(self):
        # DoRA's state carries a magnitude vector per adapted output channel on top of the
        # low-rank pair. Restoring only the pair would silently keep the wrong magnitudes.
        model = tiny_with_peft(method="dora")
        state = model.checkpoint_state()
        assert any("magnitude" in key for key in state)
        model.restore_checkpoint(state)

    def test_rejects_an_empty_state(self):
        with pytest.raises(ValueError, match="nothing to restore"):
            tiny_with_peft().restore_checkpoint({})

    def test_rejects_a_state_from_another_configuration(self, torch):
        model = tiny_with_peft()
        state = model.checkpoint_state()
        state["base_model.model.model.layers.0.self_attn.not_a_projection.lora_A.weight"] = (
            torch.zeros(4, 32)
        )
        with pytest.raises(ValueError, match="no parameter for"):
            model.restore_checkpoint(state)


class TestAlignedLoss:
    def loss(self, torch, predictions, target, levels, **kwargs: float):
        from tsfm_peft.models.timesfm import aligned_forecast_loss

        return float(
            aligned_forecast_loss(
                torch.tensor(predictions, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                levels,
                **kwargs,
            )
        )

    def test_matches_a_hand_computed_value(self, torch):
        # Columns are [point, q0.1, q0.5, q0.9]; target 0.5.
        #   q=0.1: e=1.5  -> max(-0.9 * 1.5, 0.1 * 1.5) = 0.15
        #   q=0.5: e=0.5  -> max(-0.5 * 0.5, 0.5 * 0.5) = 0.25
        #   q=0.9: e=-0.5 -> max(-0.1 * -0.5, 0.9 * -0.5) = 0.05
        # pinball = 0.45 / 3 = 0.15; point MSE = (0 - 0.5)^2 = 0.25.
        value = self.loss(torch, [[[0.0, -1.0, 0.0, 1.0]]], [[0.5]], (0.1, 0.5, 0.9))
        assert value == pytest.approx(0.40)

    def test_the_point_weight_scales_only_the_point_term(self, torch):
        args = ([[[0.0, -1.0, 0.0, 1.0]]], [[0.5]], (0.1, 0.5, 0.9))
        assert self.loss(torch, *args, point_weight=0.0) == pytest.approx(0.15)
        assert self.loss(torch, *args, point_weight=2.0) == pytest.approx(0.65)

    def test_each_level_is_scored_against_its_own_column(self, torch):
        # The alignment test. A low quantile is penalised ten times harder for overshooting
        # than for undershooting; a high quantile the other way round. Reading the columns
        # off by one would swap these two numbers.
        low_over = self.loss(torch, [[[0.0, 1.0]]], [[0.0]], (0.1,), point_weight=0.0)
        low_under = self.loss(torch, [[[0.0, -1.0]]], [[0.0]], (0.1,), point_weight=0.0)
        assert low_over == pytest.approx(0.9)
        assert low_under == pytest.approx(0.1)

        high_over = self.loss(torch, [[[0.0, 1.0]]], [[0.0]], (0.9,), point_weight=0.0)
        assert high_over == pytest.approx(0.1)

    def test_the_median_head_is_included(self, torch):
        # Upstream's loss drops the decode_index column, which is exactly the head this repo
        # reports as the point forecast. Moving it here has to change the loss.
        levels = (0.1, 0.5, 0.9)
        base = self.loss(torch, [[[0.0, 0.0, 0.0, 0.0]]], [[1.0]], levels, point_weight=0.0)
        moved = self.loss(torch, [[[0.0, 0.0, 1.0, 0.0]]], [[1.0]], levels, point_weight=0.0)
        assert moved < base

    def test_rejects_a_prediction_tensor_missing_heads(self, torch):
        with pytest.raises(ValueError, match="output columns"):
            self.loss(torch, [[[0.0, 1.0]]], [[0.0]], (0.1, 0.5, 0.9))


class TestTrainingLoss:
    def test_is_differentiable_through_the_adapters(self):
        model = tiny_with_peft()
        loss = model.training_loss(contexts(), np.zeros((4, 24)))
        assert loss.requires_grad
        loss.backward()
        # lora_A starts with a zero gradient because lora_B is zero, so only half the
        # adapter tensors have one on the very first step. Every one of them is reachable.
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.module.parameters())

    def test_a_better_forecast_scores_lower(self, tiny):
        # Sanity on direction: the model's own forecast must beat an arbitrary constant.
        batch = contexts()
        own = tiny.predict(batch, 24).point
        assert scalar(tiny.training_loss(batch, own)) < scalar(
            tiny.training_loss(batch, np.full((4, 24), 1000.0))
        )

    def test_rejects_targets_that_do_not_match_the_contexts(self, tiny):
        with pytest.raises(ValueError, match="same windows"):
            tiny.training_loss(contexts(n=4), np.zeros((3, 24)))

    def test_rejects_a_horizon_past_one_decode_step(self, tiny):
        with pytest.raises(ValueError, match="exceeds the adapter"):
            tiny.training_loss(contexts(), np.zeros((4, 64)))

    def test_rejects_a_context_the_patches_cannot_tile(self, tiny):
        with pytest.raises(ValueError, match="patch_length"):
            tiny.training_loss(contexts(length=95), np.zeros((4, 24)))

    def test_is_invariant_to_the_scale_of_a_series(self, tiny):
        # The loss normalises by each context's own statistics, so multiplying a window and
        # its target by 100 must not multiply its contribution to the gradient. Without this
        # ETTh1's large-amplitude channels would own the update.
        batch = contexts()
        target = tiny.predict(batch, 24).point + 1.0
        small = scalar(tiny.training_loss(batch, target))
        large = scalar(tiny.training_loss(batch * 100.0, target * 100.0))
        assert large == pytest.approx(small, rel=1e-3)
