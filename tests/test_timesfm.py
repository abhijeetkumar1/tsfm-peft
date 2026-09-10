"""The TimesFM 2.5 adapter, against the real checkpoint.

Marked ``slow`` and ``network``: these download roughly a gigabyte of weights and are
excluded from the CPU CI job. Run them before publishing any TimesFM number:

    uv run pytest tests/test_timesfm.py -m "slow and network"

The options-level tests that need neither torch nor weights live in ``test_models.py`` and
do run in CI, so a config typo is still caught there.
"""

import numpy as np
import pytest

from tsfm_peft.data.registry import load_dataset
from tsfm_peft.data.windows import BacktestProtocol, make_split
from tsfm_peft.evaluate import evaluate_model
from tsfm_peft.models.registry import build_model
from tsfm_peft.runtime import set_seed

pytestmark = [pytest.mark.slow, pytest.mark.network]

PATCH_LENGTH = 32
MODEL_HORIZON = 128


@pytest.fixture(scope="module")
def timesfm():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return build_model("timesfm_2p5", {"device": "cpu", "batch_size": 8})


@pytest.fixture(scope="module")
def lora():
    pytest.importorskip("peft")
    set_seed(0)
    return build_model(
        "timesfm_2p5", {"device": "cpu", "batch_size": 2, "peft": {"rank": 16, "alpha": 32}}
    )


def sinusoid(length=256, period=24, level=10.0, amplitude=3.0):
    t = np.arange(length, dtype=np.float64)
    return level + amplitude * np.sin(2.0 * np.pi * t / period)


class TestCheckpointContract:
    def test_exposes_the_decile_quantile_heads(self, timesfm):
        assert timesfm.quantile_levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def test_single_decode_horizon_is_advertised(self, timesfm):
        assert timesfm.max_horizon == MODEL_HORIZON

    def test_reports_the_200m_parameter_count(self, timesfm):
        counts = timesfm.parameter_counts()
        assert 2e8 < counts["total_parameters"] < 3e8
        # Nothing is frozen yet; the LoRA arms are what change this.
        assert counts["trainable_pct"] == pytest.approx(100.0)

    def test_point_forecast_is_the_median_head(self, timesfm):
        # Guards the index mapping: full_predictions slot 0 is the point head's own output
        # and slots 1..9 are the quantiles, so the median must land at our index 4. Getting
        # this off by one would silently score a decile as if it were the point forecast.
        forecast = timesfm.predict(sinusoid()[None, :], 32)
        np.testing.assert_allclose(forecast.quantiles[..., 4], forecast.point)

    def test_describe_records_what_a_rerun_would_need(self, timesfm):
        described = timesfm.describe()
        assert described["checkpoint"].startswith("google/timesfm-2.5")
        assert described["device"] == "cpu"
        assert described["dtype"] == "float32"
        assert described["median_head_index"] == 5
        assert described["patch_length"] == PATCH_LENGTH


class TestForecasting:
    def test_continues_a_clean_sinusoid(self, timesfm):
        # A noiseless period-24 sinusoid has an exact continuation; the model should be very
        # close to it. Loose tolerance -- this checks the context is fed in the right order
        # and orientation, not the model's accuracy.
        period, length, horizon = 24, 256, 48
        context = sinusoid(length=length + horizon, period=period)
        forecast = timesfm.predict(context[:length][None, :], horizon)
        np.testing.assert_allclose(forecast.point[0], context[length:], atol=0.5)

    def test_is_deterministic(self, timesfm):
        context = sinusoid()[None, :]
        first = timesfm.predict(context, 32)
        second = timesfm.predict(context, 32)
        np.testing.assert_array_equal(first.point, second.point)
        np.testing.assert_array_equal(first.quantiles, second.quantiles)

    def test_rejects_a_horizon_beyond_one_decode(self, timesfm):
        with pytest.raises(ValueError, match="exceeds the adapter's single-call maximum"):
            timesfm.predict(sinusoid()[None, :], MODEL_HORIZON + 1)

    def test_rejects_a_context_the_patches_do_not_tile(self, timesfm):
        # Upstream would left-pad with zeros, shifting the normalisation statistics.
        with pytest.raises(ValueError, match="not a multiple of the checkpoint's patch_length"):
            timesfm.predict(sinusoid(length=100)[None, :], 32)

    def test_batching_does_not_change_a_forecast(self, timesfm):
        contexts = np.stack([sinusoid(level=10.0), sinusoid(level=50.0), sinusoid(level=90.0)])
        batched = timesfm.predict(contexts, 32)
        for i in range(contexts.shape[0]):
            alone = timesfm.predict(contexts[i : i + 1], 32)
            np.testing.assert_allclose(batched.point[i], alone.point[0], rtol=1e-5, atol=1e-5)


class TestNegativeClamping:
    """Upstream's clamp is batch-global; ``per_series`` is why results stay batch-stable."""

    @staticmethod
    def contexts() -> tuple:
        rng = np.random.default_rng(0)
        spans_negative = sinusoid(level=0.0, amplitude=5.0)
        # Non-negative but hugging zero, so the lower quantiles want to cross it.
        hugs_zero = np.abs(0.4 * np.sin(np.arange(256) * 2 * np.pi / 24) + rng.normal(0, 0.3, 256))
        assert hugs_zero.min() >= 0.0
        return spans_negative, hugs_zero

    def forecast_pair(self, mode):
        options = {"device": "cpu", "batch_size": 8, "clamp_negative": mode}
        model = build_model("timesfm_2p5", options)
        spans_negative, hugs_zero = self.contexts()
        batched = model.predict(np.stack([spans_negative, hugs_zero]), 32)
        alone = model.predict(hugs_zero[None, :], 32)
        return batched.quantiles[1], alone.quantiles[0]

    def test_per_series_is_independent_of_batch_composition(self):
        batched, alone = self.forecast_pair("per_series")
        np.testing.assert_array_equal(batched, alone)
        assert batched.min() >= 0.0

    def test_upstream_batch_rule_is_not(self):
        # The behaviour this adapter defaults away from: sharing a batch with a series that
        # goes negative suppresses the clamp for a series that never does.
        batched, alone = self.forecast_pair("model_batch")
        assert batched.min() < 0.0
        assert alone.min() >= 0.0

    def test_never_leaves_forecasts_unclamped(self):
        batched, alone = self.forecast_pair("never")
        np.testing.assert_array_equal(batched, alone)
        assert alone.min() < 0.0


class TestEndToEnd:
    def test_scores_a_backtest_split(self, timesfm):
        split = make_split(
            load_dataset("synthetic"),
            BacktestProtocol(horizon=24, context_length=96, n_test_windows=2),
        )
        result = evaluate_model(timesfm, split)
        assert result.metrics.n_rows == len(split.test_windows)
        assert result.metrics.mase > 0
        assert result.scaler_kind is None
        assert 0.0 <= result.quantile_crossing_rate <= 1.0


class TestPeftOnTheRealCheckpoint:
    """LoRA and DoRA against the 200M weights, where the target names have to be right.

    The tiny fixture shares the architecture class, so it already proves the names match;
    what these add is the real shapes -- and the trainable-parameter count that the results
    table reports, which is arithmetic no fixture can stand in for.
    """

    def test_adapts_every_target_in_every_layer(self, lora):
        # 20 decoder layers x 6 targets (q, k, v, o, fc1, fc2) x (lora_A, lora_B).
        adapters = [name for name, _ in lora.module.named_parameters() if "lora_" in name]
        assert len(adapters) == 240

    def test_trains_two_percent_of_the_weights(self, lora):
        # Each adapted projection is 1280x1280, so rank 16 adds 1280*16 + 16*1280 = 40960
        # parameters, over 120 modules: 4,915,200 against the checkpoint's 236M.
        counts = lora.parameter_counts()
        assert counts["trainable_parameters"] == 4_915_200
        assert counts["trainable_pct"] == pytest.approx(2.08, abs=0.01)

    def test_dora_costs_a_magnitude_vector_more(self):
        pytest.importorskip("peft")
        set_seed(0)
        dora = build_model(
            "timesfm_2p5",
            {"device": "cpu", "batch_size": 2, "peft": {"method": "dora", "rank": 16}},
        )
        # One magnitude scalar per output channel of each adapted projection.
        extra = dora.parameter_counts()["trainable_parameters"] - 4_915_200
        assert extra == 120 * 1280

    def test_one_training_step_runs(self, lora):
        rng = np.random.default_rng(0)
        contexts = 10.0 + rng.normal(size=(2, 512)).cumsum(axis=1)
        targets = contexts[:, -1:] + rng.normal(size=(2, 96))
        loss = lora.training_loss(contexts, targets)
        loss.backward()
        assert np.isfinite(float(loss.detach()))
        assert all(p.grad is not None for p in lora.trainable_parameters())
