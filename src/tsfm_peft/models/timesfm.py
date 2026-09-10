"""TimesFM 2.5 adapter, via the transformers port of the official checkpoint.

Uses ``google/timesfm-2.5-200m-transformers`` (``TimesFm2_5ModelForPrediction``) rather than
the standalone ``timesfm`` package, because ``peft`` attaches to ``nn.Module`` target names
and the transformers port is what makes LoRA and DoRA a config change rather than a fork of
the model code. The two are weight-identical; the port's card documents the conversion.

Three properties of the upstream forward pass shape this adapter:

* **The horizon is fixed at ``config.horizon_length`` (128) per call.** Longer horizons need
  autoregressive rollout, which v0.1 does not implement -- :attr:`max_horizon` advertises
  the limit so a config asking for more fails immediately instead of quietly truncating.
* **Contexts are normalised inside the model** (RevIN over the context window, plus a
  per-patch normalisation), so this adapter forwards raw values and returns raw values, and
  :attr:`requires_external_scaling` stays ``False``.
* **Upstream's non-negativity clamp is batch-global.** ``truncate_negative`` compares the
  minimum over *every context in the batch* against zero and then clamps the whole batch, so
  putting a series with negative values in the same batch as a count series changes the
  count series' forecast. That makes results depend on batch composition, which breaks the
  determinism guarantee this repo makes. :class:`TimesFmOptions.clamp_negative` defaults to
  ``"per_series"``, which reproduces the intended semantics row by row instead.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsfm_peft.models.base import Array, Forecast, ForecastModel

#: The checkpoint v0.1 benchmarks. Pinned by name; the revision is pinned in the config.
DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-transformers"

#: dtype names accepted in configs, mapped to torch dtypes at build time.
DTYPES = ("float32", "bfloat16", "float16")

_MISSING_TORCH = (
    "TimesFM needs the optional model dependencies. Install them with "
    "`uv sync --extra models` (or `pip install 'tsfm-peft[models]'`)."
)


class TimesFmOptions(BaseModel):
    """Configuration for :class:`TimesFmModel`.

    Attributes:
        checkpoint: HuggingFace repo id or local path.
        revision: Checkpoint revision. ``None`` resolves to whatever ``main`` points at
            today, which makes a run irreproducible six months from now; pin a commit sha
            for any result that goes in the README.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``.
        dtype: Compute dtype. ``float32`` is the default because reduced precision changes
            the third decimal place of the metrics, and this repo compares arms at that
            resolution.
        batch_size: Contexts per forward pass.
        clamp_negative: How to apply the non-negativity clamp. ``"per_series"`` clamps a row
            iff that row's own context is non-negative -- deterministic and independent of
            batching. ``"model_batch"`` defers to upstream's batch-global rule, for parity
            with stock inference. ``"never"`` disables clamping entirely.
        force_flip_invariance: Averages the forecast with the negated-input forecast.
            ``None`` uses the checkpoint's default (on). Doubles inference cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str | None = None
    device: str = "auto"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    batch_size: int = Field(default=32, gt=0)
    clamp_negative: Literal["per_series", "model_batch", "never"] = "per_series"
    force_flip_invariance: bool | None = None


def _resolve_device(requested: str) -> str:
    """Turn ``"auto"`` into a concrete device string."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - platform dependent
        return "mps"
    return "cpu"


class TimesFmModel(ForecastModel):
    """TimesFM 2.5 wrapped in the :class:`~tsfm_peft.models.base.ForecastModel` interface."""

    name = "timesfm_2p5"

    def __init__(self, options: TimesFmOptions) -> None:
        """Load the checkpoint onto the resolved device in eval mode.

        Raises:
            ImportError: If the optional model dependencies are not installed.
        """
        try:
            import torch
            from transformers import TimesFm2_5ModelForPrediction
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MISSING_TORCH) from exc

        self._options = options
        self._device = _resolve_device(options.device)
        self._torch_dtype = getattr(torch, options.dtype)
        self.module = TimesFm2_5ModelForPrediction.from_pretrained(
            options.checkpoint, revision=options.revision
        )
        self.module = self.module.to(device=self._device, dtype=self._torch_dtype).eval()

        config = self.module.config
        # Index 0 of full_predictions is the point head's own slot; indices 1..len(quantiles)
        # carry the quantile heads, and decode_index points at the median among them. Read
        # from the config rather than hard-coded so a checkpoint with different heads either
        # works or fails loudly here.
        self._levels = tuple(float(q) for q in config.quantiles)
        self._median_index = int(config.decode_index)
        self._model_horizon = int(config.horizon_length)
        self._patch_length = int(config.patch_length)

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """The checkpoint's quantile heads: the deciles 0.1 through 0.9."""
        return self._levels

    @property
    def max_horizon(self) -> int | None:
        """One decode step's horizon; longer needs rollout, which v0.1 does not implement."""
        return self._model_horizon

    @property
    def batch_size(self) -> int:
        """Contexts per forward pass."""
        return self._options.batch_size

    @property
    def device(self) -> str:
        """The resolved device string."""
        return self._device

    def parameter_counts(self) -> dict[str, Any]:
        """Count total and trainable parameters of the loaded module."""
        return count_parameters(self.module)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the metrics artifact."""
        return {
            **super().describe(),
            "checkpoint": self._options.checkpoint,
            "revision": self._options.revision,
            "device": self._device,
            "dtype": self._options.dtype,
            "clamp_negative": self._options.clamp_negative,
            "force_flip_invariance": self._options.force_flip_invariance,
            "model_horizon": self._model_horizon,
            "patch_length": self._patch_length,
            "median_head_index": self._median_index,
        }

    def _check_context_length(self, context_length: int) -> None:
        """Reject context lengths the patch embedding cannot tile exactly."""
        if context_length % self._patch_length:
            raise ValueError(
                f"context_length {context_length} is not a multiple of the checkpoint's "
                f"patch_length {self._patch_length}; upstream would left-pad the context "
                "with zeros, which shifts the normalisation statistics and changes results"
            )

    def _predict_batch(self, contexts: Array, horizon: int) -> Forecast:
        """Run one forward pass and split point and quantile heads out of the output."""
        import torch

        context_length = int(contexts.shape[1])
        self._check_context_length(context_length)

        past_values = [
            torch.as_tensor(row, dtype=self._torch_dtype, device=self._device)
            for row in np.ascontiguousarray(contexts)
        ]
        # Upstream truncates to forecast_context_len and left-pads anything shorter. Passing
        # the batch's own length means it does neither: the harness owns the context, so a
        # protocol change shows up in the config rather than inside the model.
        with torch.no_grad():
            outputs = self.module(
                past_values=past_values,
                forecast_context_len=context_length,
                truncate_negative=(
                    None if self._options.clamp_negative == "model_batch" else False
                ),
                force_flip_invariance=self._options.force_flip_invariance,
            )

        full = outputs.full_predictions.to(torch.float32).cpu().numpy()
        point = full[:, :horizon, self._median_index].astype(np.float64)
        quantiles = full[:, :horizon, 1 : 1 + len(self._levels)].astype(np.float64)

        if self._options.clamp_negative == "per_series":
            # Row-wise version of upstream's clamp: a series whose observed history never
            # goes negative should not be forecast negative. Batch-independent by
            # construction, so the same window scores the same number at any batch size.
            non_negative = (contexts.min(axis=1) >= 0.0)[:, None]
            point = np.where(non_negative, np.maximum(point, 0.0), point)
            quantiles = np.where(non_negative[:, :, None], np.maximum(quantiles, 0.0), quantiles)

        return Forecast(point=point, quantiles=quantiles, levels=self._levels)


def count_parameters(module: Any) -> dict[str, Any]:
    """Return total, trainable and frozen parameter counts for a torch module.

    The headline number of any PEFT result: reporting accuracy without it hides the whole
    point of the method.

    Args:
        module: Any ``torch.nn.Module``.

    Returns:
        ``{"total_parameters", "trainable_parameters", "frozen_parameters", "trainable_pct"}``.
    """
    total = 0
    trainable = 0
    for parameter in module.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def build_timesfm(options: TimesFmOptions) -> TimesFmModel:
    """Build a :class:`TimesFmModel` from validated options."""
    return TimesFmModel(options)
