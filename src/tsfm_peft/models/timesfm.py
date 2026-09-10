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

A fourth property matters only for fine-tuning, and it is a bug rather than a design choice:
**upstream's own training loss is misaligned and this adapter does not use it.** Passing
``future_values`` makes ``forward`` return ``mse + quantile_loss``, but the quantile term
builds its prediction tensor by dropping the ``decode_index`` column from the
``[point, q_0.1 ... q_0.9]`` output and then zips what remains against ``config.quantiles``
positionally. Level 0.1 is therefore scored against the *point* head, 0.2 against the 0.1
head and so on, while the median head -- the one this adapter reports as the point forecast
-- is excluded from the pinball term entirely. :meth:`TimesFmModel.training_loss` computes
the aligned loss instead; see its docstring for the normalisation it uses.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsfm_peft.models.base import Array, FineTunableModel, Forecast, as_context_batch
from tsfm_peft.models.lora import PeftOptions, apply_peft

#: The checkpoint v0.1 benchmarks. Pinned by name; the revision is pinned in the config.
DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-transformers"

#: dtype names accepted in configs, mapped to torch dtypes at build time.
DTYPES = ("float32", "bfloat16", "float16")

#: ``TimesFm2_5Model.tolerance``: below this the model treats a context as constant and
#: normalises by 1 instead of by its standard deviation. Mirrored here so the training loss
#: normalises in exactly the space the model does.
REVIN_TOLERANCE = 1e-6

_MISSING_TORCH = (
    "TimesFM needs the optional model dependencies. Install them with "
    "`uv sync --extra models` (or `pip install 'tsfm-peft[models]'`)."
)


class TimesFmRuntimeOptions(BaseModel):
    """Options shared by every TimesFM variant: device, precision, batching and PEFT.

    Attributes:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``.
        dtype: Compute dtype. ``float32`` is the default because reduced precision changes
            the third decimal place of the metrics, and this repo compares arms at that
            resolution.
        batch_size: Contexts per forward pass.
        clamp_negative: How to apply the non-negativity clamp. ``"per_series"`` clamps a row
            iff that row's own context is non-negative -- deterministic and independent of
            batching. ``"model_batch"`` defers to upstream's batch-global rule, for parity
            with stock inference. ``"never"`` disables clamping entirely. Inference only:
            the clamp is never applied while computing the training loss.
        force_flip_invariance: Averages the forecast with the negated-input forecast.
            ``None`` uses the checkpoint's default (on). Doubles inference cost -- and
            during fine-tuning it doubles the backward pass and the activation memory too,
            since gradients flow through both decodes. Whatever it is set to applies to
            training and evaluation alike, so the two cannot disagree about what the model
            computes.
        peft: LoRA/DoRA configuration. ``None`` is the zero-shot arm: no adapters are
            attached and nothing is trainable.
        point_loss_weight: Weight on the point head's MSE term in the training loss,
            relative to the pinball term over the nine quantile heads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = "auto"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    batch_size: int = Field(default=32, gt=0)
    clamp_negative: Literal["per_series", "model_batch", "never"] = "per_series"
    force_flip_invariance: bool | None = None
    peft: PeftOptions | None = None
    point_loss_weight: float = Field(default=1.0, ge=0.0)


class TimesFmOptions(TimesFmRuntimeOptions):
    """Configuration for :class:`TimesFmModel`.

    Attributes:
        checkpoint: HuggingFace repo id or local path.
        revision: Checkpoint revision. ``None`` resolves to whatever ``main`` points at
            today, which makes a run irreproducible six months from now; pin a commit sha
            for any result that goes in the README.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str | None = None


class TinyTimesFmOptions(TimesFmRuntimeOptions):
    """Configuration for :class:`TinyTimesFmModel`, the CPU smoke fixture.

    The architecture fields exist so the fixture can be made smaller or larger from a config
    file; the defaults are sized to train for a handful of steps in seconds on a laptop CPU.

    Attributes:
        hidden_size: Model width.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Attention heads per layer.
        head_dim: Dimension per head.
        intermediate_size: MLP width.
        patch_length: Patch size. Context lengths must be a multiple of it.
        horizon_length: Steps decoded per call; the adapter's ``max_horizon``.
        output_quantile_len: Length of the continuous quantile head's output.
        context_length: Longest context the position embeddings cover.
    """

    hidden_size: int = Field(default=32, gt=0)
    num_hidden_layers: int = Field(default=2, gt=0)
    num_attention_heads: int = Field(default=2, gt=0)
    head_dim: int = Field(default=16, gt=0)
    intermediate_size: int = Field(default=32, gt=0)
    patch_length: int = Field(default=8, gt=0)
    horizon_length: int = Field(default=32, gt=0)
    output_quantile_len: int = Field(default=32, gt=0)
    context_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=8, gt=0)


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


class BaseTimesFmModel(FineTunableModel):
    """Shared implementation for the TimesFM variants.

    Subclasses differ only in where the weights come from: the benchmarked adapter loads a
    pretrained checkpoint, the smoke fixture initialises a tiny one at random.
    """

    name = "unnamed"

    def __init__(self, options: TimesFmRuntimeOptions) -> None:
        """Build the module, attach adapters if configured, and put it in eval mode.

        Raises:
            ImportError: If the optional model dependencies are not installed.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MISSING_TORCH) from exc

        self._options = options
        self._device = _resolve_device(options.device)
        self._torch_dtype = getattr(torch, options.dtype)

        module = self._load_module().to(device=self._device, dtype=self._torch_dtype)

        config = module.config
        # Index 0 of full_predictions is the point head's own slot; indices 1..len(quantiles)
        # carry the quantile heads, and decode_index points at the median among them. Read
        # from the config rather than hard-coded so a checkpoint with different heads either
        # works or fails loudly here.
        self._levels = tuple(float(q) for q in config.quantiles)
        self._median_index = int(config.decode_index)
        self._model_horizon = int(config.horizon_length)
        self._patch_length = int(config.patch_length)

        if options.peft is not None:
            # Attached after the move so the adapter tensors are created on the target
            # device and in the target dtype rather than being copied there afterwards.
            module = apply_peft(module, options.peft)
        self._module = module.eval()

    def _load_module(self) -> Any:
        """Construct the underlying ``TimesFm2_5ModelForPrediction``."""
        raise NotImplementedError

    @property
    def module(self) -> Any:
        """The torch module, wrapped in a ``PeftModel`` when adapters are configured."""
        return self._module

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

    @property
    def peft_options(self) -> PeftOptions | None:
        """The PEFT configuration, or ``None`` for the zero-shot arm."""
        return self._options.peft

    def parameter_counts(self) -> dict[str, Any]:
        """Count total and trainable parameters of the loaded module."""
        return count_parameters(self._module)

    def _source_description(self) -> dict[str, Any]:
        """Return the fields describing where the weights came from."""
        return {}

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the metrics artifact."""
        return {
            **super().describe(),
            **self._source_description(),
            "device": self._device,
            "dtype": self._options.dtype,
            "clamp_negative": self._options.clamp_negative,
            "force_flip_invariance": self._options.force_flip_invariance,
            "model_horizon": self._model_horizon,
            "patch_length": self._patch_length,
            "median_head_index": self._median_index,
            "peft": self._options.peft.describe() if self._options.peft else None,
        }

    def _check_context_length(self, context_length: int) -> None:
        """Reject context lengths the patch embedding cannot tile exactly."""
        if context_length % self._patch_length:
            raise ValueError(
                f"context_length {context_length} is not a multiple of the checkpoint's "
                f"patch_length {self._patch_length}; upstream would left-pad the context "
                "with zeros, which shifts the normalisation statistics and changes results"
            )

    def _past_values(self, contexts: Array) -> list[Any]:
        """Turn a context batch into the list of 1-D tensors upstream's forward expects."""
        import torch

        return [
            torch.as_tensor(row, dtype=self._torch_dtype, device=self._device)
            for row in np.ascontiguousarray(contexts)
        ]

    def _predict_batch(self, contexts: Array, horizon: int) -> Forecast:
        """Run one forward pass and split point and quantile heads out of the output."""
        import torch

        context_length = int(contexts.shape[1])
        self._check_context_length(context_length)

        # Upstream truncates to forecast_context_len and left-pads anything shorter. Passing
        # the batch's own length means it does neither: the harness owns the context, so a
        # protocol change shows up in the config rather than inside the model.
        with torch.no_grad():
            outputs = self._module(
                past_values=self._past_values(contexts),
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

    def training_loss(self, contexts: Any, targets: Any) -> Any:
        """Return the aligned pinball + MSE loss for one batch, in normalised space.

        Two decisions are worth stating explicitly, because both are places a plausible
        alternative would quietly measure something else:

        * **Alignment.** The pinball term pairs level ``q_i`` with output column ``i + 1``,
          which is the head trained to emit it. Upstream's own loss does not (see the module
          docstring), so this does not call it.
        * **Normalisation.** The loss is computed after dividing predictions and targets by
          the same per-context location and scale the model normalises with internally, so a
          batch mixing ETTh1's large- and small-amplitude channels weights them equally
          rather than letting the loudest channel own the gradient. Because RevIN is affine,
          dividing the public ``full_predictions`` by those statistics recovers exactly the
          normalised forecast the model produced, with no reliance on private methods. The
          one exception is a context whose standard deviation falls below
          :data:`REVIN_TOLERANCE`: upstream denormalises such a row by a near-zero scale, so
          its forecast collapses onto the context mean and the row contributes almost no
          gradient. A constant context window is degenerate data, not a case worth modelling.

        The non-negativity clamp is never applied here: it is a non-differentiable
        postprocessing step, and wherever it binds it would zero the gradient.

        Args:
            contexts: ``(n_rows, context_length)`` raw observations from the fit region.
            targets: ``(n_rows, horizon)`` raw observations following each context.

        Returns:
            A scalar loss tensor.

        Raises:
            ValueError: If the shapes disagree or the horizon exceeds one decode step.
        """
        import torch

        batch = as_context_batch(contexts)
        # Copied rather than viewed: targets often arrive from a frozen Forecast or a
        # read-only window array, which torch refuses to wrap without a warning.
        target = np.array(targets, dtype=np.float64)
        if target.ndim != 2:
            raise ValueError(f"targets must be 2-D (n_rows, horizon); got {target.shape}")
        if target.shape[0] != batch.shape[0]:
            raise ValueError(
                f"got {batch.shape[0]} contexts but {target.shape[0]} targets; they must "
                "describe the same windows"
            )
        horizon = int(target.shape[1])
        if horizon > self._model_horizon:
            raise ValueError(
                f"{self.name}: horizon {horizon} exceeds the adapter's single-call maximum "
                f"{self._model_horizon}; v0.1 does not implement autoregressive rollout"
            )
        context_length = int(batch.shape[1])
        self._check_context_length(context_length)

        past_values = self._past_values(batch)
        outputs = self._module(
            past_values=past_values,
            forecast_context_len=context_length,
            truncate_negative=False,
            force_flip_invariance=self._options.force_flip_invariance,
        )

        # float32 throughout the loss even when the forward pass ran in bfloat16: the
        # reductions below are over context_length and horizon, where low-precision
        # accumulation is a real source of run-to-run disagreement.
        stacked = torch.stack(past_values).to(torch.float32)
        predictions = outputs.full_predictions[:, :horizon, :].to(torch.float32)
        future = torch.as_tensor(target, dtype=torch.float32, device=predictions.device)

        location = stacked.mean(dim=1, keepdim=True)
        scale = stacked.std(dim=1, keepdim=True)
        safe_scale = torch.where(scale < REVIN_TOLERANCE, torch.ones_like(scale), scale)

        normalised_predictions = (predictions - location[..., None]) / safe_scale[..., None]
        normalised_target = (future - location) / safe_scale

        return aligned_forecast_loss(
            normalised_predictions,
            normalised_target,
            self._levels,
            point_weight=self._options.point_loss_weight,
        )


class TimesFmModel(BaseTimesFmModel):
    """TimesFM 2.5 wrapped in the :class:`~tsfm_peft.models.base.ForecastModel` interface."""

    name = "timesfm_2p5"

    def _load_module(self) -> Any:
        """Load the pretrained checkpoint."""
        try:
            from transformers import TimesFm2_5ModelForPrediction
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MISSING_TORCH) from exc

        options: TimesFmOptions = self._options  # type: ignore[assignment]
        return TimesFm2_5ModelForPrediction.from_pretrained(
            options.checkpoint, revision=options.revision
        )

    def _source_description(self) -> dict[str, Any]:
        """Record which checkpoint and revision produced the numbers."""
        options: TimesFmOptions = self._options  # type: ignore[assignment]
        return {"checkpoint": options.checkpoint, "revision": options.revision}


class TinyTimesFmModel(BaseTimesFmModel):
    """A randomly initialised, few-thousand-parameter TimesFM 2.5, for CPU smoke tests.

    Its forecasts are meaningless -- the weights are noise -- and nothing it produces belongs
    in a results table. What it is for is proving the pipeline end to end without a GPU or a
    gigabyte of downloads: the real module names, the real forward pass and the real ``peft``
    wiring, in seconds. A hand-written stand-in would test the harness against a model that
    does not exist; this tests it against the architecture actually being benchmarked.
    """

    name = "timesfm_2p5_tiny"

    def _load_module(self) -> Any:
        """Build a tiny model from a config, with weights drawn from the seeded torch RNG."""
        try:
            from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MISSING_TORCH) from exc

        options: TinyTimesFmOptions = self._options  # type: ignore[assignment]
        config = TimesFm2_5Config(
            hidden_size=options.hidden_size,
            num_hidden_layers=options.num_hidden_layers,
            num_attention_heads=options.num_attention_heads,
            num_key_value_heads=options.num_attention_heads,
            head_dim=options.head_dim,
            intermediate_size=options.intermediate_size,
            patch_length=options.patch_length,
            horizon_length=options.horizon_length,
            output_quantile_len=options.output_quantile_len,
            context_length=options.context_length,
            max_position_embeddings=options.context_length,
        )
        return TimesFm2_5ModelForPrediction(config)

    def _source_description(self) -> dict[str, Any]:
        """Mark the artifact so a fixture run can never be mistaken for a benchmark."""
        return {"checkpoint": None, "revision": None, "random_init": True}


def aligned_forecast_loss(
    predictions: Any, target: Any, levels: tuple[float, ...], *, point_weight: float = 1.0
) -> Any:
    """Pinball loss over the quantile heads plus MSE on the point head.

    Column ``0`` of ``predictions`` is the point head and columns ``1..len(levels)`` are the
    quantile heads in the order ``levels`` gives them -- so level ``levels[i]`` is scored
    against column ``i + 1``. Getting that pairing wrong is silent: the loss still falls, the
    model still trains, and the heads simply learn the wrong quantiles. It is what upstream's
    own loss gets wrong (see the module docstring), which is why this exists.

    Args:
        predictions: ``(n_rows, horizon, 1 + n_levels)`` forecasts, normalised.
        target: ``(n_rows, horizon)`` observations, normalised the same way.
        levels: The quantile levels, matching the checkpoint's heads.
        point_weight: Weight on the point head's MSE relative to the pinball term.

    Returns:
        A scalar loss tensor.

    Raises:
        ValueError: If the prediction tensor does not carry one column per head.
    """
    import torch
    from torch.nn import functional

    expected = 1 + len(levels)
    if predictions.shape[-1] < expected:
        raise ValueError(
            f"predictions have {predictions.shape[-1]} output columns but {expected} are "
            f"needed for a point head plus {len(levels)} quantile heads"
        )
    point = predictions[..., 0]
    quantiles = predictions[..., 1:expected]
    level_tensor = torch.as_tensor(levels, dtype=quantiles.dtype, device=quantiles.device)

    errors = target[..., None] - quantiles
    pinball = torch.maximum((level_tensor - 1.0) * errors, level_tensor * errors).mean()
    return point_weight * functional.mse_loss(point, target) + pinball


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


def build_tiny_timesfm(options: TinyTimesFmOptions) -> TinyTimesFmModel:
    """Build a :class:`TinyTimesFmModel` from validated options."""
    return TinyTimesFmModel(options)
