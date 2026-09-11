"""The fine-tuning loop: fit adapters on the training prefix, select on validation.

What this module is careful about, in order of how expensive the mistake would be:

* **It only ever sees the fit region.** Training batches come from
  :func:`~tsfm_peft.data.windows.make_training_windows`, which cuts from ``split.train``.
  The validation windows it selects on sit *after* ``fit_end`` and are never trained on;
  the test windows are not touched here at all.
* **Selection is on the reported metric.** The checkpoint kept is the one with the best
  validation MASE, not the best validation loss. Selecting on one number and publishing
  another lets the two disagree about which step was best, and the published one loses.
* **It is deterministic, and says so only because it is checked.** Batch order comes from a
  seeded generator, so two runs of the same config see the same windows in the same order,
  and every knob that could change the result is in the config and lands in the artifact.
  The kernels are handled a level up: :func:`~tsfm_peft.runtime.set_seed` restricts attention
  to a backend whose backward pass is deterministic, because the fused ones are not and the
  gradients they produce differ in their last bits between runs. Anything that still falls
  back is recorded by :func:`~tsfm_peft.runtime.watch_nondeterminism` and named under the
  results table.
* **It reports cost, not just accuracy.** Steps, wall clock and peak memory are recorded
  alongside the trainable-parameter count, because a PEFT result that only reports accuracy
  is not reporting the thing the method is for.

Without validation windows the loop has nothing to select on and simply keeps the last step.
That is a legitimate configuration -- it trains on more data -- but it is recorded in the
artifact as ``selection: "last_step"`` so a reader can see no early stopping happened.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_peft.data.windows import BacktestSplit, make_training_windows, stack_windows
from tsfm_peft.evaluate import evaluate_model
from tsfm_peft.models.base import FineTunableModel, ForecastModel

#: Validation metrics a run may select its checkpoint on. All three are "lower is better".
SELECTION_METRICS = ("mase", "smape", "wql")


class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters, validated straight from YAML.

    Attributes:
        max_steps: Optimiser steps to run. Windows are cycled until this is reached, so a
            step count larger than one pass over the data means multiple epochs.
        batch_size: Windows per optimiser step, before accumulation.
        gradient_accumulation_steps: Micro-batches accumulated into one optimiser step. The
            effective batch is ``batch_size * gradient_accumulation_steps``; use it to keep
            an effective batch fixed while fitting a long context into less memory.
        learning_rate: Peak learning rate for AdamW. LoRA tolerates -- and usually needs --
            a rate one to two orders of magnitude above full fine-tuning.
        weight_decay: AdamW decay, applied to the adapter parameters only, since they are
            the only ones with gradients.
        warmup_steps: Linear warmup before the schedule proper.
        schedule: ``"cosine"`` decay to zero, or ``"constant"`` after warmup.
        grad_clip: Global gradient-norm clip. ``None`` disables it.
        train_stride: Step between training-window origins. Defaults to the protocol's
            horizon, so the targets within one epoch do not overlap.
        max_windows_per_series: Cap on training windows per series, keeping the ones nearest
            the fit boundary. Useful to stop a long series dominating a short one.
        eval_every: Steps between validation passes. ``0`` disables validation, which also
            disables checkpoint selection and early stopping.
        patience: Stop after this many validation passes without improvement. ``None``
            disables early stopping.
        selection_metric: Which validation metric selects the checkpoint.
        log_every: Steps between recorded training-loss points. Each recorded value is the
            mean loss since the previous point, which is more informative than a single
            noisy step and keeps the artifact small.
        save_adapter: Write the selected adapter weights next to the metrics artifact.
        seed_offset: Added to the experiment seed to seed batch shuffling, so changing the
            batch order is possible without changing model initialisation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(default=500, gt=0)
    batch_size: int = Field(default=16, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_steps: int = Field(default=0, ge=0)
    schedule: Literal["cosine", "constant"] = "cosine"
    grad_clip: float | None = Field(default=1.0, gt=0)
    train_stride: int | None = Field(default=None, gt=0)
    max_windows_per_series: int | None = Field(default=None, gt=0)
    eval_every: int = Field(default=50, ge=0)
    patience: int | None = Field(default=None, gt=0)
    selection_metric: Literal["mase", "smape", "wql"] = "mase"
    log_every: int = Field(default=10, gt=0)
    save_adapter: bool = True
    seed_offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _warmup_fits(self) -> TrainingConfig:
        """Reject a warmup that never ends, which would train at a fraction of the rate."""
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} must be fewer than max_steps "
                f"{self.max_steps}; otherwise the learning rate never reaches its peak"
            )
        return self


def learning_rate_multiplier(
    step: int, *, max_steps: int, warmup_steps: int = 0, schedule: str = "cosine"
) -> float:
    """Return the learning-rate multiplier for a zero-based optimiser step.

    Args:
        step: Zero-based step index.
        max_steps: Total steps the run will take.
        warmup_steps: Steps of linear warmup.
        schedule: ``"cosine"`` or ``"constant"``.

    Returns:
        A multiplier in ``[0, 1]``.
    """
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    span = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def iter_batches(
    n_windows: int, batch_size: int, n_batches: int, *, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield ``n_batches`` index arrays, reshuffling on every pass over the windows.

    Cycling with a fresh permutation per pass -- rather than sampling with replacement --
    means each window is seen the same number of times up to the final partial pass, so a
    step count does not quietly weight some windows more than others.

    Args:
        n_windows: Number of training windows available.
        batch_size: Windows per batch. The final batch of a pass may be smaller.
        n_batches: How many batches to yield.
        rng: Seeded generator; the only source of randomness in the batch order.

    Yields:
        Arrays of window indices.
    """
    if n_windows < 1:
        raise ValueError("cannot batch an empty set of training windows")
    order = rng.permutation(n_windows)
    cursor = 0
    for _ in range(n_batches):
        if cursor >= n_windows:
            order = rng.permutation(n_windows)
            cursor = 0
        yield order[cursor : cursor + batch_size]
        cursor += batch_size


@dataclass
class TrainingRecord:
    """What a fine-tuning run did, for the metrics artifact.

    Attributes:
        n_train_windows: Training windows the fit region yielded.
        steps: Optimiser steps actually taken, which is fewer than ``max_steps`` when early
            stopping triggered.
        epochs: Passes over the training windows those steps amount to.
        loss_curve: ``{"step", "loss"}`` points, each the mean loss since the previous one.
        val_curve: ``{"step", "mase", "smape", "wql"}`` points from each validation pass.
        best_step: Step whose weights were kept, or ``None`` without validation.
        best_metric: The selection metric's value at that step.
        selection: ``"best_val"`` or ``"last_step"``.
        selection_metric: Which validation metric drove selection.
        early_stopped: Whether patience ended the run before ``max_steps``.
        adapter_path: Where the selected adapter was written, if it was.
        parameters: Trainable/total parameter counts for the trained model.
    """

    n_train_windows: int
    steps: int = 0
    epochs: float = 0.0
    loss_curve: list[dict[str, float]] = field(default_factory=list)
    val_curve: list[dict[str, float]] = field(default_factory=list)
    best_step: int | None = None
    best_metric: float | None = None
    selection: Literal["best_val", "last_step"] = "last_step"
    selection_metric: str = "mase"
    early_stopped: bool = False
    adapter_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float | None:
        """The last recorded mean training loss, or ``None`` if nothing was recorded."""
        return self.loss_curve[-1]["loss"] if self.loss_curve else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "n_train_windows": self.n_train_windows,
            "steps": self.steps,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "loss_curve": self.loss_curve,
            "val_curve": self.val_curve,
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "selection": self.selection,
            "selection_metric": self.selection_metric,
            "early_stopped": self.early_stopped,
            "adapter_path": self.adapter_path,
            **self.parameters,
        }


def _validation_metrics(model: ForecastModel, split: BacktestSplit) -> dict[str, float]:
    """Score the validation windows through the ordinary evaluation path.

    Reusing :func:`~tsfm_peft.evaluate.evaluate_model` rather than writing a second scoring
    path is deliberate: a bespoke validation metric that drifts from the published one would
    select the wrong checkpoint and nothing would show it.
    """
    result = evaluate_model(model, split, window_set="val")
    return {
        "mase": float(result.metrics.mase),
        "smape": float(result.metrics.smape),
        "wql": float(result.metrics.wql),
    }


def train_model(
    model: ForecastModel,
    split: BacktestSplit,
    config: TrainingConfig,
    *,
    seed: int = 0,
    adapter_dir: str | Path | None = None,
) -> TrainingRecord:
    """Fine-tune a model's adapters on the fit region and select on validation.

    Args:
        model: A :class:`~tsfm_peft.models.base.FineTunableModel` with adapters attached.
        split: The backtest split. Only ``split.train`` and the validation windows are read.
        config: Hyperparameters.
        seed: Experiment seed; batch order is drawn from ``seed + config.seed_offset``.
        adapter_dir: Directory to write the selected adapter to, when
            ``config.save_adapter`` is set.

    Returns:
        The :class:`TrainingRecord`.

    Raises:
        TypeError: If the model does not support fine-tuning.
        ValueError: If the model has no trainable parameters, or validation was requested
            but the protocol produced no validation windows.
    """
    if not isinstance(model, FineTunableModel):
        raise TypeError(
            f"{model.name} does not support fine-tuning; remove the training block or use "
            "an adapter that does"
        )
    if not model.supports_checkpointing:
        raise ValueError(
            f"{model.name} has no adapters attached, so there is nothing to fine-tune "
            "efficiently and nothing to checkpoint. v0.1 trains with LoRA or DoRA only: "
            "add a peft block to the model options, or drop the training block."
        )
    parameters = model.trainable_parameters()
    if not parameters:  # pragma: no cover - apply_peft rejects this first
        raise ValueError(f"{model.name} has no trainable parameters")
    if config.eval_every and not split.val_windows:
        raise ValueError(
            "training asks for validation every "
            f"{config.eval_every} steps but the protocol produced no validation windows; "
            "set n_val_windows > 0 in the data config, or eval_every: 0 to train for a "
            "fixed number of steps and keep the last one"
        )

    # Imported after the checks above so a misconfigured run says what is wrong with the
    # config rather than reporting a missing dependency it never got far enough to need.
    import torch

    windows = make_training_windows(
        split, stride=config.train_stride, max_per_series=config.max_windows_per_series
    )
    contexts, targets, _ = stack_windows(windows)
    record = TrainingRecord(n_train_windows=len(windows), selection_metric=config.selection_metric)

    optimiser = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: learning_rate_multiplier(
            step,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            schedule=config.schedule,
        ),
    )

    rng = np.random.default_rng(seed + config.seed_offset)
    micro_batches = iter_batches(
        len(windows),
        config.batch_size,
        config.max_steps * config.gradient_accumulation_steps,
        rng=rng,
    )

    best_state: dict[str, Any] | None = None
    best_metric = math.inf
    since_improvement = 0
    running: list[float] = []
    windows_seen = 0

    for step in range(config.max_steps):
        model.set_train_mode(True)
        optimiser.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            index = next(micro_batches)
            windows_seen += len(index)
            loss = model.training_loss(contexts[index], targets[index])
            # Scale so the accumulated gradient equals the one a single batch of the
            # effective size would have produced, not its sum.
            (loss / config.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / config.gradient_accumulation_steps
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        optimiser.step()
        scheduler.step()

        running.append(step_loss)
        completed = step + 1
        if completed % config.log_every == 0 or completed == config.max_steps:
            record.loss_curve.append(
                {
                    "step": completed,
                    "loss": float(np.mean(running)),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            running = []

        if config.eval_every and (
            completed % config.eval_every == 0 or completed == config.max_steps
        ):
            model.set_train_mode(False)
            metrics = _validation_metrics(model, split)
            record.val_curve.append({"step": completed, **metrics})
            current = metrics[config.selection_metric]
            if current < best_metric:
                best_metric, best_state = current, model.checkpoint_state()
                since_improvement = 0
                record.best_step, record.best_metric = completed, current
            else:
                since_improvement += 1
                if config.patience is not None and since_improvement >= config.patience:
                    record.early_stopped = True
                    record.steps = completed
                    break

    model.set_train_mode(False)
    record.steps = record.steps or config.max_steps
    record.epochs = windows_seen / len(windows)

    if best_state is not None:
        # Restoring here is what makes the reported test metrics the selected step's, not
        # whatever the last step happened to leave behind.
        model.restore_checkpoint(best_state)
        record.selection = "best_val"

    record.parameters = model.parameter_counts()
    if config.save_adapter and adapter_dir is not None:
        record.adapter_path = str(model.save_checkpoint(adapter_dir))
    return record
