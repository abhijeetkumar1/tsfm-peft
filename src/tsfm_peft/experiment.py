"""Running one experiment end to end: config in, metrics artifact out.

The order here is the order that matters for reproducibility. Seeding happens before the
model is constructed, because ``from_pretrained`` initialises any head it has to and reads
the global RNG when it does. Data loading happens before the model is built so that a
dataset problem surfaces in seconds rather than after a multi-gigabyte checkpoint download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsfm_peft.config import ExperimentConfig
from tsfm_peft.data.windows import BacktestSplit
from tsfm_peft.evaluate import EvaluationResult, evaluate_model
from tsfm_peft.models.base import ForecastModel
from tsfm_peft.results import adapter_path, artifact_path, build_artifact, write_artifact
from tsfm_peft.runtime import (
    ResourceLog,
    collect_environment,
    set_seed,
    watch_nondeterminism,
)
from tsfm_peft.training import TrainingRecord, train_model


@dataclass(frozen=True)
class RunOutcome:
    """What a completed run produced.

    Attributes:
        config: The config that was run.
        split: The backtest split it was evaluated on.
        model: The adapter, after training and evaluation.
        result: The scored evaluation.
        training: What the fine-tuning loop did, or ``None`` for a zero-shot arm.
        artifact: The metrics artifact mapping.
        path: Where the artifact was written, or ``None`` if writing was suppressed.
    """

    config: ExperimentConfig
    split: BacktestSplit
    model: ForecastModel
    result: EvaluationResult
    training: TrainingRecord | None
    artifact: dict[str, Any]
    path: Path | None


def run_experiment(
    config: ExperimentConfig,
    *,
    results_dir: str | Path | None = None,
    write: bool = True,
    force_download: bool = False,
) -> RunOutcome:
    """Run one experiment and write its metrics artifact.

    Args:
        config: The validated experiment config.
        results_dir: Directory for the artifact. Defaults to the configured results root.
        write: Whether to write the artifact. ``False`` is for tests and dry runs.
        force_download: Re-download the dataset even if a valid cached copy exists.

    Returns:
        The :class:`RunOutcome`.
    """
    resources = ResourceLog()
    seed_record = set_seed(config.seed, deterministic=config.deterministic)

    # Spans everything that touches a kernel, so a fallback in either training or inference
    # is recorded. Reading it afterwards is what turns "we asked for determinism" into
    # "here is where we did not get it".
    with watch_nondeterminism() as nondeterministic_kernels:
        with resources.phase("load_data"):
            split = config.data.build_split(force_download=force_download)

        with resources.phase("load_model"):
            model = config.model.build()

        training = None
        if config.training is not None:
            # LEAKAGE BOUNDARY: train_model reads split.train and the validation windows.
            # The test windows are scored below, after the selected checkpoint has been
            # restored, and nothing in this function hands them to the trainer.
            device = getattr(model, "device", None)
            with resources.phase("train", device=device):
                training = train_model(
                    model,
                    split,
                    config.training,
                    seed=config.seed,
                    adapter_dir=(
                        adapter_path(config.name, results_dir)
                        if write and config.training.save_adapter
                        else None
                    ),
                )

        result = evaluate_model(
            model,
            split,
            window_set="test",
            # The scaler is a property of the data config, but only an adapter that asks for
            # external scaling receives it; evaluate_model raises if the two disagree.
            scaler_kind=config.data.scaler if model.requires_external_scaling else None,
            resources=resources,
        )

    # Empty means every op had a deterministic implementation, so the run is bit-reproducible
    # on this machine. Non-empty names the ops that were not, and the table says so.
    seed_record["nondeterministic_kernels"] = list(nondeterministic_kernels)

    artifact = build_artifact(
        config,
        split,
        model,
        result,
        seed_record=seed_record,
        resources=resources,
        environment=collect_environment(),
        training=training,
    )

    path = None
    if write:
        path = write_artifact(artifact, artifact_path(config.name, results_dir))
    return RunOutcome(
        config=config,
        split=split,
        model=model,
        result=result,
        training=training,
        artifact=artifact,
        path=path,
    )


def summarise(outcome: RunOutcome) -> str:
    """Return a one-screen human summary of a run, for the console.

    Deliberately includes the cost columns next to the accuracy ones: a PEFT result that
    reports only accuracy is not reporting the thing the method is for.
    """
    metrics = outcome.result.metrics
    model = outcome.artifact["model"]
    resources = outcome.artifact["resources"]
    trainable = model.get("trainable_parameters", 0)
    total = model.get("total_parameters", 0)
    peak_gpu = resources.get("peak_gpu_bytes")
    lines = [
        f"experiment      {outcome.config.name}",
        f"dataset         {outcome.split.dataset.name} "
        f"({metrics.n_series} series, {metrics.n_rows} windows, horizon {metrics.horizon})",
        f"model           {model['name']}",
        f"MASE            {metrics.mase:.4f}",
        f"sMAPE           {metrics.smape:.4f}",
        f"wMAPE           {metrics.wmape:.4f}  (macro {metrics.wmape_macro:.4f})",
        f"WQL             {metrics.wql:.4f}  (macro {metrics.wql_macro:.4f})",
        f"crossings       {outcome.result.quantile_crossing_rate:.4f}",
        f"trainable       {trainable:,} / {total:,}"
        + (f" ({100.0 * trainable / total:.4f}%)" if total else ""),
        f"wall clock      {resources['total_seconds']:.1f}s",
        f"peak GPU        {peak_gpu / 1e9:.2f} GB" if peak_gpu else "peak GPU        n/a (CPU)",
        f"seed            {outcome.config.seed}",
    ]
    if outcome.training is not None:
        record = outcome.training
        selected = (
            f"step {record.best_step} of {record.steps} on val {record.selection_metric} "
            f"{record.best_metric:.4f}"
            if record.selection == "best_val"
            else f"last of {record.steps} steps, no validation"
        )
        lines[7:7] = [
            f"training        {record.steps} steps over {record.n_train_windows} windows "
            f"({record.epochs:.1f} epochs)",
            f"final loss      {record.final_loss:.4f}"
            if record.final_loss is not None
            else "final loss      n/a",
            f"selected        {selected}" + ("  [early stopped]" if record.early_stopped else ""),
        ]
    if outcome.artifact["seed"].get("nondeterministic_kernels"):
        lines.append("determinism     a kernel fell back; reproducible but not bit-exact")
    if outcome.path is not None:
        lines.append(f"artifact        {outcome.path}")
    if outcome.training is not None and outcome.training.adapter_path:
        lines.append(f"adapter         {outcome.training.adapter_path}")
    return "\n".join(lines)
