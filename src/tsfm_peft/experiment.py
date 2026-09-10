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
from tsfm_peft.results import artifact_path, build_artifact, write_artifact
from tsfm_peft.runtime import ResourceLog, collect_environment, set_seed


@dataclass(frozen=True)
class RunOutcome:
    """What a completed run produced.

    Attributes:
        config: The config that was run.
        split: The backtest split it was evaluated on.
        model: The adapter, after evaluation.
        result: The scored evaluation.
        artifact: The metrics artifact mapping.
        path: Where the artifact was written, or ``None`` if writing was suppressed.
    """

    config: ExperimentConfig
    split: BacktestSplit
    model: ForecastModel
    result: EvaluationResult
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

    with resources.phase("load_data"):
        split = config.data.build_split(force_download=force_download)

    with resources.phase("load_model"):
        model = config.model.build()

    result = evaluate_model(
        model,
        split,
        window_set="test",
        # The scaler is a property of the data config, but only an adapter that asks for
        # external scaling receives it; evaluate_model raises if the two disagree.
        scaler_kind=config.data.scaler if model.requires_external_scaling else None,
        resources=resources,
    )

    artifact = build_artifact(
        config,
        split,
        model,
        result,
        seed_record=seed_record,
        resources=resources,
        environment=collect_environment(),
    )

    path = None
    if write:
        path = write_artifact(artifact, artifact_path(config.name, results_dir))
    return RunOutcome(
        config=config, split=split, model=model, result=result, artifact=artifact, path=path
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
        f"WQL             {metrics.wql:.4f}  (macro {metrics.wql_macro:.4f})",
        f"crossings       {outcome.result.quantile_crossing_rate:.4f}",
        f"trainable       {trainable:,} / {total:,}"
        + (f" ({100.0 * trainable / total:.4f}%)" if total else ""),
        f"wall clock      {resources['total_seconds']:.1f}s",
        f"peak GPU        {peak_gpu / 1e9:.2f} GB" if peak_gpu else "peak GPU        n/a (CPU)",
        f"seed            {outcome.config.seed}",
    ]
    if outcome.path is not None:
        lines.append(f"artifact        {outcome.path}")
    return "\n".join(lines)
