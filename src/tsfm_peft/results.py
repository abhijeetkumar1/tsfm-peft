"""The metrics artifact: the only place a benchmark number is allowed to come from.

Every run writes one JSON file. The README results table is generated from those files by
:mod:`scripts.build_readme_table`, never hand-edited, so a number in the README can always
be traced back to the config, seed, commit and machine that produced it -- and a number with
no artifact behind it simply cannot appear.

Artifacts are named after the experiment, so re-running a config overwrites its own artifact
rather than accumulating a pile of near-duplicates the table generator would have to guess
between. The table is then a pure function of the config set. Keep a run you care about by
copying the file, or by pointing ``TSFM_PEFT_RESULTS`` somewhere else.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsfm_peft.config import ExperimentConfig
from tsfm_peft.data.windows import BacktestSplit
from tsfm_peft.evaluate import EvaluationResult
from tsfm_peft.models.base import ForecastModel
from tsfm_peft.paths import results_root
from tsfm_peft.runtime import ResourceLog, collect_environment
from tsfm_peft.training import TrainingRecord

#: Bumped whenever the artifact layout changes incompatibly. The table generator refuses
#: artifacts it does not understand rather than silently misreading an old field.
SCHEMA_VERSION = 2


def adapter_path(name: str, root: str | Path | None = None) -> Path:
    """Return the directory an experiment's fine-tuned adapter is written to.

    Kept beside the artifacts rather than inside them: the adapter is binary and a few
    megabytes, and the artifact is meant to stay readable and diffable.

    Args:
        name: The experiment name.
        root: Results directory. Defaults to :func:`~tsfm_peft.paths.results_root`.

    Returns:
        The directory path, not created here.
    """
    directory = Path(root) if root is not None else results_root()
    return directory / "adapters" / name


def artifact_path(name: str, root: str | Path | None = None) -> Path:
    """Return the path an experiment's artifact is written to.

    Args:
        name: The experiment name.
        root: Directory to write into. Defaults to :func:`~tsfm_peft.paths.results_root`.

    Returns:
        The path, with parent directories created.
    """
    directory = Path(root) if root is not None else results_root()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.json"


def build_artifact(
    config: ExperimentConfig,
    split: BacktestSplit,
    model: ForecastModel,
    result: EvaluationResult,
    *,
    seed_record: dict[str, Any],
    resources: ResourceLog,
    environment: dict[str, Any] | None = None,
    training: TrainingRecord | None = None,
) -> dict[str, Any]:
    """Assemble the artifact for one run.

    Args:
        config: The validated experiment config, dumped verbatim so the run is repeatable
            from the artifact alone.
        split: The backtest split, for the dataset and protocol provenance.
        model: The evaluated adapter, for its description and parameter counts.
        result: The scored evaluation.
        seed_record: The return value of :func:`~tsfm_peft.runtime.set_seed`.
        resources: Timings and peak memory for each phase.
        environment: Provenance block; collected if not supplied.
        training: What the fine-tuning loop did, or ``None`` for a zero-shot arm.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": {"name": config.name, "notes": config.notes},
        "config": config.model_dump(mode="json"),
        "seed": seed_record,
        "data": split.describe(),
        "model": model.describe(),
        "training": training.to_dict() if training is not None else None,
        "results": result.to_dict(),
        "resources": resources.to_dict(),
        "environment": environment if environment is not None else collect_environment(),
    }


def write_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Write an artifact to disk as indented JSON with a trailing newline.

    Indented and key-stable so that two runs of the same config produce files that diff
    cleanly -- the quickest way to see what actually changed between them.

    Args:
        artifact: The mapping from :func:`build_artifact`.
        path: Destination file.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return destination


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Read an artifact and check its schema version.

    Args:
        path: The artifact file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file is not an artifact this version understands.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload)}")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: artifact schema version {version!r} but this build writes and reads "
            f"{SCHEMA_VERSION}. Re-run the experiment rather than reading it as if it "
            "matched; the fields may mean different things."
        )
    return payload


def load_artifacts(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load every artifact in a directory, sorted by experiment name.

    Args:
        root: Directory to scan. Defaults to :func:`~tsfm_peft.paths.results_root`.

    Returns:
        The artifacts. An empty list if the directory does not exist, since "no runs yet"
        is the normal state of a fresh clone rather than an error.
    """
    directory = Path(root) if root is not None else results_root()
    if not directory.is_dir():
        return []
    artifacts = [load_artifact(path) for path in sorted(directory.glob("*.json"))]
    return sorted(artifacts, key=lambda a: a["experiment"]["name"])
