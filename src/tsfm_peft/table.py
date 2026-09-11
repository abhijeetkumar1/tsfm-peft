"""Render the README results table from metrics artifacts.

The table is a pure function of the artifacts on disk, which are in turn a pure function of
the config set: one config file produces one artifact produces one row. Nothing here invents
a number, rounds one in from somewhere else, or lets a row appear without an artifact behind
it -- a config that has never been run renders as an explicitly pending row rather than as a
blank that a reader could mistake for a zero.

Two guards run before anything is rendered, because a results table's whole value is that
the rows are comparable:

* Every row in a dataset's table must have been scored on the same test windows. Horizon,
  context length, window count, stride and the dataset's own shape all feed the check.
  ``n_val_windows`` deliberately does not: validation windows are carved out of the fit
  region and the test origins are computed from the end of each series, so a zero-shot arm
  with no validation windows and a fine-tuned arm with several are still scored on exactly
  the same targets.
* Every row must come from the ``test`` window set. A validation score published as a test
  score is the single easiest way to accidentally publish a flattering lie.

Rows that fail either check raise rather than rendering, since a table that silently mixes
protocols is worse than no table.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsfm_peft.config import ExperimentConfig, load_experiment
from tsfm_peft.data.registry import DATASETS
from tsfm_peft.models.registry import MODELS
from tsfm_peft.results import load_artifacts

#: The README is edited between these, and only between these.
BEGIN_MARKER = "<!-- BEGIN RESULTS TABLE -->"
END_MARKER = "<!-- END RESULTS TABLE -->"

#: Printed for any cell whose value does not exist, matching the console summary's wording.
#: Never an empty cell, which reads as a zero at a glance.
MISSING = "n/a"

#: Printed for every cell of a config that has not been run yet.
PENDING = "--"

#: Cosmetic only, like DISPLAY_NAMES: an unmapped method falls back to its own key.
METHOD_NAMES = {"lora": "LoRA", "dora": "DoRA"}

#: Cosmetic only. An unregistered or unmapped model falls back to its registry key, so a new
#: adapter renders a slightly ugly row rather than no row.
DISPLAY_NAMES = {
    "seasonal_naive": "Seasonal naive",
    "timesfm_2p5": "TimesFM 2.5",
    "timesfm_2p5_tiny": "TimesFM 2.5 tiny (fixture)",
}

COLUMNS = (
    "Arm",
    "MASE",
    "sMAPE",
    "wMAPE",
    "WQL",
    "WQL (macro)",
    "Trained params",
    "Peak GPU",
    "Train time",
)

#: Arm column left-aligned, every number right-aligned so decimal points line up.
ALIGNMENTS = ("---", *("---:",) * (len(COLUMNS) - 1))


def _finetunable(model: str) -> bool:
    """Whether a model supports fine-tuning, ``False`` for one no longer registered.

    Artifacts outlive the registry: a model renamed or dropped after a run still has its
    artifact on disk, and the table should render that row with a plain label rather than
    crashing on a missing key.
    """
    return model in MODELS and MODELS[model].finetunable


@dataclass(frozen=True)
class Arm:
    """What a row is an instance of: a base model, optionally adapted.

    Two experiments on different datasets share an arm when these three fields match, which
    is what lets the aggregate section line a model up across datasets without matching on
    experiment names.

    Attributes:
        model: The registry key of the base model.
        method: ``"lora"``, ``"dora"``, or ``None`` for an unadapted run.
        rank: The adapter rank, or ``None`` when there is no adapter.
    """

    model: str
    method: str | None = None
    rank: int | None = None

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> Arm:
        """Read the arm out of a metrics artifact's model block."""
        model = artifact["model"]
        peft = model.get("peft")
        if not peft:
            return cls(model=model["name"])
        return cls(model=model["name"], method=peft["method"], rank=peft["rank"])

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Arm:
        """Read the arm out of a config, for an experiment that has not been run."""
        options = config.model.resolved_options()
        peft = getattr(options, "peft", None)
        if peft is None:
            return cls(model=config.model.name)
        return cls(model=config.model.name, method=peft.method, rank=peft.rank)

    @property
    def label(self) -> str:
        """Return the human label for the arm column."""
        base = DISPLAY_NAMES.get(self.model, self.model)
        if self.method is None:
            # "zero-shot" only means something for a model that could have been tuned;
            # calling the seasonal-naive baseline zero-shot would be noise.
            return f"{base} zero-shot" if _finetunable(self.model) else base
        method = METHOD_NAMES.get(self.method, self.method)
        return f"{base} + {method} r{self.rank}"

    @property
    def sort_key(self) -> tuple[int, int, str, str]:
        """Order arms as a reader expects to meet them: floor, baseline, then adapters."""
        if not _finetunable(self.model):
            group = 0
        elif self.method is None:
            group = 1
        elif self.method == "lora":
            group = 2
        elif self.method == "dora":
            group = 3
        else:
            group = 4
        return (group, self.rank or 0, self.model, self.method or "")


@dataclass(frozen=True)
class Row:
    """One line of the table, whether or not it has been run.

    Attributes:
        experiment: The experiment name, which is also its artifact's filename stem.
        dataset: The dataset registry key, which decides the row's section.
        arm: What was run.
        protocol: The backtest protocol, from the artifact or the config.
        stride: The effective stride, resolved from the protocol.
        artifact: The metrics artifact, or ``None`` for a config with no run behind it.
    """

    experiment: str
    dataset: str
    arm: Arm
    protocol: dict[str, Any]
    stride: int
    artifact: dict[str, Any] | None = None

    @property
    def pending(self) -> bool:
        """Whether this row is a config awaiting a run."""
        return self.artifact is None

    @property
    def results(self) -> dict[str, Any]:
        """The results block. Only valid when the row is not pending."""
        assert self.artifact is not None
        return self.artifact["results"]

    @property
    def train_seconds(self) -> float | None:
        """Wall-clock of the training phase, or ``None`` for an arm that did not train."""
        if self.artifact is None:
            return None
        for phase in self.artifact["resources"]["phases"]:
            if phase["label"] == "train":
                return float(phase["seconds"])
        return None

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> Row:
        """Build a row from a metrics artifact."""
        data = artifact["data"]
        return cls(
            experiment=artifact["experiment"]["name"],
            dataset=data["dataset"]["name"],
            arm=Arm.from_artifact(artifact),
            protocol=data["protocol"],
            stride=data["effective_stride"],
            artifact=artifact,
        )

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Row:
        """Build a pending row from a config that has no artifact."""
        return cls(
            experiment=config.name,
            dataset=config.data.dataset,
            arm=Arm.from_config(config),
            protocol=config.data.protocol.model_dump(),
            stride=config.data.protocol.effective_stride,
            artifact=None,
        )


def _format_metric(value: float | None) -> str:
    """Format an accuracy metric to three decimals."""
    return MISSING if value is None else f"{value:.3f}"


def _compact_count(count: int) -> str:
    """Abbreviate a parameter count, which is otherwise nine digits wide in a table cell."""
    if count >= 1_000_000_000:
        return f"{count / 1e9:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}K"
    return str(count)


def _format_parameters(artifact: dict[str, Any]) -> str:
    """Format the parameters a run actually updated, and their share of the whole model.

    Counted from the run, not from the model: a base model that was loaded and never trained
    still reports every one of its weights as ``requires_grad``, so reading the model block
    alone would credit the zero-shot arm with having trained 231M parameters. An arm with no
    training block updated nothing, and that is what the column says.

    The percentage is the number PEFT exists to make small, so it is always shown next to the
    count rather than left for the reader to divide out.
    """
    model = artifact["model"]
    total = model.get("total_parameters", 0)
    if not total:
        # A non-parametric baseline. "0" would imply it has weights it chose not to train,
        # which is a different claim.
        return MISSING
    if artifact.get("training") is None:
        return "0"
    trainable = model.get("trainable_parameters", 0)
    return f"{_compact_count(trainable)} ({100.0 * trainable / total:.2f}%)"


def _format_gpu(peak_bytes: int | None) -> str:
    """Format peak CUDA memory in GB, matching the console summary's units."""
    return MISSING if not peak_bytes else f"{peak_bytes / 1e9:.2f} GB"


def _format_duration(seconds: float | None) -> str:
    """Format a wall-clock duration at whatever resolution reads cleanly."""
    if seconds is None:
        return MISSING
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _comparability(row: Row) -> tuple[Any, ...]:
    """Return everything that decides which targets a row was scored against.

    Anything in this tuple differing between two rows of the same dataset means they are not
    measuring the same thing. ``n_val_windows`` is excluded on purpose -- see the module
    docstring.
    """
    assert row.artifact is not None
    dataset = row.artifact["data"]["dataset"]
    return (
        row.protocol["horizon"],
        row.protocol["context_length"],
        row.protocol["n_test_windows"],
        row.stride,
        dataset["n_series"],
        dataset["total_observations"],
    )


def _describe_comparability(row: Row) -> str:
    """Render a row's comparability tuple for an error message."""
    horizon, context, windows, stride, series, observations = _comparability(row)
    return (
        f"horizon={horizon} context={context} n_test_windows={windows} stride={stride} "
        f"n_series={series} total_observations={observations}"
    )


def check_comparable(rows: Sequence[Row]) -> None:
    """Raise unless every scored row in each dataset measures the same thing.

    Args:
        rows: The rows destined for one table. Pending rows are ignored, having no numbers
            to be incomparable with.

    Raises:
        ValueError: If a row was scored on the validation windows, or if two rows of the
            same dataset disagree about the protocol or the dataset itself.
    """
    for row in rows:
        if row.pending:
            continue
        window_set = row.results.get("window_set")
        if window_set != "test":
            raise ValueError(
                f"{row.experiment}: scored on the {window_set!r} windows, not 'test'. "
                "Validation scores select checkpoints; publishing one as a result reports a "
                "number the model was tuned against."
            )

    by_dataset: dict[str, list[Row]] = {}
    for row in rows:
        if not row.pending:
            by_dataset.setdefault(row.dataset, []).append(row)

    for dataset, group in by_dataset.items():
        reference = group[0]
        expected = _comparability(reference)
        for row in group[1:]:
            if _comparability(row) != expected:
                raise ValueError(
                    f"{dataset}: {row.experiment} was not scored on the same windows as "
                    f"{reference.experiment}, so they cannot share a table.\n"
                    f"  {reference.experiment}: {_describe_comparability(reference)}\n"
                    f"  {row.experiment}: {_describe_comparability(row)}\n"
                    "Point both experiments at the same configs/data/*.yaml file and re-run "
                    "the one that is stale."
                )


def is_fixture_row(row: Row) -> bool:
    """Whether a row exists to exercise the pipeline rather than to be published.

    The synthetic dataset and the random-init tiny checkpoint both run in CI on every push.
    Their numbers are real in the sense that they were measured, and meaningless in the sense
    that a random-init model forecasting generated data says nothing about anything -- so
    they stay out of the published table unless explicitly asked for.
    """
    dataset_is_fixture = row.dataset in DATASETS and DATASETS[row.dataset].is_generated
    model_is_fixture = row.arm.model in MODELS and MODELS[row.arm.model].is_fixture
    return dataset_is_fixture or model_is_fixture


def collect_rows(
    artifacts: Iterable[dict[str, Any]],
    configs: Iterable[ExperimentConfig] = (),
    *,
    include_fixtures: bool = False,
) -> list[Row]:
    """Build the row set from artifacts, filling in unrun configs as pending rows.

    An artifact always wins over a config of the same name: the artifact is what actually
    ran, and the config on disk may have been edited since.

    Args:
        artifacts: Loaded metrics artifacts.
        configs: Experiment configs, so that arms with no run yet still appear. Pass nothing
            to render only what has been run.
        include_fixtures: Keep the synthetic dataset and fixture-model rows that
            :func:`is_fixture_row` otherwise drops.

    Returns:
        Rows sorted by dataset, then by arm, then by experiment name.
    """
    rows: dict[str, Row] = {}
    for artifact in artifacts:
        row = Row.from_artifact(artifact)
        rows[row.experiment] = row
    for config in configs:
        if config.name not in rows:
            rows[config.name] = Row.from_config(config)
    kept = [row for row in rows.values() if include_fixtures or not is_fixture_row(row)]
    return sorted(kept, key=lambda row: (row.dataset, row.arm.sort_key, row.experiment))


def _render_markdown_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Render a markdown table. Cells are used verbatim, so they must not contain pipes."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(ALIGNMENTS) + "|",
    ]
    lines.extend("| " + " | ".join(cells) + " |" for cells in body)
    return "\n".join(lines)


def _row_cells(row: Row) -> list[str]:
    """Render one row's cells."""
    if row.pending:
        return [f"{row.arm.label} *(not run)*", *([PENDING] * (len(COLUMNS) - 1))]
    results = row.results
    artifact = row.artifact
    assert artifact is not None
    return [
        row.arm.label,
        _format_metric(results["mase"]),
        _format_metric(results["smape"]),
        # ``.get``: wMAPE was added after the first artifacts were written, and a row from
        # before it renders as missing rather than as a number the run never computed.
        _format_metric(results.get("wmape")),
        _format_metric(results["wql"]),
        _format_metric(results["wql_macro"]),
        _format_parameters(artifact),
        _format_gpu(artifact["resources"].get("peak_gpu_bytes")),
        _format_duration(row.train_seconds),
    ]


def _caption(rows: Sequence[Row]) -> str:
    """Describe the protocol a dataset's rows were scored under.

    Taken from the first scored row, which :func:`check_comparable` has already established
    agrees with every other scored row in the group. With nothing scored yet it falls back
    to the config, which is all a pending section can honestly claim.
    """
    scored = [row for row in rows if not row.pending]
    reference = scored[0] if scored else rows[0]
    protocol = reference.protocol
    parts = [
        f"horizon {protocol['horizon']}",
        f"context {protocol['context_length']}",
        f"{protocol['n_test_windows']} rolling origins per series",
    ]
    if scored:
        assert reference.artifact is not None
        results = reference.results
        parts.append(f"{results['n_series']} series")
        parts.append(f"{results['n_rows']} forecast windows")
    return "Test windows: " + ", ".join(parts) + "."


def _aggregate_rows(rows: Sequence[Row]) -> tuple[list[list[str]], list[str]]:
    """Average each arm's accuracy across the datasets it has been scored on.

    An arm appears only when it has a scored row on *every* dataset that has any scored row.
    An average taken over whichever subset happened to finish would move when an unrelated
    run completes, and would flatter whichever arm ran on the easier dataset.

    The pooled WQL and pooled wMAPE are excluded: both are dominated by whichever series
    have the largest magnitudes, so averaging them across datasets of different scale
    measures mostly the scales. The macro variants, which normalise per series, are averaged
    instead.

    Returns:
        The body rows, and the names of the datasets they average over.
    """
    scored = [row for row in rows if not row.pending]
    datasets = sorted({row.dataset for row in scored})
    if len(datasets) < 2:
        # One dataset is not an aggregate, it is the same table again.
        return [], datasets

    by_arm: dict[Arm, dict[str, Row]] = {}
    for row in scored:
        by_arm.setdefault(row.arm, {})[row.dataset] = row

    body: list[list[str]] = []
    for arm in sorted(by_arm, key=lambda a: a.sort_key):
        covered = by_arm[arm]
        if len(covered) != len(datasets):
            continue
        members = [covered[dataset] for dataset in datasets]

        def mean(metric: str, members: Sequence[Row] = members) -> float | None:
            """Average a metric across datasets, or ``None`` if any row predates it."""
            values = [row.results.get(metric) for row in members]
            if any(value is None for value in values):
                return None
            return sum(values) / len(values)  # type: ignore[arg-type]

        body.append(
            [
                arm.label,
                _format_metric(mean("mase")),
                _format_metric(mean("smape")),
                _format_metric(mean("wmape_macro")),
                _format_metric(mean("wql_macro")),
            ]
        )
    return body, datasets


def _provenance_notes(rows: Sequence[Row]) -> list[str]:
    """Return the caveats that belong under the table, derived from the artifacts.

    Every one of these is a way a table can be quietly wrong: numbers from a dirty tree
    cannot be tied to a commit, cost columns from different machines are not comparable, and
    rows that used different seeds differ partly for reasons the table does not show.
    """
    notes: list[str] = []
    if any(row.pending for row in rows):
        notes.append(f"`{PENDING}` marks an arm whose config exists but has not been run yet.")

    scored = [row for row in rows if not row.pending]
    if not scored:
        return notes

    environments = [row.artifact["environment"] for row in scored if row.artifact]

    commits = {env["git"]["commit"] for env in environments}
    if len(commits) == 1 and (commit := commits.pop()):
        notes.append(f"Produced at commit `{commit[:12]}`.")
    elif len(commits) > 1:
        notes.append(
            "Rows come from more than one commit; re-run the whole set from one tree before "
            "publishing."
        )
    if any(env["git"].get("dirty") for env in environments):
        notes.append(
            "At least one row was produced from a dirty working tree and cannot be tied to a "
            "commit."
        )

    machines = {
        tuple(device["name"] for device in env["hardware"]["devices"]) or ("CPU",)
        for env in environments
    }
    if len(machines) == 1:
        (machine,) = machines
        notes.append(f"Hardware: {', '.join(machine)}.")
    else:
        notes.append(
            "Rows were produced on more than one machine, so the peak-memory and train-time "
            "columns are not comparable between them."
        )

    seeds = {row.artifact["seed"]["seed"] for row in scored if row.artifact}
    if len(seeds) > 1:
        notes.append(f"Rows used different seeds ({', '.join(str(s) for s in sorted(seeds))}).")

    # ``.get`` because artifacts written before this was recorded have no such key, and an
    # absent record is not evidence of a deterministic run -- it is no evidence either way.
    fallback = sorted(
        row.experiment
        for row in scored
        if row.artifact and row.artifact["seed"].get("nondeterministic_kernels")
    )
    if fallback:
        notes.append(
            "Ran on a kernel with no deterministic implementation, so these rows reproduce "
            "only to within floating-point accumulation order, not bit-exactly: "
            f"{', '.join(f'`{name}`' for name in fallback)}. The artifact names the kernel."
        )

    return notes


def render(rows: Sequence[Row]) -> str:
    """Render the whole results block: one table per dataset, then the aggregate and notes.

    Args:
        rows: The rows from :func:`collect_rows`.

    Returns:
        The markdown block that goes between the README markers, without them.

    Raises:
        ValueError: If :func:`check_comparable` rejects the row set.
    """
    if not rows:
        return (
            "No runs yet. Every number in this table comes from a metrics artifact written "
            "by `tsfm-peft run`; until the experiments have been run there is nothing "
            "honest to put here."
        )

    check_comparable(rows)

    sections: list[str] = []
    for dataset in sorted({row.dataset for row in rows}):
        group = [row for row in rows if row.dataset == dataset]
        body = [_row_cells(row) for row in group]
        sections.append(
            f"### {dataset}\n\n{_caption(group)}\n\n{_render_markdown_table(COLUMNS, body)}"
        )

    aggregate, datasets = _aggregate_rows(rows)
    if aggregate:
        header = ("Arm", "MASE", "sMAPE", "wMAPE (macro)", "WQL (macro)")
        alignment = ("---", *("---:",) * (len(header) - 1))
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(alignment) + "|",
        ]
        lines.extend("| " + " | ".join(cells) + " |" for cells in aggregate)
        sections.append(
            f"### Aggregate\n\nUnweighted mean over {len(datasets)} datasets "
            f"({', '.join(datasets)}), for arms scored on all of them.\n\n" + "\n".join(lines)
        )

    notes = _provenance_notes(rows)
    if notes:
        sections.append("\n".join(f"- {note}" for note in notes))
    return "\n\n".join(sections)


def render_readme(readme: str, block: str) -> str:
    """Replace the marked region of a README with a freshly rendered block.

    Args:
        readme: The current README text.
        block: The output of :func:`render`.

    Returns:
        The updated README text.

    Raises:
        ValueError: If the markers are missing, duplicated, or out of order.
    """
    for marker in (BEGIN_MARKER, END_MARKER):
        if readme.count(marker) != 1:
            raise ValueError(
                f"expected exactly one {marker!r} in the README, found "
                f"{readme.count(marker)}; the table is written between the markers and "
                "cannot be placed without them"
            )
    start = readme.index(BEGIN_MARKER)
    end = readme.index(END_MARKER)
    if end < start:
        raise ValueError(f"{END_MARKER!r} appears before {BEGIN_MARKER!r} in the README")
    return f"{readme[:start]}{BEGIN_MARKER}\n\n{block}\n\n{readme[end:]}"


def build(
    results_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    *,
    include_fixtures: bool = False,
) -> str:
    """Load artifacts and configs from disk and render the block.

    Args:
        results_dir: Where artifacts live. Defaults to the results root.
        config_dir: Experiment config directory, so unrun arms appear as pending rows. Pass
            ``None`` to render only what has been run.
        include_fixtures: Keep synthetic and fixture-model rows.

    Returns:
        The rendered markdown block.
    """
    artifacts = load_artifacts(results_dir)
    configs: list[ExperimentConfig] = []
    if config_dir is not None:
        configs = [load_experiment(path) for path in sorted(Path(config_dir).glob("*.yaml"))]
    return render(collect_rows(artifacts, configs, include_fixtures=include_fixtures))
