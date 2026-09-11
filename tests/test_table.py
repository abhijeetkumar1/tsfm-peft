"""The README table generator.

The table is the repo's public face, so these tests are mostly about what must *never*
reach it: a number with no artifact behind it, a validation score dressed up as a test
score, rows from incomparable protocols sitting in one table, or a zero-shot arm credited
with having trained the whole model.

Artifacts are built as dictionaries rather than by running experiments. The shape is pinned
by tests/test_experiment.py against the real writer; duplicating a run here would make these
tests slow and torch-dependent for no extra coverage.

These tests are torch-free and run in the CPU CI job.
"""

import json
from pathlib import Path

import pytest

from tsfm_peft.cli import main
from tsfm_peft.config import load_experiment
from tsfm_peft.table import (
    BEGIN_MARKER,
    COLUMNS,
    END_MARKER,
    MISSING,
    PENDING,
    Arm,
    check_comparable,
    collect_rows,
    render,
    render_readme,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"


def make_artifact(
    name="etth1-timesfm-lora",
    dataset="etth1",
    model="timesfm_2p5",
    peft=("lora", 16),
    trained=True,
    *,
    horizon=96,
    context_length=512,
    n_test_windows=8,
    n_val_windows=2,
    stride=None,
    mase=0.8,
    wmape=None,
    window_set="test",
    total_parameters=236_000_000,
    trainable_parameters=4_915_200,
    peak_gpu_bytes=None,
    train_seconds=None,
    commit="abc123def456789",
    dirty=False,
    devices=(),
    concurrent_runs=None,
    seed=0,
    n_series=7,
    total_observations=121_940,
):
    """Build a metrics artifact of the shape tsfm_peft.results.build_artifact writes."""
    model_block = {
        "name": model,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    if peft is not None:
        method, rank = peft
        model_block["peft"] = {"method": method, "rank": rank, "alpha": rank * 2}
    phases = [{"label": "predict_test", "seconds": 3.0, "peak_gpu_bytes": peak_gpu_bytes}]
    if train_seconds is not None:
        phases.insert(
            0, {"label": "train", "seconds": train_seconds, "peak_gpu_bytes": peak_gpu_bytes}
        )
    return {
        "schema_version": 2,
        "experiment": {"name": name, "notes": None},
        "config": {},
        "seed": {"seed": seed, "deterministic": True},
        "data": {
            "dataset": {
                "name": dataset,
                "n_series": n_series,
                "total_observations": total_observations,
            },
            "protocol": {
                "horizon": horizon,
                "context_length": context_length,
                "n_test_windows": n_test_windows,
                "n_val_windows": n_val_windows,
                "stride": stride,
            },
            "effective_stride": stride if stride is not None else horizon,
        },
        "model": model_block,
        "training": {"steps": 1000} if trained else None,
        "results": {
            "window_set": window_set,
            "mase": mase,
            "smape": 0.25,
            "wql": 0.12,
            "wql_macro": 0.15,
            "n_series": n_series,
            "n_rows": n_series * n_test_windows,
            **({} if wmape is None else {"wmape": wmape, "wmape_macro": wmape + 1.0}),
        },
        "resources": {
            "phases": phases,
            "total_seconds": sum(p["seconds"] for p in phases),
            "peak_gpu_bytes": peak_gpu_bytes,
        },
        "environment": {
            "git": {"commit": commit, "dirty": dirty},
            "hardware": {"devices": [{"name": d} for d in devices]},
            **(
                {}
                if concurrent_runs is None
                else {"scheduling": {"concurrent_runs": concurrent_runs}}
            ),
        },
    }


class TestArm:
    def test_baseline_is_not_called_zero_shot(self):
        # "zero-shot" only means something for a model that could have been tuned.
        assert Arm("seasonal_naive").label == "Seasonal naive"

    def test_untuned_foundation_model_is_zero_shot(self):
        assert Arm("timesfm_2p5").label == "TimesFM 2.5 zero-shot"

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("lora", "TimesFM 2.5 + LoRA r16"), ("dora", "TimesFM 2.5 + DoRA r16")],
    )
    def test_method_casing(self, method, expected):
        assert Arm("timesfm_2p5", method, 16).label == expected

    def test_a_model_no_longer_registered_still_renders(self):
        # Artifacts outlive the registry. A renamed or dropped model must degrade to a
        # plain label rather than crash the whole table.
        assert Arm("retired_model").label == "retired_model"
        assert Arm("retired_model").sort_key[0] == 0

    def test_ordering_runs_floor_baseline_then_adapters_by_rank(self):
        arms = [
            Arm("timesfm_2p5", "dora", 16),
            Arm("timesfm_2p5", "lora", 32),
            Arm("seasonal_naive"),
            Arm("timesfm_2p5", "lora", 4),
            Arm("timesfm_2p5", "lora", 128),
            Arm("timesfm_2p5"),
        ]
        # r128 after r32, not between r128 and r4: the rank sorts as a number, and a table
        # that ordered it as a string would put the largest adapter in the middle.
        assert [a.label for a in sorted(arms, key=lambda a: a.sort_key)] == [
            "Seasonal naive",
            "TimesFM 2.5 zero-shot",
            "TimesFM 2.5 + LoRA r4",
            "TimesFM 2.5 + LoRA r32",
            "TimesFM 2.5 + LoRA r128",
            "TimesFM 2.5 + DoRA r16",
        ]

    def test_read_from_artifact(self):
        assert Arm.from_artifact(make_artifact(peft=("dora", 8))) == Arm("timesfm_2p5", "dora", 8)

    def test_read_from_artifact_without_peft(self):
        assert Arm.from_artifact(make_artifact(peft=None)) == Arm("timesfm_2p5")

    def test_read_from_config(self):
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-dora.yaml")
        assert Arm.from_config(config) == Arm("timesfm_2p5", "dora", 16)

    def test_read_from_config_without_peft(self):
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-zeroshot.yaml")
        assert Arm.from_config(config) == Arm("timesfm_2p5")

    def test_read_from_config_of_a_non_peft_model(self):
        config = load_experiment(EXPERIMENTS / "etth1-seasonal-naive.yaml")
        assert Arm.from_config(config) == Arm("seasonal_naive")


class TestComparabilityGuards:
    def test_matching_rows_pass(self):
        rows = collect_rows([make_artifact(name="a"), make_artifact(name="b", peft=None)])
        check_comparable(rows)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("horizon", 48),
            ("context_length", 256),
            ("n_test_windows", 4),
            ("stride", 48),
            ("n_series", 6),
            ("total_observations", 999),
        ],
    )
    def test_a_row_scored_on_different_windows_is_refused(self, field, value):
        rows = collect_rows([make_artifact(name="a"), make_artifact(name="b", **{field: value})])
        with pytest.raises(ValueError, match="not scored on the same windows"):
            check_comparable(rows)

    def test_validation_windows_do_not_break_comparability(self):
        # Test origins are computed from the end of each series, so reserving validation
        # windows shortens the fit region without moving a single test target. A zero-shot
        # arm with no validation windows is still comparable to a fine-tuned arm with them.
        rows = collect_rows(
            [
                make_artifact(name="zero", peft=None, trained=False, n_val_windows=0),
                make_artifact(name="lora", n_val_windows=2),
            ]
        )
        check_comparable(rows)

    def test_different_datasets_are_not_compared_to_each_other(self):
        rows = collect_rows(
            [
                make_artifact(name="a", dataset="etth1", horizon=96),
                make_artifact(name="b", dataset="nn5_daily", horizon=56, n_series=111),
            ]
        )
        check_comparable(rows)

    def test_a_validation_score_is_refused(self):
        rows = collect_rows([make_artifact(window_set="val")])
        with pytest.raises(ValueError, match="not 'test'"):
            check_comparable(rows)

    def test_render_applies_the_guards(self):
        rows = collect_rows([make_artifact(name="a"), make_artifact(name="b", horizon=48)])
        with pytest.raises(ValueError, match="not scored on the same windows"):
            render(rows)


class TestTrainedParameters:
    def cell(self, artifact, header="Trained params"):
        """Return the named column of the single data row the artifact renders to."""
        lines = render(collect_rows([artifact])).splitlines()
        columns = [c.strip() for c in lines[4].strip("| ").split(" | ")]
        cells = [c.strip() for c in lines[6].strip("| ").split(" | ")]
        return dict(zip(columns, cells, strict=True))[header]

    def test_a_zero_shot_arm_trained_nothing(self):
        # The base model reports every weight as requires_grad, so reading the model block
        # alone would credit zero-shot with having trained 236M parameters.
        assert (
            self.cell(make_artifact(peft=None, trained=False, trainable_parameters=236_000_000))
            == "0"
        )

    def test_a_non_parametric_baseline_has_no_count(self):
        artifact = make_artifact(
            model="seasonal_naive",
            peft=None,
            trained=False,
            total_parameters=0,
            trainable_parameters=0,
        )
        assert self.cell(artifact) == MISSING

    def test_a_tuned_arm_reports_the_count_and_the_share(self):
        assert self.cell(make_artifact()) == "4.92M (2.08%)"


class TestFormatting:
    def cells(self, artifact):
        lines = render(collect_rows([artifact])).splitlines()
        columns = [c.strip() for c in lines[4].strip("| ").split(" | ")]
        values = [c.strip() for c in lines[6].strip("| ").split(" | ")]
        return dict(zip(columns, values, strict=True))

    def test_metrics_are_three_decimals(self):
        cells = self.cells(make_artifact(mase=0.8333333))
        assert cells["MASE"] == "0.833"

    def test_peak_gpu_on_a_cpu_run_is_not_blank(self):
        # A blank cell reads as zero; "n/a" cannot be misread as "used no memory".
        assert self.cells(make_artifact(peak_gpu_bytes=None))["Peak GPU"] == MISSING

    def test_peak_gpu_is_reported_in_gb(self):
        assert self.cells(make_artifact(peak_gpu_bytes=12_300_000_000))["Peak GPU"] == "12.30 GB"

    def test_an_arm_that_did_not_train_has_no_train_time(self):
        assert self.cells(make_artifact(train_seconds=None))["Train time"] == MISSING

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(9.4, "9s"), (75.0, "1m 15s"), (3_661.0, "1h 01m")],
    )
    def test_durations_scale_with_magnitude(self, seconds, expected):
        assert self.cells(make_artifact(train_seconds=seconds))["Train time"] == expected

    def test_numeric_columns_are_right_aligned(self):
        alignment = render(collect_rows([make_artifact()])).splitlines()[5]
        assert alignment == "|---|" + "---:|" * (len(COLUMNS) - 1)


class TestPendingRows:
    def test_a_config_with_no_artifact_renders_as_pending(self):
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-lora.yaml")
        rows = collect_rows([], [config])
        assert len(rows) == 1
        assert rows[0].pending
        block = render(rows)
        assert "*(not run)*" in block
        assert PENDING in block

    def test_a_pending_row_carries_no_numbers(self):
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-lora.yaml")
        data_row = render(collect_rows([], [config])).splitlines()[6]
        values = [c.strip() for c in data_row.strip("| ").split(" | ")][1:]
        assert set(values) == {PENDING}

    def test_the_pending_marker_is_explained_even_when_nothing_has_run(self):
        # The all-pending table is the state of a fresh clone, so this is exactly when the
        # legend matters most.
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-lora.yaml")
        assert "has not been run yet" in render(collect_rows([], [config]))

    def test_an_artifact_wins_over_its_config(self):
        config = load_experiment(EXPERIMENTS / "etth1-timesfm-lora.yaml")
        artifact = make_artifact(name=config.name)
        (row,) = collect_rows([artifact], [config])
        assert not row.pending


class TestFixtureRows:
    def test_the_synthetic_dataset_is_excluded(self):
        rows = collect_rows([make_artifact(dataset="synthetic")])
        assert rows == []

    def test_the_fixture_model_is_excluded(self):
        rows = collect_rows([make_artifact(model="timesfm_2p5_tiny")])
        assert rows == []

    def test_fixtures_can_be_asked_for(self):
        rows = collect_rows([make_artifact(dataset="synthetic")], include_fixtures=True)
        assert len(rows) == 1

    def test_an_empty_row_set_says_so_rather_than_rendering_an_empty_table(self):
        assert "No runs yet" in render([])


def two_dataset_rows(**overrides: object):
    """Artifacts for the same three arms on two datasets."""
    artifacts = []
    for dataset, horizon, n_series in (("etth1", 96, 7), ("nn5_daily", 56, 111)):
        for suffix, peft, trained in (
            ("zeroshot", None, False),
            ("lora", ("lora", 16), True),
        ):
            artifacts.append(
                make_artifact(
                    name=f"{dataset}-timesfm-{suffix}",
                    dataset=dataset,
                    peft=peft,
                    trained=trained,
                    horizon=horizon,
                    n_series=n_series,
                    **overrides,
                )
            )
    return artifacts


class TestAggregate:
    def test_one_dataset_gets_no_aggregate(self):
        # Averaging one dataset is the same table printed twice.
        assert "### Aggregate" not in render(collect_rows([make_artifact()]))

    def test_arms_scored_on_every_dataset_are_averaged(self):
        block = render(collect_rows(two_dataset_rows()))
        assert "### Aggregate" in block
        assert "Unweighted mean over 2 datasets" in block

    def test_an_arm_missing_from_a_dataset_is_left_out(self):
        artifacts = two_dataset_rows()
        artifacts.append(
            make_artifact(name="etth1-timesfm-dora", dataset="etth1", peft=("dora", 16), horizon=96)
        )
        aggregate = render(collect_rows(artifacts)).split("### Aggregate")[1]
        # An average over whichever dataset happened to finish would move when an unrelated
        # run completes, so a partially covered arm is dropped rather than averaged.
        assert "DoRA" not in aggregate
        assert "LoRA r16" in aggregate

    def test_the_aggregate_averages_the_macro_wql_not_the_pooled_one(self):
        # Pooled WQL is dominated by whichever series are largest, so averaging it across
        # datasets of different scale would mostly measure the scales.
        aggregate = render(collect_rows(two_dataset_rows())).split("### Aggregate")[1]
        header = next(line for line in aggregate.splitlines() if line.startswith("| Arm"))
        assert "WQL (macro)" in header
        assert "| WQL |" not in header

    def test_pending_rows_are_not_averaged(self):
        config = load_experiment(EXPERIMENTS / "nn5_daily-timesfm-dora.yaml")
        rows = collect_rows(two_dataset_rows(), [config])
        assert "DoRA" not in render(rows).split("### Aggregate")[1]


class TestWmapeColumn:
    def header(self, artifacts):
        """The per-dataset table's header row."""
        return render(collect_rows(artifacts)).splitlines()[4]

    def test_column_is_present(self):
        assert "| wMAPE |" in self.header([make_artifact(wmape=23.4)])

    def test_renders_the_pooled_value(self):
        table = render(collect_rows([make_artifact(wmape=23.456)]))
        assert "23.456" in table

    def body(self, artifacts):
        """The first body row of the per-dataset table, as stripped cells."""
        line = render(collect_rows(artifacts)).splitlines()[6]
        return [cell.strip() for cell in line.strip("|").split("|")]

    def test_sits_between_smape_and_wql(self):
        assert self.body([make_artifact(wmape=23.456)])[3] == "23.456"

    def test_rows_written_before_wmape_render_as_missing(self):
        # The alternative -- a blank, or a zero -- reads as a measurement rather than as an
        # artifact that predates the metric.
        assert self.body([make_artifact()])[3] == MISSING

    def test_aggregate_averages_the_macro_variant(self):
        rows = collect_rows(
            [
                make_artifact(name="etth1-timesfm-lora", dataset="etth1", wmape=10.0),
                make_artifact(name="nn5_daily-timesfm-lora", dataset="nn5_daily", wmape=20.0),
            ]
        )
        aggregate = render(rows).split("### Aggregate")[1]
        assert "wMAPE (macro)" in aggregate
        # macro is pooled + 1.0 in the fixture, so the mean of 11.0 and 21.0.
        assert "16.000" in aggregate

    def test_aggregate_skips_it_when_a_dataset_predates_it(self):
        rows = collect_rows(
            [
                make_artifact(name="etth1-timesfm-lora", dataset="etth1", wmape=10.0),
                make_artifact(name="nn5_daily-timesfm-lora", dataset="nn5_daily"),
            ]
        )
        aggregate = render(rows).split("### Aggregate")[1]
        assert MISSING in aggregate


class TestContentionNotes:
    def notes(self, artifacts):
        """The bullet list rendered under the table."""
        return [
            line for line in render(collect_rows(artifacts)).splitlines() if line.startswith("- ")
        ]

    def test_names_the_rows_that_shared_the_machine(self):
        notes = self.notes(
            [
                make_artifact(name="etth1-timesfm-lora", concurrent_runs=2),
                make_artifact(name="etth1-timesfm-dora", concurrent_runs=1),
            ]
        )
        (note,) = [n for n in notes if "shared the machine" in n]
        assert "`etth1-timesfm-lora`" in note
        assert "`etth1-timesfm-dora`" not in note

    def test_silent_when_every_row_had_the_host_to_itself(self):
        notes = self.notes([make_artifact(concurrent_runs=1)])
        assert not any("shared the machine" in n for n in notes)

    def test_silent_for_artifacts_written_before_the_field(self):
        # An unrecorded run is not evidence of an uncontended one, but it is not evidence of
        # contention either, and a note has to be about something the artifact actually says.
        assert not any("shared the machine" in n for n in self.notes([make_artifact()]))


class TestProvenanceNotes:
    def notes(self, artifacts, configs=()):
        return [
            line
            for line in render(collect_rows(artifacts, configs)).splitlines()
            if line.startswith("- ")
        ]

    def test_a_single_clean_commit_is_reported(self):
        assert any("`abc123def456`" in note for note in self.notes([make_artifact()]))

    def test_a_dirty_tree_is_called_out(self):
        notes = self.notes([make_artifact(dirty=True)])
        assert any("dirty working tree" in note for note in notes)

    def test_rows_from_different_commits_are_called_out(self):
        notes = self.notes(
            [make_artifact(name="a", commit="aaa"), make_artifact(name="b", commit="bbb")]
        )
        assert any("more than one commit" in note for note in notes)

    def test_rows_from_different_machines_invalidate_the_cost_columns(self):
        notes = self.notes(
            [
                make_artifact(name="a", devices=("NVIDIA A100",)),
                make_artifact(name="b", devices=()),
            ]
        )
        assert any("not comparable" in note for note in notes)

    def test_a_single_machine_is_named(self):
        notes = self.notes([make_artifact(devices=("NVIDIA A100-SXM4-40GB",))])
        assert any("NVIDIA A100-SXM4-40GB" in note for note in notes)

    def test_mixed_seeds_are_called_out(self):
        notes = self.notes([make_artifact(name="a", seed=0), make_artifact(name="b", seed=1)])
        assert any("different seeds" in note for note in notes)

    def test_notes_which_rows_were_not_bit_exact(self):
        loose = make_artifact(name="etth1-timesfm-lora", trained=True)
        loose["seed"]["nondeterministic_kernels"] = ["attention used a non-deterministic algo"]
        notes = self.notes([make_artifact(name="etth1-timesfm-zeroshot"), loose])
        (note,) = [n for n in notes if "bit-exactly" in n]
        assert "`etth1-timesfm-lora`" in note
        assert "`etth1-timesfm-zeroshot`" not in note

    def test_says_nothing_when_every_kernel_was_deterministic(self):
        artifact = make_artifact()
        artifact["seed"]["nondeterministic_kernels"] = []
        assert not any("bit-exactly" in note for note in self.notes([artifact]))

    def test_says_nothing_for_artifacts_written_before_it_was_recorded(self):
        # No key at all: absence of a record is not a record of absence, so claim nothing.
        assert not any("bit-exactly" in note for note in self.notes([make_artifact()]))


class TestReadmeInjection:
    def readme(self, body="old table"):
        return f"# Title\n\n{BEGIN_MARKER}\n\n{body}\n\n{END_MARKER}\n\n## Method\n"

    def test_the_block_replaces_what_is_between_the_markers(self):
        updated = render_readme(self.readme(), "new table")
        assert "old table" not in updated
        assert "new table" in updated

    def test_everything_outside_the_markers_survives(self):
        updated = render_readme(self.readme(), "new table")
        assert updated.startswith("# Title\n")
        assert updated.endswith("## Method\n")

    def test_writing_twice_is_the_same_as_writing_once(self):
        # This is what makes `--check` trustworthy: a stale README differs from a fresh one
        # only when the numbers differ, never because of whitespace drift.
        once = render_readme(self.readme(), "new table")
        assert render_readme(once, "new table") == once

    @pytest.mark.parametrize(
        "readme",
        [
            "# Title\n\nno markers at all\n",
            f"# Title\n{BEGIN_MARKER}\nbody\n",
            f"# Title\n{END_MARKER}\nbody\n",
            f"{BEGIN_MARKER}\na\n{END_MARKER}\n{BEGIN_MARKER}\nb\n{END_MARKER}\n",
        ],
    )
    def test_missing_or_duplicated_markers_are_refused(self, readme):
        with pytest.raises(ValueError, match="expected exactly one"):
            render_readme(readme, "new table")

    def test_markers_in_the_wrong_order_are_refused(self):
        readme = f"# Title\n{END_MARKER}\nbody\n{BEGIN_MARKER}\n"
        with pytest.raises(ValueError, match="appears before"):
            render_readme(readme, "new table")


class TestShippedConfigs:
    """The generator against the real config set, which is what CI will run it on."""

    def configs(self):
        return [load_experiment(path) for path in sorted(EXPERIMENTS.glob("*.yaml"))]

    def test_every_shipped_arm_gets_a_row(self):
        rows = collect_rows([], self.configs())
        by_dataset = {}
        for row in rows:
            by_dataset.setdefault(row.dataset, []).append(row.arm.label)
        assert set(by_dataset) == {"etth1", "nn5_daily"}
        assert by_dataset["etth1"] == [
            "Seasonal naive",
            "TimesFM 2.5 zero-shot",
            "TimesFM 2.5 + LoRA r4",
            "TimesFM 2.5 + LoRA r8",
            "TimesFM 2.5 + LoRA r16",
            "TimesFM 2.5 + LoRA r32",
            "TimesFM 2.5 + LoRA r64",
            "TimesFM 2.5 + LoRA r128",
            "TimesFM 2.5 + DoRA r4",
            "TimesFM 2.5 + DoRA r8",
            "TimesFM 2.5 + DoRA r16",
            "TimesFM 2.5 + DoRA r32",
            "TimesFM 2.5 + DoRA r64",
            "TimesFM 2.5 + DoRA r128",
        ]
        assert by_dataset["nn5_daily"] == [
            "Seasonal naive",
            "TimesFM 2.5 zero-shot",
            "TimesFM 2.5 + LoRA r16",
            "TimesFM 2.5 + DoRA r16",
        ]

    def test_the_synthetic_smoke_configs_stay_out_of_the_table(self):
        assert all(row.dataset != "synthetic" for row in collect_rows([], self.configs()))

    def test_the_unrun_table_contains_no_numbers_that_look_like_results(self):
        # The state of a fresh clone. Every accuracy cell must be the pending marker: a
        # README that shipped a plausible-looking zero would be worse than one with no table.
        block = render(collect_rows([], self.configs()))
        for line in block.splitlines():
            if line.startswith("| ") and "*(not run)*" in line:
                values = [c.strip() for c in line.strip("| ").split(" | ")][1:]
                assert set(values) == {PENDING}


class TestCommandLine:
    """The ``table`` subcommand, which is what CI and the release flow actually call."""

    def write_artifact(self, directory, artifact):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact['experiment']['name']}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def readme(self, path, body="placeholder"):
        path.write_text(
            f"# tsfm-peft\n\n{BEGIN_MARKER}\n\n{body}\n\n{END_MARKER}\n\n## Method\n",
            encoding="utf-8",
        )
        return path

    def test_it_prints_the_table_by_default(self, tmp_path, capsys):
        self.write_artifact(tmp_path / "results", make_artifact())
        assert main(["table", "--results-dir", str(tmp_path / "results")]) == 0
        assert "### etth1" in capsys.readouterr().out

    def test_write_updates_the_readme(self, tmp_path):
        self.write_artifact(tmp_path / "results", make_artifact())
        readme = self.readme(tmp_path / "README.md")
        code = main(
            [
                "table",
                "--results-dir",
                str(tmp_path / "results"),
                "--readme",
                str(readme),
                "--write",
            ]
        )
        assert code == 0
        text = readme.read_text(encoding="utf-8")
        assert "placeholder" not in text
        assert "TimesFM 2.5 + LoRA r16" in text
        assert text.endswith("## Method\n")

    def test_check_passes_on_a_freshly_written_readme(self, tmp_path):
        self.write_artifact(tmp_path / "results", make_artifact())
        readme = self.readme(tmp_path / "README.md")
        args = ["table", "--results-dir", str(tmp_path / "results"), "--readme", str(readme)]
        assert main([*args, "--write"]) == 0
        assert main([*args, "--check"]) == 0

    def test_check_fails_on_a_stale_readme(self, tmp_path, capsys):
        self.write_artifact(tmp_path / "results", make_artifact())
        readme = self.readme(tmp_path / "README.md")
        args = ["table", "--results-dir", str(tmp_path / "results"), "--readme", str(readme)]
        assert main([*args, "--write"]) == 0
        # A new run lands. The README now claims something the artifacts do not say.
        self.write_artifact(
            tmp_path / "results", make_artifact(name="etth1-timesfm-dora", peft=("dora", 16))
        )
        assert main([*args, "--check"]) == 1
        assert "out of date" in capsys.readouterr().err

    def test_check_fails_on_a_hand_edited_number(self, tmp_path):
        self.write_artifact(tmp_path / "results", make_artifact())
        readme = self.readme(tmp_path / "README.md")
        args = ["table", "--results-dir", str(tmp_path / "results"), "--readme", str(readme)]
        assert main([*args, "--write"]) == 0
        readme.write_text(
            readme.read_text(encoding="utf-8").replace("0.800", "0.100"), encoding="utf-8"
        )
        assert main([*args, "--check"]) == 1

    def test_check_does_not_write(self, tmp_path):
        self.write_artifact(tmp_path / "results", make_artifact())
        readme = self.readme(tmp_path / "README.md")
        before = readme.read_text(encoding="utf-8")
        args = ["table", "--results-dir", str(tmp_path / "results"), "--readme", str(readme)]
        assert main([*args, "--check"]) == 1
        assert readme.read_text(encoding="utf-8") == before

    def test_configs_fill_in_the_arms_that_have_not_run(self, tmp_path, capsys):
        out = tmp_path / "results"
        out.mkdir()
        assert main(["table", "--results-dir", str(out), "--configs", str(EXPERIMENTS)]) == 0
        printed = capsys.readouterr().out
        assert "TimesFM 2.5 + DoRA r16 *(not run)*" in printed
