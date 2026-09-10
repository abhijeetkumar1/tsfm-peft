"""Experiment configs, the metrics artifact, the runner and the CLI."""

import json
from pathlib import Path

import pytest

from tsfm_peft.cli import main
from tsfm_peft.config import ExperimentConfig, ModelConfig, load_experiment
from tsfm_peft.experiment import run_experiment, summarise
from tsfm_peft.results import (
    SCHEMA_VERSION,
    artifact_path,
    load_artifact,
    load_artifacts,
    write_artifact,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO / "configs" / "experiments"
SMOKE_CONFIG = EXPERIMENT_DIR / "synthetic-seasonal-naive.yaml"

DATA_BLOCK = {
    "dataset": "synthetic",
    "protocol": {"horizon": 24, "context_length": 96, "n_test_windows": 2, "n_val_windows": 1},
    "scaler": "identity",
}


def experiment_dict(**overrides: object):
    payload = {
        "name": "unit-test",
        "data": dict(DATA_BLOCK),
        "model": {"name": "seasonal_naive"},
    }
    payload.update(overrides)
    return payload


class TestModelConfig:
    def test_rejects_an_unregistered_model(self):
        with pytest.raises(ValueError, match="unknown model"):
            ModelConfig(name="moirai")

    def test_rejects_unknown_options(self):
        with pytest.raises(ValueError, match="rank"):
            ModelConfig(name="seasonal_naive", options={"rank": 8})

    def test_validates_timesfm_options_without_torch(self):
        config = ModelConfig(name="timesfm_2p5", options={"batch_size": 4})
        assert config.resolved_options().batch_size == 4

    def test_builds_the_adapter(self):
        assert ModelConfig(name="seasonal_naive").build().name == "seasonal_naive"


class TestExperimentConfig:
    def test_defaults_are_explicit(self):
        config = ExperimentConfig.model_validate(experiment_dict())
        assert config.seed == 0
        assert config.deterministic is True
        assert config.notes is None

    @pytest.mark.parametrize("name", ["", "a/b", "with space", "../escape"])
    def test_rejects_names_that_are_not_filesystem_safe(self, name):
        with pytest.raises(ValueError, match="used as a filename"):
            ExperimentConfig.model_validate(experiment_dict(name=name))

    def test_rejects_unknown_top_level_keys(self):
        with pytest.raises(ValueError, match="epochs"):
            ExperimentConfig.model_validate(experiment_dict(epochs=3))

    def test_rejects_a_negative_seed(self):
        with pytest.raises(ValueError, match="seed"):
            ExperimentConfig.model_validate(experiment_dict(seed=-1))


class TestLoadExperiment:
    def test_expands_a_data_file_reference(self):
        config = load_experiment(SMOKE_CONFIG)
        assert config.data.dataset == "synthetic"
        assert config.data.protocol.horizon == 24

    def test_accepts_an_inline_data_block(self, tmp_path):
        path = tmp_path / "inline.yaml"
        path.write_text(json.dumps(experiment_dict()), encoding="utf-8")
        assert load_experiment(path).data.dataset == "synthetic"

    def test_resolves_the_reference_relative_to_the_config(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "d.yaml").write_text(json.dumps(DATA_BLOCK), encoding="utf-8")
        (tmp_path / "exp").mkdir()
        path = tmp_path / "exp" / "e.yaml"
        path.write_text(json.dumps(experiment_dict(data="../data/d.yaml")), encoding="utf-8")
        assert load_experiment(path).data.dataset == "synthetic"

    def test_names_a_missing_data_file(self, tmp_path):
        path = tmp_path / "e.yaml"
        path.write_text(json.dumps(experiment_dict(data="nope.yaml")), encoding="utf-8")
        with pytest.raises(ValueError, match="does not exist"):
            load_experiment(path)

    def test_rejects_a_non_mapping_file(self, tmp_path):
        path = tmp_path / "e.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping at the top level"):
            load_experiment(path)

    def test_every_shipped_config_loads(self):
        configs = [load_experiment(p) for p in sorted(EXPERIMENT_DIR.glob("*.yaml"))]
        assert configs
        # The name is the artifact filename, so a collision would silently overwrite a run.
        names = [c.name for c in configs]
        assert len(set(names)) == len(names)

    def test_shipped_config_names_match_their_filenames(self):
        for path in sorted(EXPERIMENT_DIR.glob("*.yaml")):
            assert load_experiment(path).name == path.stem


class TestArtifact:
    def test_path_is_named_after_the_experiment(self, tmp_path):
        assert artifact_path("etth1-lora", tmp_path) == tmp_path / "etth1-lora.json"

    def test_round_trips(self, tmp_path):
        artifact = {"schema_version": SCHEMA_VERSION, "experiment": {"name": "x"}}
        path = write_artifact(artifact, tmp_path / "x.json")
        assert load_artifact(path) == artifact
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_rejects_a_future_schema_version(self, tmp_path):
        path = write_artifact({"schema_version": SCHEMA_VERSION + 1}, tmp_path / "x.json")
        with pytest.raises(ValueError, match="artifact schema version"):
            load_artifact(path)

    def test_missing_results_directory_is_not_an_error(self, tmp_path):
        assert load_artifacts(tmp_path / "absent") == []

    def test_loads_artifacts_sorted_by_experiment_name(self, tmp_path):
        for name in ("zeta", "alpha"):
            write_artifact(
                {"schema_version": SCHEMA_VERSION, "experiment": {"name": name}},
                tmp_path / f"{name}.json",
            )
        assert [a["experiment"]["name"] for a in load_artifacts(tmp_path)] == ["alpha", "zeta"]


class TestRunExperiment:
    def test_runs_end_to_end_and_writes_an_artifact(self, tmp_path):
        outcome = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path)
        assert outcome.path == tmp_path / "synthetic-seasonal-naive.json"
        artifact = load_artifact(outcome.path)
        assert set(artifact) == {
            "schema_version",
            "created_at",
            "experiment",
            "config",
            "seed",
            "data",
            "model",
            "results",
            "resources",
            "environment",
        }

    def test_artifact_carries_the_reproduction_provenance(self, tmp_path):
        artifact = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path).artifact
        assert artifact["seed"]["seed"] == 0
        assert artifact["config"]["data"]["protocol"]["horizon"] == 24
        assert artifact["environment"]["packages"]["numpy"]
        assert "platform" in artifact["environment"]["hardware"]
        assert artifact["data"]["dataset"]["license"]

    def test_artifact_reports_cost_alongside_accuracy(self, tmp_path):
        artifact = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path).artifact
        assert artifact["results"]["mase"] > 0
        assert set(artifact["model"]) >= {"total_parameters", "trainable_parameters"}
        assert {p["label"] for p in artifact["resources"]["phases"]} == {
            "load_data",
            "load_model",
            "predict_test",
        }
        assert artifact["resources"]["total_seconds"] > 0

    def test_two_runs_of_a_config_agree(self, tmp_path):
        # The determinism claim. Timestamps and timings are expected to differ; nothing else.
        volatile = ("created_at", "resources")
        first = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path).artifact
        second = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path).artifact
        assert {k: v for k, v in first.items() if k not in volatile} == {
            k: v for k, v in second.items() if k not in volatile
        }

    def test_no_write_suppresses_the_artifact(self, tmp_path):
        outcome = run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path, write=False)
        assert outcome.path is None
        assert not list(tmp_path.glob("*.json"))

    def test_summary_reports_accuracy_and_cost(self, tmp_path):
        summary = summarise(run_experiment(load_experiment(SMOKE_CONFIG), results_dir=tmp_path))
        for field in ("MASE", "sMAPE", "WQL", "trainable", "wall clock", "seed", "artifact"):
            assert field in summary


class TestCli:
    def test_check_validates_without_loading_data(self, capsys):
        assert main(["run", "--check", str(SMOKE_CONFIG)]) == 0
        assert "ok" in capsys.readouterr().out

    def test_check_reports_a_bad_config(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(json.dumps(experiment_dict(model={"name": "nope"})), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown model"):
            main(["run", "--check", str(path)])

    def test_run_writes_to_the_requested_directory(self, tmp_path, capsys):
        assert main(["run", str(SMOKE_CONFIG), "--results-dir", str(tmp_path)]) == 0
        assert (tmp_path / "synthetic-seasonal-naive.json").is_file()
        assert "MASE" in capsys.readouterr().out

    def test_list_names_datasets_and_models(self, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "etth1" in out
        assert "timesfm_2p5" in out
        assert "generated fixture" in out

    def test_bare_invocation_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage" in capsys.readouterr().out
