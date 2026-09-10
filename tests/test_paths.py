"""Cache and results locations, including the env overrides the Dockerfile relies on."""

from pathlib import Path

import pytest

from tsfm_peft.paths import CACHE_ENV_VAR, RESULTS_ENV_VAR, cache_root, dataset_cache, results_root


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in (CACHE_ENV_VAR, RESULTS_ENV_VAR, "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)


class TestCacheRoot:
    def test_defaults_under_the_home_cache(self):
        assert cache_root() == Path.home() / ".cache" / "tsfm_peft"

    def test_honours_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_root() == tmp_path / "tsfm_peft"

    def test_env_override_wins_over_xdg(self, monkeypatch, tmp_path):
        # The Dockerfile sets TSFM_PEFT_CACHE to the mounted volume; it must take priority.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "explicit"))
        assert cache_root() == tmp_path / "explicit"

    def test_expands_a_tilde(self, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, "~/somewhere")
        assert cache_root() == Path.home() / "somewhere"


class TestDatasetCache:
    def test_creates_the_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        path = dataset_cache("etth1")
        assert path == tmp_path / "datasets" / "etth1"
        assert path.is_dir()

    def test_is_idempotent(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        assert dataset_cache("etth1") == dataset_cache("etth1")

    def test_without_a_name_returns_the_shared_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        assert dataset_cache() == tmp_path / "datasets"


class TestResultsRoot:
    def test_defaults_to_a_relative_results_directory(self):
        assert results_root() == Path("results")

    def test_honours_the_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(RESULTS_ENV_VAR, str(tmp_path / "out"))
        assert results_root() == tmp_path / "out"
