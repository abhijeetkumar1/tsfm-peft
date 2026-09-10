"""Seeding, provenance capture and resource measurement."""

import random
import sys

import numpy as np
import pytest

from tsfm_peft.runtime import (
    ResourceLog,
    ResourceUsage,
    collect_environment,
    git_revision,
    hardware_info,
    package_versions,
    peak_host_memory_bytes,
    set_seed,
    track_resources,
)


class TestSetSeed:
    # These deliberately exercise numpy's legacy global RNG (hence the NPY002 waivers):
    # verifying that it is seeded is the point, because third-party libraries use it.

    def test_makes_numpy_and_random_reproducible(self):
        set_seed(123)
        first = (np.random.rand(4).tolist(), [random.random() for _ in range(4)])  # noqa: NPY002
        set_seed(123)
        second = (np.random.rand(4).tolist(), [random.random() for _ in range(4)])  # noqa: NPY002
        assert first == second

    def test_different_seeds_differ(self):
        set_seed(1)
        first = np.random.rand(8).tolist()  # noqa: NPY002
        set_seed(2)
        second = np.random.rand(8).tolist()  # noqa: NPY002
        assert second != first

    def test_records_what_it_seeded(self):
        record = set_seed(0, deterministic=False)
        assert record["seed"] == 0
        assert record["deterministic"] is False
        assert "torch_seeded" in record

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            set_seed(-1)

    def test_record_is_json_serialisable(self):
        import json

        json.dumps(set_seed(3))


class TestProvenance:
    def test_reports_none_for_missing_packages(self):
        versions = package_versions(["tsfm-peft", "definitely-not-installed"])
        assert versions["tsfm-peft"]
        assert versions["definitely-not-installed"] is None

    def test_hardware_info_has_the_expected_keys(self):
        info = hardware_info()
        assert set(info) >= {"platform", "machine", "cpu_count", "cuda_available", "devices"}
        assert isinstance(info["cuda_available"], bool)

    def test_git_revision_reports_commit_and_dirtiness(self):
        revision = git_revision()
        assert set(revision) == {"commit", "dirty"}

    def test_environment_is_json_serialisable(self):
        import json

        payload = json.loads(json.dumps(collect_environment()))
        assert payload["python"] == sys.version.split()[0]
        assert "torch" in payload["packages"]

    @pytest.mark.skipif(sys.platform == "win32", reason="ru_maxrss is POSIX only")
    def test_peak_host_memory_is_plausible(self):
        # A Python process with numpy loaded is well over 1 MB and well under 100 GB; the
        # point of the bound is to catch the kilobytes/bytes unit mistake, not to be tight.
        peak = peak_host_memory_bytes()
        assert 1e6 < peak < 1e11


class TestResourceTracking:
    def test_times_the_enclosed_block(self):
        with track_resources("predict", device="cpu") as usage:
            sum(range(10_000))
        assert usage.seconds > 0
        assert usage.label == "predict"
        assert usage.device == "cpu"

    def test_records_usage_even_when_the_block_raises(self):
        with pytest.raises(RuntimeError), track_resources("boom") as usage:
            raise RuntimeError("boom")
        assert usage.seconds > 0

    def test_no_gpu_measurement_on_cpu(self):
        with track_resources("predict", device="cpu") as usage:
            pass
        assert usage.peak_gpu_bytes is None

    def test_log_accumulates_phases_in_order(self):
        log = ResourceLog()
        with log.phase("load"):
            pass
        with log.phase("predict"):
            pass
        summary = log.to_dict()
        assert [p["label"] for p in summary["phases"]] == ["load", "predict"]
        assert summary["total_seconds"] == pytest.approx(
            sum(p["seconds"] for p in summary["phases"])
        )

    def test_empty_log_reports_no_peaks(self):
        summary = ResourceLog().to_dict()
        assert summary["peak_gpu_bytes"] is None
        assert summary["total_seconds"] == 0

    def test_peak_gpu_is_the_max_across_phases(self):
        log = ResourceLog(
            phases=[
                ResourceUsage(label="a", peak_gpu_bytes=10),
                ResourceUsage(label="b", peak_gpu_bytes=30),
                ResourceUsage(label="c", peak_gpu_bytes=None),
            ]
        )
        assert log.to_dict()["peak_gpu_bytes"] == 30
