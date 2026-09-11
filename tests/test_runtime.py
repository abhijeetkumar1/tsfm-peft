"""Seeding, provenance capture and resource measurement."""

import random
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

from tsfm_peft.runtime import (
    DETERMINISTIC_SDP_BACKENDS,
    ResourceLog,
    ResourceUsage,
    collect_environment,
    git_revision,
    hardware_info,
    package_versions,
    peak_host_memory_bytes,
    scheduling_info,
    set_seed,
    track_resources,
    watch_nondeterminism,
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


class TestDeterministicAttention:
    # The fused attention kernels have no deterministic backward, and under warn_only=True
    # they are what a GPU fine-tuning run silently falls back to. Turning them off is the
    # difference between a run that reports non-determinism and one that does not have it.

    @pytest.fixture(autouse=True)
    def _restore_backends(self):
        """SDPA backend selection is process-global; put it back for the other tests."""
        torch = pytest.importorskip("torch")
        backends = torch.backends.cuda
        before = {
            name: getattr(backends, f"{name}_sdp_enabled")()
            for name, _ in DETERMINISTIC_SDP_BACKENDS
            if hasattr(backends, f"{name}_sdp_enabled")
        }
        yield
        for name, value in before.items():
            getattr(backends, f"enable_{name}_sdp")(value)

    def test_leaves_only_the_math_backend_enabled(self):
        record = set_seed(0, deterministic=True)
        assert record["attention_backends"]["math"] is True
        assert not any(
            enabled for name, enabled in record["attention_backends"].items() if name != "math"
        )

    def test_takes_effect_on_the_torch_globals_not_just_the_record(self):
        torch = pytest.importorskip("torch")
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        set_seed(0, deterministic=True)
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False

    def test_a_non_deterministic_run_is_left_alone(self):
        torch = pytest.importorskip("torch")
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        record = set_seed(0, deterministic=False)
        assert "attention_backends" not in record
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is True

    def test_attention_still_computes_with_the_fused_kernels_off(self):
        torch = pytest.importorskip("torch")
        set_seed(0, deterministic=True)
        q = torch.randn(2, 4, 16, 32)
        assert torch.nn.functional.scaled_dot_product_attention(q, q, q).shape == q.shape


class TestWatchNondeterminism:
    # The warning these look for is the only signal torch gives that `warn_only=True` let a
    # non-deterministic kernel through, so the artifact's claim rests on catching it.

    @contextmanager
    def warned(self, *messages: str):
        """Emit warnings the way torch's autograd engine does, without escaping the test.

        ``record=True`` swallows the display, which also keeps these fixtures out of
        pytest's warning summary; the watcher is entered inside it and wraps it.
        """
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter("always")
            with watch_nondeterminism() as seen:
                for message in messages:
                    warnings.warn(message, UserWarning, stacklevel=2)
                yield seen, shown

    def test_records_a_determinism_fallback(self):
        with self.warned(
            "Memory Efficient attention defaults to a non-deterministic algorithm."
        ) as (seen, _):
            pass
        assert len(seen) == 1
        assert "non-deterministic" in seen[0]

    def test_records_the_upsample_wording_too(self):
        message = "upsample_bilinear2d_backward does not have a deterministic implementation"
        with self.warned(message) as (seen, _):
            pass
        assert len(seen) == 1

    def test_ignores_unrelated_warnings(self):
        with self.warned("You are sending unauthenticated requests to the HF Hub.") as (seen, _):
            pass
        assert seen == []

    def test_deduplicates_repeats(self):
        with self.warned(*["attention used a non-deterministic algorithm"] * 3) as (seen, _):
            pass
        assert len(seen) == 1

    def test_is_empty_when_nothing_warns(self):
        with watch_nondeterminism() as seen:
            pass
        assert seen == []

    def test_restores_the_previous_handler(self):
        before = warnings.showwarning
        with watch_nondeterminism():
            assert warnings.showwarning is not before
        assert warnings.showwarning is before

    def test_still_shows_the_warning(self):
        # The watcher wraps whatever handler it found rather than replacing it, so a warning
        # it records must also reach the handler underneath -- here, the recorder.
        with self.warned("a non-deterministic algorithm was used") as (seen, shown):
            pass
        assert len(seen) == 1
        assert len(shown) == 1


class TestSchedulingInfo:
    # The point of the field is the cost columns: a timing taken while other arms competed
    # for the host is not the measurement a serial run produces.

    def test_reports_the_concurrency_that_was_set(self, monkeypatch):
        monkeypatch.setenv("TSFM_PEFT_CONCURRENCY", "4")
        assert scheduling_info()["concurrent_runs"] == 4

    def test_unset_is_unknown_not_one(self, monkeypatch):
        # Silence is not a claim that the run had the machine to itself.
        monkeypatch.delenv("TSFM_PEFT_CONCURRENCY", raising=False)
        assert scheduling_info()["concurrent_runs"] is None

    @pytest.mark.parametrize("value", ["", "two", "0", "-1", "2.5"])
    def test_unusable_values_are_unknown_rather_than_fatal(self, monkeypatch, value):
        # Assembled after the run has finished; a typo must not destroy a completed arm.
        monkeypatch.setenv("TSFM_PEFT_CONCURRENCY", value)
        assert scheduling_info()["concurrent_runs"] is None

    def test_records_the_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert scheduling_info()["visible_devices"] == "1"

    def test_is_in_the_environment_block(self, monkeypatch):
        monkeypatch.setenv("TSFM_PEFT_CONCURRENCY", "2")
        assert collect_environment()["scheduling"]["concurrent_runs"] == 2


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
