"""Seeding, environment capture and resource measurement.

The claim this repo makes is that a run reproduces on the same machine, and that the
accuracy figures come with their cost. Neither claim is checkable unless the run records
what it ran on, so every metrics artifact carries the seed, the resolved package versions,
the hardware and the wall-clock and memory a run consumed.

Determinism has a cost worth being explicit about: ``deterministic=True`` disables cuDNN's
autotuner and picks deterministic kernels, which is typically a few percent slower. Since a
wall-clock number is one of the things being reported, the flag is recorded in the artifact
so a timing is never compared against one taken under different settings.

Determinism also has a limit worth being explicit about. It is requested with
``warn_only=True``, so an op with no deterministic implementation warns and falls back
instead of failing the run -- which is the right trade (a benchmark that refuses to run
measures nothing) but means "deterministic" is not the same as "bit-exact". The fallback is
observable only as a warning, so :func:`watch_nondeterminism` collects those warnings and the
run records them in its artifact next to the seed.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

#: Packages whose versions materially affect results, recorded in every artifact.
TRACKED_PACKAGES: tuple[str, ...] = (
    "tsfm-peft",
    "numpy",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "pydantic",
)

#: Required before cuBLAS can be deterministic; must be set before the first CUDA call.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

#: Set by ``scripts/run_all.sh`` to the number of arms running at once on this host. A
#: timing taken while other arms competed for CPU and disk is not the same measurement as one
#: taken with the machine to itself, and nothing else in the artifact would show the
#: difference.
CONCURRENCY_ENV_VAR = "TSFM_PEFT_CONCURRENCY"

#: Lower-cased fragments torch uses when it falls back to a non-deterministic kernel under
#: ``warn_only=True``. Matched against warning text rather than against a torch API, because
#: torch exposes no way to ask after the fact whether a fallback happened.
NONDETERMINISM_WARNING_MARKERS: tuple[str, ...] = (
    "non-deterministic",
    "nondeterministic",
    "does not have a deterministic implementation",
)


def _torch() -> Any | None:
    """Return the ``torch`` module if the optional model dependencies are installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG the pipeline touches and optionally force deterministic kernels.

    Args:
        seed: The seed. Recorded in the artifact; changing it is a different experiment.
        deterministic: Select deterministic algorithms and disable cuDNN autotuning.
            ``warn_only`` is used so an op with no deterministic implementation degrades to
            a warning rather than aborting a long run -- the warning is the signal that a
            specific op, not the whole run, is non-reproducible.

    Returns:
        A JSON-serialisable record of what was seeded, for the metrics artifact.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}")
    random.seed(seed)
    # NPY002 prefers a Generator, and this repo's own code uses one. The legacy global RNG
    # is seeded anyway because transformers, peft and datasets reach for np.random.* module
    # functions internally; leaving it unseeded would leave part of a run unreproducible.
    np.random.seed(seed)  # noqa: NPY002

    record: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_seeded": False,
    }

    torch = _torch()
    if torch is None:
        return record

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    record["torch_seeded"] = True
    if deterministic:
        # Setting this after the first CUDA call has no effect, which is why it is set here
        # rather than left to the caller's shell.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    record["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return record


@contextmanager
def watch_nondeterminism() -> Iterator[list[str]]:
    """Collect torch's warnings that a kernel with no deterministic implementation was used.

    Because :func:`set_seed` asks for deterministic algorithms with ``warn_only=True``, such
    an op -- memory-efficient attention's backward pass, on any GPU run that trains -- warns
    and carries on. That warning is the only evidence the run is not bit-reproducible, and it
    goes to stderr, where a log nobody kept is the last place it existed. Collecting it lets
    the artifact carry it instead, so the results table can say which rows reproduce exactly
    and which reproduce only to within floating-point accumulation order.

    Warnings are still displayed exactly as they would have been; this only observes them.

    Yields:
        The distinct warning messages seen, in the order they were emitted. The list fills as
        the block runs, so read it after the block exits.
    """
    seen: list[str] = []
    previous = warnings.showwarning

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        """Record a determinism fallback, then show the warning the way it would have been."""
        text = str(message).strip()
        lowered = text.lower()
        if any(marker in lowered for marker in NONDETERMINISM_WARNING_MARKERS) and text not in seen:
            seen.append(text)
        previous(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    try:
        yield seen
    finally:
        warnings.showwarning = previous


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Return installed versions for ``names``, ``None`` for anything not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def git_revision() -> dict[str, Any]:
    """Return the current commit and whether the tree is dirty.

    A dirty tree means the artifact cannot be tied to a commit, so the flag is recorded
    rather than the run being blocked -- but a README number should come from a clean one.
    """

    def run(*args: str) -> str | None:
        """Run a read-only git command, returning ``None`` if it fails."""
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git installed
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def hardware_info() -> dict[str, Any]:
    """Return a description of the machine, including any visible CUDA devices."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "devices": [],
    }
    torch = _torch()
    if torch is None:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:  # pragma: no cover - requires a GPU
        info["cuda_version"] = torch.version.cuda
        info["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
                "capability": ".".join(str(c) for c in torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def scheduling_info() -> dict[str, Any]:
    """Return how this run was scheduled: arms sharing the host, and the devices it can see.

    ``concurrent_runs`` is ``None`` when nothing said, which is deliberately not the same as
    ``1``: a run that did not set the variable may still have shared the machine, and
    recording silence as "had the host to itself" would invent the fact the field exists to
    carry. A value that does not parse as a positive integer is treated the same way --
    unknown rather than fatal, because this is assembled after the run has finished and a
    typo in an environment variable should not destroy a completed arm.

    Returns:
        A JSON-serialisable mapping for the artifact's environment block.
    """
    raw = os.environ.get(CONCURRENCY_ENV_VAR)
    concurrent: int | None = None
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        concurrent = parsed if parsed > 0 else None
    return {
        "concurrent_runs": concurrent,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def collect_environment() -> dict[str, Any]:
    """Return the full provenance block written into every metrics artifact."""
    return {
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "hardware": hardware_info(),
        "scheduling": scheduling_info(),
        "git": git_revision(),
    }


def peak_host_memory_bytes() -> int | None:
    """Return peak resident set size for this process, in bytes, if the OS reports it.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS -- an easy factor-of-1024 error
    to publish. It is also a high-water mark for the whole process, so it is reported as
    "peak process RSS", not as the memory a single phase used.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


@dataclass
class ResourceUsage:
    """Wall-clock and memory consumed by one phase of a run.

    Attributes:
        label: Which phase this measures, e.g. ``"predict"`` or ``"train"``.
        seconds: Wall-clock duration. Monotonic, so it is unaffected by clock adjustments.
        peak_gpu_bytes: Peak CUDA memory allocated during the phase, or ``None`` on CPU.
        peak_host_rss_bytes: Peak process RSS observed at the end of the phase.
        device: The device the phase ran on, for context on the memory figures.
    """

    label: str
    seconds: float = 0.0
    peak_gpu_bytes: int | None = None
    peak_host_rss_bytes: int | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "label": self.label,
            "seconds": self.seconds,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "device": self.device,
        }


@contextmanager
def track_resources(label: str, device: str | None = None) -> Iterator[ResourceUsage]:
    """Measure wall-clock and peak memory for the enclosed block.

    CUDA is synchronised before both readings; without that the timer stops while kernels
    are still queued and the reported duration is meaninglessly short.

    Args:
        label: Phase name recorded in the artifact.
        device: Device string. CUDA statistics are only collected when it names a CUDA
            device, since ``max_memory_allocated`` is per-device state.

    Yields:
        The :class:`ResourceUsage`, populated when the block exits.
    """
    usage = ResourceUsage(label=label, device=device)
    torch = _torch()
    wants_cuda = bool(device) and str(device).startswith("cuda")
    on_cuda = bool(torch and wants_cuda and torch.cuda.is_available())
    if on_cuda:  # pragma: no cover - requires a GPU
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    try:
        yield usage
    finally:
        if on_cuda:  # pragma: no cover - requires a GPU
            torch.cuda.synchronize(device)
            usage.peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
        usage.seconds = time.perf_counter() - start
        usage.peak_host_rss_bytes = peak_host_memory_bytes()


@dataclass
class ResourceLog:
    """Ordered collection of :class:`ResourceUsage` records for a run."""

    phases: list[ResourceUsage] = field(default_factory=list)

    @contextmanager
    def phase(self, label: str, device: str | None = None) -> Iterator[ResourceUsage]:
        """Time a phase and append it to the log."""
        with track_resources(label, device) as usage:
            yield usage
        self.phases.append(usage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary keyed by phase label."""
        return {
            "phases": [usage.to_dict() for usage in self.phases],
            "total_seconds": sum(usage.seconds for usage in self.phases),
            "peak_gpu_bytes": max(
                (u.peak_gpu_bytes for u in self.phases if u.peak_gpu_bytes is not None),
                default=None,
            ),
            "peak_host_rss_bytes": max(
                (u.peak_host_rss_bytes for u in self.phases if u.peak_host_rss_bytes is not None),
                default=None,
            ),
        }
