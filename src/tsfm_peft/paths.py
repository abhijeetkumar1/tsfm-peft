"""Filesystem locations for cached datasets, model weights and run artifacts.

Nothing here writes into the repository. Datasets and weights go to a user cache directory
so that the repo stays free of data (and of the licensing questions that come with
redistributing it), and so that a container can mount a single volume.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Overrides the cache root. Set in the Dockerfile so the image can mount one volume.
CACHE_ENV_VAR = "TSFM_PEFT_CACHE"

#: Overrides where run artifacts are written.
RESULTS_ENV_VAR = "TSFM_PEFT_RESULTS"


def cache_root() -> Path:
    """Return the cache root, honouring ``TSFM_PEFT_CACHE`` then ``XDG_CACHE_HOME``."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "tsfm_peft"


def dataset_cache(name: str = "") -> Path:
    """Return the cache directory for downloaded raw dataset files.

    Args:
        name: Optional dataset name to nest under.

    Returns:
        The directory, created if it does not exist.
    """
    path = cache_root() / "datasets"
    if name:
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_root() -> Path:
    """Return the directory metrics artifacts are written to."""
    override = os.environ.get(RESULTS_ENV_VAR)
    return Path(override).expanduser() if override else Path("results")
