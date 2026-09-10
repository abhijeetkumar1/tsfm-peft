"""Checksummed, cached downloads of raw dataset files.

Two properties matter for reproducibility:

* **Pinned content.** Every source is pinned by SHA-256. Both v0.1 datasets are served from
  mutable locations (a branch tip on GitHub, a Zenodo record), so without a checksum a
  silent upstream edit would change published numbers with nothing in the diff to show it.
  A mismatch raises instead of proceeding.
* **Atomic writes.** Downloads land in a temporary file in the same directory and are moved
  into place only after the checksum passes, so an interrupted download can never be picked
  up as a valid cache entry on the next run.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_CHUNK = 1 << 20
_USER_AGENT = "tsfm-peft/0.1 (+https://github.com/abhijeet/tsfm-peft)"


class ChecksumMismatchError(RuntimeError):
    """Raised when downloaded or cached bytes do not match the pinned SHA-256."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, destination: Path, timeout: float) -> None:
    """Stream ``url`` into ``destination`` atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    handle, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            open(tmp, "wb") as out,
        ):
            shutil.copyfileobj(response, out, _CHUNK)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def cached_download(
    url: str,
    destination: Path,
    *,
    sha256: str,
    timeout: float = 120.0,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``destination`` unless a matching cached copy already exists.

    Args:
        url: Source URL.
        destination: Where the file should end up.
        sha256: Expected hex digest of the file contents.
        timeout: Per-connection timeout in seconds.
        force: Re-download even if a valid cached copy exists.

    Returns:
        ``destination``.

    Raises:
        ChecksumMismatchError: If the downloaded bytes do not match ``sha256``. A cached
            file that fails the check is treated as corrupt and re-downloaded once; a fresh
            download that fails means the upstream content changed, which is a
            reproducibility problem the caller must resolve deliberately.
    """
    if destination.exists() and not force:
        if sha256_of(destination) == sha256:
            return destination
        destination.unlink()

    _fetch(url, destination, timeout)
    actual = sha256_of(destination)
    if actual != sha256:
        destination.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{url} does not match its pinned checksum.\n"
            f"  expected {sha256}\n  actual   {actual}\n"
            "The upstream file has changed. Verify the new contents before updating the "
            "pin, and re-run every arm afterwards: published numbers are only comparable "
            "across identical inputs."
        )
    return destination


def extract_member(archive: Path, member: str, destination: Path) -> Path:
    """Extract one named member from a zip archive if it is not already present.

    Args:
        archive: Path to the ``.zip`` file.
        member: Name of the member to extract.
        destination: File path to write the member to.

    Returns:
        ``destination``.
    """
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Extract by explicit member name and write to an explicit path: never
    # zf.extractall, which honours whatever paths the archive claims.
    with zipfile.ZipFile(archive) as zf, zf.open(member) as src:
        handle, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            tmp.replace(destination)
        finally:
            tmp.unlink(missing_ok=True)
    return destination
