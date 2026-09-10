"""Parameter-efficient fine-tuning and honest benchmarking of time series foundation models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsfm-peft")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
