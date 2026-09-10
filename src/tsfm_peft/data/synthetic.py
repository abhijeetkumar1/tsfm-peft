"""A deterministic synthetic dataset, generated rather than downloaded.

This exists so the full pipeline -- windowing, scaling, prediction, metrics, artifact
writing -- can be exercised end to end on CPU with no network and no model weights, which
is what lets CI prove the plumbing works without a GPU.

It is a smoke fixture, not a benchmark. The series are a trend plus one clean sinusoid plus
light Gaussian noise, so they are close to trivially forecastable and any accuracy number
computed on them says nothing about a model. :func:`load_synthetic` is registered under
``"synthetic"`` so it can be named in a config like any other dataset, but results on it
must never appear in the README table.
"""

from __future__ import annotations

import numpy as np

from tsfm_peft.data.dataset import TimeSeriesDataset

#: Shape of the dataset registered as ``"synthetic"``.
DEFAULT_N_SERIES = 3
DEFAULT_LENGTH = 600
DEFAULT_SEASONALITY = 24
DEFAULT_SEED = 0

SYNTHETIC_LICENSE = "CC0-1.0"
SYNTHETIC_SOURCE = "synthetic://tsfm_peft.data.synthetic"


def make_synthetic_dataset(
    n_series: int = DEFAULT_N_SERIES,
    length: int = DEFAULT_LENGTH,
    seasonality: int = DEFAULT_SEASONALITY,
    freq: str = "h",
    seed: int = DEFAULT_SEED,
    name: str = "synthetic",
) -> TimeSeriesDataset:
    """Generate a trend + seasonal + noise dataset.

    The generator is a pure function of its arguments: one ``default_rng`` is drawn from
    once per series, in order, so the same arguments always produce byte-identical series.

    Args:
        n_series: Number of series.
        length: Observations per series.
        seasonality: Period of the sinusoid, and the dataset's declared MASE lag.
        freq: Frequency string to declare.
        seed: Seed for the noise.
        name: Dataset name.

    Returns:
        The generated :class:`~tsfm_peft.data.dataset.TimeSeriesDataset`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    arrays = []
    for i in range(n_series):
        level = 10.0 * (i + 1)
        values = (
            level
            + 0.01 * t
            + 3.0 * np.sin(2.0 * np.pi * t / seasonality)
            + rng.normal(0.0, 0.1, size=length)
        )
        arrays.append((f"s{i}", values))
    return TimeSeriesDataset.from_arrays(
        name,
        arrays,
        freq=freq,
        seasonality=seasonality,
        license=SYNTHETIC_LICENSE,
        source_url=SYNTHETIC_SOURCE,
        description=(
            "Deterministic trend + seasonal + noise series; a smoke fixture, not a benchmark."
        ),
    )


def load_synthetic(*, force_download: bool = False) -> TimeSeriesDataset:
    """Return the canonical synthetic dataset.

    Args:
        force_download: Accepted for registry-loader signature compatibility and ignored;
            nothing is downloaded or cached, the data is generated on every call.

    Returns:
        The dataset.
    """
    del force_download
    return make_synthetic_dataset()
