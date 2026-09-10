"""Shared synthetic fixtures. Nothing here touches the network or the dataset cache."""

import numpy as np
import pytest

from tsfm_peft.data.dataset import TimeSeriesDataset


def build_dataset(
    n_series: int = 2,
    length: int = 400,
    seasonality: int = 24,
    freq: str = "h",
    seed: int = 0,
    name: str = "synthetic",
) -> TimeSeriesDataset:
    """Build a deterministic seasonal + trend + noise dataset."""
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
        license="CC0-1.0",
        source_url="synthetic://",
        description="Deterministic synthetic series for tests.",
    )


@pytest.fixture
def dataset() -> TimeSeriesDataset:
    """A two-series hourly dataset of length 400."""
    return build_dataset()


@pytest.fixture
def make_dataset():
    """The :func:`build_dataset` factory, for tests needing a custom shape."""
    return build_dataset
