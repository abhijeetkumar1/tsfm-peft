"""Shared synthetic fixtures. Nothing here touches the network or the dataset cache."""

import pytest

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.synthetic import make_synthetic_dataset


def build_dataset(
    n_series: int = 2,
    length: int = 400,
    seasonality: int = 24,
    freq: str = "h",
    seed: int = 0,
    name: str = "synthetic",
) -> TimeSeriesDataset:
    """Build a deterministic seasonal + trend + noise dataset.

    Delegates to the library generator so the fixture and the registered ``synthetic``
    dataset cannot drift apart.
    """
    return make_synthetic_dataset(
        n_series=n_series,
        length=length,
        seasonality=seasonality,
        freq=freq,
        seed=seed,
        name=name,
    )


@pytest.fixture
def dataset() -> TimeSeriesDataset:
    """A two-series hourly dataset of length 400."""
    return build_dataset()


@pytest.fixture
def make_dataset():
    """The :func:`build_dataset` factory, for tests needing a custom shape."""
    return build_dataset
