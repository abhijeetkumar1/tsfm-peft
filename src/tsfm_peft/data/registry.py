"""The dataset registry: the single place a new dataset is wired in.

A dataset is described statically (frequency, seasonality, license, source) so that
documentation and the README datasets table can be generated without downloading anything,
and lazily loaded through its ``loader``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.loaders import (
    ETTH1_LICENSE,
    ETTH1_URL,
    NN5_LICENSE,
    NN5_URL,
    load_etth1,
    load_nn5_daily,
)
from tsfm_peft.data.synthetic import (
    DEFAULT_SEASONALITY as SYNTHETIC_SEASONALITY,
)
from tsfm_peft.data.synthetic import (
    SYNTHETIC_LICENSE,
    SYNTHETIC_SOURCE,
    load_synthetic,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a registered dataset.

    Attributes:
        name: Registry key, used in config files.
        loader: Callable returning the loaded dataset.
        freq: Frequency string.
        seasonality: MASE seasonal lag.
        license: Source data license.
        source_url: Where the raw file comes from.
        description: One-line description for the README datasets table.
    """

    name: str
    loader: Callable[..., TimeSeriesDataset]
    freq: str
    seasonality: int
    license: str
    source_url: str
    description: str

    @property
    def is_generated(self) -> bool:
        """Whether the data is generated locally rather than downloaded.

        The README datasets table and the results table both skip generated fixtures:
        they have no upstream to attribute and no number worth publishing.
        """
        return self.source_url.startswith("synthetic://")

    def to_dict(self) -> dict[str, Any]:
        """Return the static fields as a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "freq": self.freq,
            "seasonality": self.seasonality,
            "license": self.license,
            "source_url": self.source_url,
            "description": self.description,
            "is_generated": self.is_generated,
        }


DATASETS: dict[str, DatasetSpec] = {
    # A generated smoke fixture, registered so the CPU end-to-end path is driven by the
    # same code as a real run. Never publish a number computed on it.
    "synthetic": DatasetSpec(
        name="synthetic",
        loader=load_synthetic,
        freq="h",
        seasonality=SYNTHETIC_SEASONALITY,
        license=SYNTHETIC_LICENSE,
        source_url=SYNTHETIC_SOURCE,
        description="Generated trend + seasonal + noise, 3 series, 600 steps. Smoke fixture.",
    ),
    "etth1": DatasetSpec(
        name="etth1",
        loader=load_etth1,
        freq="h",
        seasonality=24,
        license=ETTH1_LICENSE,
        source_url=ETTH1_URL,
        description="Electricity Transformer Temperature, hourly, 7 channels, 17420 steps.",
    ),
    "nn5_daily": DatasetSpec(
        name="nn5_daily",
        loader=load_nn5_daily,
        freq="D",
        seasonality=7,
        license=NN5_LICENSE,
        source_url=NN5_URL,
        description="Daily ATM cash withdrawals, 111 series, 791 steps (Monash NN5).",
    ),
}


def available_datasets() -> tuple[str, ...]:
    """Return the registered dataset names, sorted."""
    return tuple(sorted(DATASETS))


def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name``.

    Args:
        name: Registry key.

    Returns:
        The spec.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(available_datasets())}"
        ) from None


def load_dataset(name: str, *, force_download: bool = False) -> TimeSeriesDataset:
    """Load a registered dataset, downloading and caching it on first use.

    Args:
        name: Registry key.
        force_download: Re-download even if a valid cached copy exists.

    Returns:
        The loaded dataset.
    """
    spec = get_spec(name)
    dataset = spec.loader(force_download=force_download)
    # The spec is what configs and docs read; a drift between it and the loader would make
    # the documented seasonality wrong, which would quietly change every MASE.
    if (dataset.freq, dataset.seasonality) != (spec.freq, spec.seasonality):
        raise AssertionError(
            f"{name}: loader returned freq/seasonality "
            f"{(dataset.freq, dataset.seasonality)} but the registry declares "
            f"{(spec.freq, spec.seasonality)}"
        )
    return dataset
