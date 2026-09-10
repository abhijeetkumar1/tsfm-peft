"""In-memory representation of a univariate forecasting dataset.

Time series foundation models of the TimesFM family are univariate: they consume one
context vector and emit one forecast. A multivariate table such as ETTh1 is therefore
represented here as several independent series, one per channel, rather than as a single
multivariate array. Cross-channel information is not used by any v0.1 arm, so making that
explicit in the data model keeps the evaluation honest.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Series:
    """A single univariate series.

    Attributes:
        series_id: Identifier, unique within its dataset. Appears verbatim in the
            per-series breakdown of the metrics artifact.
        values: 1-D observations, oldest first, with no gaps.
        start: ISO-8601 timestamp of the first observation, if the source provides one.
            Carried for provenance and plotting; nothing in the evaluation path uses it.
    """

    series_id: str
    values: Array
    start: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the observation array."""
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"series {self.series_id!r}: values must be 1-D, got {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"series {self.series_id!r}: values is empty")
        if not np.isfinite(arr).all():
            raise ValueError(
                f"series {self.series_id!r}: values contain NaN or inf. Datasets are expected "
                "to be gap-free; impute or drop upstream in the loader, not here."
            )
        arr.flags.writeable = False
        object.__setattr__(self, "values", arr)

    def __len__(self) -> int:
        """Number of observations."""
        return int(self.values.size)


@dataclass(frozen=True, eq=False)
class TimeSeriesDataset:
    """A named collection of univariate series sharing a frequency and seasonality.

    Attributes:
        name: Registry key, e.g. ``"etth1"``.
        freq: Pandas-style frequency string, e.g. ``"h"`` or ``"D"``.
        seasonality: Dominant seasonal period in steps, used as the MASE lag (24 for
            hourly data, 7 for daily). Use 1 for series with no usable seasonality.
        series: The series, in a stable order.
        license: SPDX-ish identifier or short name of the source data license.
        source_url: Where the raw data was fetched from.
        description: One-line human description for the README datasets table.
    """

    name: str
    freq: str
    seasonality: int
    series: tuple[Series, ...]
    license: str
    source_url: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate seasonality, non-emptiness and identifier uniqueness."""
        if self.seasonality < 1:
            raise ValueError(f"{self.name}: seasonality must be >= 1, got {self.seasonality}")
        if not self.series:
            raise ValueError(f"{self.name}: dataset has no series")
        ids = [s.series_id for s in self.series]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"{self.name}: duplicate series ids {duplicates}")
        object.__setattr__(self, "series", tuple(self.series))

    def __len__(self) -> int:
        """Number of series."""
        return len(self.series)

    def __iter__(self) -> Iterator[Series]:
        """Iterate over series in registry order."""
        return iter(self.series)

    def __getitem__(self, series_id: str) -> Series:
        """Look up a series by identifier."""
        for s in self.series:
            if s.series_id == series_id:
                return s
        raise KeyError(f"{self.name}: no series {series_id!r}")

    @property
    def series_ids(self) -> tuple[str, ...]:
        """Identifiers in registry order."""
        return tuple(s.series_id for s in self.series)

    @property
    def lengths(self) -> tuple[int, ...]:
        """Length of each series, in registry order."""
        return tuple(len(s) for s in self.series)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance record for the metrics artifact."""
        lengths = self.lengths
        return {
            "name": self.name,
            "freq": self.freq,
            "seasonality": self.seasonality,
            "n_series": len(self.series),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_observations": int(sum(lengths)),
            "license": self.license,
            "source_url": self.source_url,
            "description": self.description,
        }

    @classmethod
    def from_arrays(
        cls,
        name: str,
        arrays: Sequence[tuple[str, Any]],
        *,
        freq: str,
        seasonality: int,
        license: str = "unknown",
        source_url: str = "",
        description: str = "",
        starts: Sequence[str | None] | None = None,
    ) -> TimeSeriesDataset:
        """Build a dataset from ``(series_id, values)`` pairs.

        Args:
            name: Dataset name.
            arrays: ``(series_id, values)`` pairs in the desired order.
            freq: Frequency string.
            seasonality: MASE seasonal lag.
            license: Source data license.
            source_url: Source URL.
            description: One-line description.
            starts: Optional start timestamps, aligned with ``arrays``.

        Returns:
            The constructed dataset.
        """
        if starts is not None and len(starts) != len(arrays):
            raise ValueError(f"starts has {len(starts)} entries but arrays has {len(arrays)}")
        series = tuple(
            Series(
                series_id=sid,
                values=np.asarray(values, dtype=np.float64),
                start=None if starts is None else starts[i],
            )
            for i, (sid, values) in enumerate(arrays)
        )
        return cls(
            name=name,
            freq=freq,
            seasonality=seasonality,
            series=series,
            license=license,
            source_url=source_url,
            description=description,
        )
