"""Dataset loaders for v0.1: ETTh1 and NN5 Daily.

Both are downloaded on first use into the cache directory (see :mod:`tsfm_peft.paths`) and
pinned by SHA-256. No data is committed to this repository; see the datasets section of the
README for each source's license.

Adding a dataset means writing one function here that returns a
:class:`~tsfm_peft.data.dataset.TimeSeriesDataset` and adding one entry to
:mod:`tsfm_peft.data.registry` -- no other module needs to change.
"""

from __future__ import annotations

import csv

import numpy as np

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.data.download import cached_download, extract_member
from tsfm_peft.data.tsf import normalise_timestamp, read_tsf
from tsfm_peft.paths import dataset_cache

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
# CC BY-ND 4.0: attribution required, no derivatives distributed. This project downloads the
# file at runtime and never redistributes or ships a modified copy of it.
ETTH1_LICENSE = "CC-BY-ND-4.0"

NN5_URL = (
    "https://zenodo.org/records/4656117/files/"
    "nn5_daily_dataset_without_missing_values.zip?download=1"
)
NN5_SHA256 = "14d504384a97f7730a4b2595da9f35f77ee82b1974ffcf515eb17d607f7a07bb"
NN5_MEMBER = "nn5_daily_dataset_without_missing_values.tsf"
NN5_LICENSE = "CC-BY-4.0"


def load_etth1(*, force_download: bool = False) -> TimeSeriesDataset:
    """Load ETTh1: two years of hourly electricity transformer measurements.

    The file is a 7-column multivariate table (six load channels plus oil temperature).
    Each column becomes an independent univariate series, since no v0.1 arm uses
    cross-channel information. Seasonality is 24 (hourly data with a daily cycle).

    Args:
        force_download: Re-download even if a valid cached copy exists.

    Returns:
        A 7-series hourly dataset.
    """
    path = cached_download(
        ETTH1_URL,
        dataset_cache("etth1") / "ETTh1.csv",
        sha256=ETTH1_SHA256,
        force=force_download,
    )
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]

    if header[0].lower() != "date":
        raise ValueError(f"ETTh1: expected a leading 'date' column, got {header[0]!r}")
    channels = header[1:]
    table = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float64)
    if table.shape[1] != len(channels):
        raise ValueError(f"ETTh1: {len(channels)} columns declared but {table.shape[1]} found")

    start = normalise_timestamp(rows[0][0])
    return TimeSeriesDataset.from_arrays(
        "etth1",
        [(name, table[:, i]) for i, name in enumerate(channels)],
        freq="h",
        seasonality=24,
        license=ETTH1_LICENSE,
        source_url=ETTH1_URL,
        description=(
            "Electricity Transformer Temperature, hourly, 7 channels treated as "
            "independent univariate series."
        ),
        starts=[start] * len(channels),
    )


def load_nn5_daily(*, force_download: bool = False) -> TimeSeriesDataset:
    """Load NN5 Daily: daily cash withdrawals at 111 UK ATMs.

    The Monash "without missing values" variant is used, in which the original gaps were
    filled upstream by the repository authors with the median of the same weekday. That
    imputation is part of the published dataset, not something this project does; it is
    noted in the README limitations because imputed points are still scored.

    Seasonality is 7 (daily data with a weekly cycle).

    Args:
        force_download: Re-download even if a valid cached copy exists.

    Returns:
        A 111-series daily dataset.
    """
    cache = dataset_cache("nn5_daily")
    archive = cached_download(
        NN5_URL, cache / "nn5_daily.zip", sha256=NN5_SHA256, force=force_download
    )
    tsf_path = extract_member(archive, NN5_MEMBER, cache / NN5_MEMBER)
    parsed = read_tsf(tsf_path)

    arrays = []
    starts = []
    for record in parsed.records:
        arrays.append((record.attributes["series_name"], record.values))
        starts.append(normalise_timestamp(record.attributes["start_timestamp"]))

    return TimeSeriesDataset.from_arrays(
        "nn5_daily",
        arrays,
        freq="D",
        seasonality=7,
        license=NN5_LICENSE,
        source_url="https://zenodo.org/records/4656117",
        description="Daily cash withdrawals at 111 UK ATMs (NN5 competition, Monash).",
        starts=starts,
    )
