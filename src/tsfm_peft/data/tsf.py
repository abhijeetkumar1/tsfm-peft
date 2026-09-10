"""Reader for the Monash forecasting repository's ``.tsf`` format.

``.tsf`` is a small ARFF-like format: ``#`` comments, ``@``-prefixed metadata and attribute
declarations, then one line per series holding its attributes and a comma-separated value
list, all separated by colons. Missing values appear as ``?``.

Written by hand rather than pulled from a dependency: it is ~60 lines, it keeps the CPU test
environment free of pandas, and it lets missing values raise loudly instead of being filled
in by a default the evaluation would then silently depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class TsfRecord:
    """One series from a ``.tsf`` file.

    Attributes:
        attributes: Declared attribute values, keyed by attribute name, as raw strings.
        values: The observation vector.
    """

    attributes: dict[str, str]
    values: Array


@dataclass(frozen=True, eq=False)
class TsfFile:
    """A parsed ``.tsf`` file.

    Attributes:
        metadata: ``@``-declared metadata such as ``frequency`` and ``horizon``.
        attribute_names: Declared attribute names, in order.
        records: The series.
    """

    metadata: dict[str, str]
    attribute_names: tuple[str, ...]
    records: tuple[TsfRecord, ...]


def _parse_values(raw: str, line_number: int, allow_missing: bool) -> Array:
    """Parse a comma-separated value list, rejecting ``?`` unless missing data is allowed."""
    tokens = [t for t in raw.split(",") if t != ""]
    if not tokens:
        raise ValueError(f"line {line_number}: series has no values")
    if not allow_missing and "?" in tokens:
        raise ValueError(
            f"line {line_number}: series contains missing values (?). Pass "
            "allow_missing=True only if the caller imputes them explicitly; silent "
            "imputation would make the evaluation depend on an undocumented choice."
        )
    return np.array([np.nan if t == "?" else float(t) for t in tokens], dtype=np.float64)


def read_tsf(path: str | Path, *, allow_missing: bool = False) -> TsfFile:
    """Parse a ``.tsf`` file.

    Args:
        path: Path to the file.
        allow_missing: Permit ``?`` entries, which become ``nan``.

    Returns:
        The parsed :class:`TsfFile`.

    Raises:
        ValueError: On a malformed header, a data line before ``@data``, or an
            attribute-count mismatch.
    """
    metadata: dict[str, str] = {}
    attribute_names: list[str] = []
    records: list[TsfRecord] = []
    in_data = False

    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                if in_data:
                    raise ValueError(f"line {line_number}: '@' directive after @data")
                tag, _, rest = line[1:].partition(" ")
                tag = tag.lower()
                if tag == "data":
                    in_data = True
                elif tag == "attribute":
                    name = rest.split()[0]
                    attribute_names.append(name)
                else:
                    metadata[tag] = rest.strip()
                continue

            if not in_data:
                raise ValueError(f"line {line_number}: data line before @data")

            parts = line.split(":")
            if len(parts) != len(attribute_names) + 1:
                raise ValueError(
                    f"line {line_number}: expected {len(attribute_names) + 1} colon-separated "
                    f"fields ({len(attribute_names)} attributes plus values), got {len(parts)}"
                )
            records.append(
                TsfRecord(
                    attributes=dict(zip(attribute_names, parts[:-1], strict=True)),
                    values=_parse_values(parts[-1], line_number, allow_missing),
                )
            )

    if not in_data:
        raise ValueError(f"{path}: no @data section")
    if not records:
        raise ValueError(f"{path}: @data section is empty")
    return TsfFile(
        metadata=metadata,
        attribute_names=tuple(attribute_names),
        records=tuple(records),
    )


def normalise_timestamp(raw: str) -> str:
    """Convert a ``.tsf`` timestamp (``YYYY-MM-DD HH-MM-SS``) to ISO-8601."""
    date, _, time = raw.strip().partition(" ")
    return f"{date}T{time.replace('-', ':')}" if time else date
