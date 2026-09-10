"""The forecasting-model interface every arm of the benchmark is evaluated through.

A model adapter's only job is to turn a batch of raw contexts into point and quantile
forecasts. Everything else -- windowing, scaling, metric computation, provenance -- lives
outside, so that adding Moirai, Chronos or Lag-Llama in a later version means writing one
subclass and one registry entry, not touching the evaluation path.

Two properties of the interface are deliberate:

* **Contexts are raw.** Adapters receive observations in the units the dataset ships in and
  must return forecasts in those same units. Every model in the TimesFM family normalises
  each context internally; asking the harness to normalise first would double-normalise and
  would silently change what zero-shot means. Adapters that genuinely need an external
  scaler say so with :attr:`ForecastModel.requires_external_scaling`.
* **Quantile levels belong to the model.** A model exposes the levels it was trained to
  emit rather than being asked for arbitrary ones, because interpolating quantiles a model
  never learned would report a number the model cannot actually produce. The evaluation
  driver checks the levels against the metric configuration instead of papering over a
  mismatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DatasetContext:
    """Dataset *metadata* handed to an adapter before evaluation.

    Deliberately carries no observations. An adapter that needs to know the sampling
    frequency or the seasonal period (the seasonal-naive baseline does; Moirai will) can
    read it here, while an adapter still cannot reach a value it is about to be scored on
    even by accident. Passing the :class:`~tsfm_peft.data.dataset.TimeSeriesDataset` itself
    would hand every adapter the test targets and make leakage a one-line mistake.

    Attributes:
        name: Dataset registry key.
        freq: Pandas-style frequency string.
        seasonality: Dominant seasonal period in steps.
    """

    name: str
    freq: str
    seasonality: int


@dataclass(frozen=True, eq=False)
class Forecast:
    """Point and quantile forecasts for a batch of windows, in the dataset's raw units.

    Attributes:
        point: ``(n_rows, horizon)`` point forecasts. For quantile models this is normally
            the median head rather than the mean, which matters: the median minimises MAE
            and hence MASE, the mean minimises MSE. Adapters document which one they return.
        quantiles: ``(n_rows, horizon, n_levels)`` quantile forecasts, last axis aligned
            with :attr:`levels`.
        levels: Strictly increasing quantile levels in ``(0, 1)``.
    """

    point: Array
    quantiles: Array
    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate shapes and finiteness, and freeze the arrays."""
        point = np.asarray(self.point, dtype=np.float64)
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        levels = tuple(float(q) for q in self.levels)
        if point.ndim != 2:
            raise ValueError(f"point must be 2-D (n_rows, horizon); got shape {point.shape}")
        expected = (*point.shape, len(levels))
        if quantiles.shape != expected:
            raise ValueError(f"quantiles must have shape {expected}; got {quantiles.shape}")
        if not levels:
            raise ValueError("levels must be non-empty")
        if not all(0.0 < q < 1.0 for q in levels):
            raise ValueError(f"levels must lie strictly in (0, 1); got {levels}")
        if any(b <= a for a, b in pairwise(levels)):
            raise ValueError(f"levels must be strictly increasing; got {levels}")
        for name, arr in (("point", point), ("quantiles", quantiles)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN or inf")
        point.flags.writeable = False
        quantiles.flags.writeable = False
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "levels", levels)

    @property
    def n_rows(self) -> int:
        """Number of forecast rows."""
        return int(self.point.shape[0])

    @property
    def horizon(self) -> int:
        """Forecast horizon in steps."""
        return int(self.point.shape[1])

    def quantile_crossing_rate(self) -> float:
        """Fraction of adjacent quantile pairs that are out of order.

        Quantile heads are trained independently, so nothing forces the 0.6 forecast to sit
        above the 0.5 one. Crossings are not corrected here -- sorting them away would
        improve the reported pinball loss without the model having got any better -- but the
        rate is recorded in the metrics artifact so a degenerate predictive distribution is
        visible rather than hidden inside an aggregate.

        Returns:
            The crossing fraction in ``[0, 1]``; ``0.0`` when there is only one level.
        """
        if len(self.levels) < 2:
            return 0.0
        return float((np.diff(self.quantiles, axis=-1) < 0).mean())

    @classmethod
    def concatenate(cls, parts: Sequence[Forecast]) -> Forecast:
        """Join per-batch forecasts, in order, into one.

        Args:
            parts: Forecasts sharing a horizon and quantile levels.

        Returns:
            The concatenated forecast.
        """
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of forecasts")
        head = parts[0]
        for part in parts[1:]:
            if part.levels != head.levels:
                raise ValueError(f"quantile levels differ: {head.levels} vs {part.levels}")
            if part.horizon != head.horizon:
                raise ValueError(f"horizons differ: {head.horizon} vs {part.horizon}")
        return cls(
            point=np.concatenate([p.point for p in parts], axis=0),
            quantiles=np.concatenate([p.quantiles for p in parts], axis=0),
            levels=head.levels,
        )


def as_context_batch(contexts: Any) -> Array:
    """Coerce a batch of contexts to a finite ``(n_rows, context_length)`` float array."""
    arr = np.asarray(contexts, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"contexts must be 2-D (n_rows, context_length); got {arr.shape}")
    if arr.size == 0:
        raise ValueError("contexts is empty")
    if not np.isfinite(arr).all():
        raise ValueError("contexts contain NaN or inf")
    return arr


class ForecastModel(ABC):
    """Base class for model adapters.

    Subclasses implement :meth:`_predict_batch` and the descriptive properties; batching,
    validation and result assembly are handled here so every adapter behaves identically
    from the evaluation driver's point of view.
    """

    #: Registry key, mirrored into the metrics artifact.
    name: str = "unnamed"

    @property
    @abstractmethod
    def quantile_levels(self) -> tuple[float, ...]:
        """Quantile levels this model emits, strictly increasing."""

    @property
    @abstractmethod
    def max_horizon(self) -> int | None:
        """Longest horizon forecastable in one call, or ``None`` if unbounded."""

    @property
    def requires_external_scaling(self) -> bool:
        """Whether the harness must scale contexts before handing them over.

        ``False`` for every model in v0.1: TimesFM normalises each context internally, and
        the seasonal-naive baseline is scale-equivariant by construction.
        """
        return False

    @property
    def batch_size(self) -> int:
        """Rows per forward pass. Recorded in the artifact because it can affect results."""
        return 32

    def bind_dataset(self, context: DatasetContext) -> None:
        """Optional hook: receive dataset metadata before the first :meth:`predict` call.

        Called once by the evaluation driver. The default does nothing. Implementations
        must not require it to have been called for :meth:`predict` to be correct unless
        they raise a clear error, since an adapter may also be driven directly.

        Args:
            context: Dataset metadata. Contains no observations by design.
        """
        return None

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the metrics artifact."""
        return {
            "name": self.name,
            "quantile_levels": list(self.quantile_levels),
            "max_horizon": self.max_horizon,
            "batch_size": self.batch_size,
            "requires_external_scaling": self.requires_external_scaling,
            **self.parameter_counts(),
        }

    def parameter_counts(self) -> dict[str, Any]:
        """Return total/trainable parameter counts. Zero for non-parametric baselines."""
        return {"total_parameters": 0, "trainable_parameters": 0, "trainable_pct": 0.0}

    @abstractmethod
    def _predict_batch(self, contexts: Array, horizon: int) -> Forecast:
        """Forecast one batch. ``contexts`` is ``(n_rows, context_length)`` in raw units."""

    def predict(self, contexts: Any, horizon: int) -> Forecast:
        """Forecast a batch of contexts, chunked into forward passes of :attr:`batch_size`.

        Args:
            contexts: ``(n_rows, context_length)`` raw observations, oldest first.
            horizon: Number of steps to forecast.

        Returns:
            The :class:`Forecast`, in the same row order as ``contexts``.

        Raises:
            ValueError: If ``horizon`` is not positive or exceeds :attr:`max_horizon`.
        """
        batch = as_context_batch(contexts)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        if self.max_horizon is not None and horizon > self.max_horizon:
            raise ValueError(
                f"{self.name}: horizon {horizon} exceeds the adapter's single-call maximum "
                f"{self.max_horizon}. Autoregressive rollout is not implemented in v0.1; "
                "shorten the horizon or add rollout to the adapter."
            )
        size = max(1, self.batch_size)
        parts = [
            self._check_batch(self._predict_batch(batch[i : i + size], horizon), horizon, size)
            for i in range(0, batch.shape[0], size)
        ]
        forecast = Forecast.concatenate(parts)
        if forecast.n_rows != batch.shape[0]:  # pragma: no cover - defensive
            raise AssertionError(
                f"{self.name}: returned {forecast.n_rows} rows for {batch.shape[0]} contexts"
            )
        return forecast

    def _check_batch(self, forecast: Forecast, horizon: int, size: int) -> Forecast:
        """Validate one batch's forecast against the requested horizon and model levels."""
        if forecast.horizon != horizon:
            raise ValueError(
                f"{self.name}: returned horizon {forecast.horizon}, expected {horizon}"
            )
        if forecast.levels != tuple(self.quantile_levels):
            raise ValueError(
                f"{self.name}: returned levels {forecast.levels}, expected "
                f"{tuple(self.quantile_levels)}"
            )
        if forecast.n_rows > size:  # pragma: no cover - defensive
            raise AssertionError(f"{self.name}: batch of {size} produced {forecast.n_rows} rows")
        return forecast
