"""Rolling-origin backtesting: split a dataset into a fit region and evaluation windows.

The protocol, for a series of length ``T`` with horizon ``H``, ``W`` test windows and
stride ``s``::

    index 0                                fit_end     first test origin           T
          |<-------- fit region -------->|<- val ->|<---- test windows ---->|
                                                    o_0    o_1    o_2  ...  o_{W-1}

* The last test origin is pinned to ``T - H`` so the final window always ends at the last
  observation; earlier origins step back by ``s``. With the default ``s = H`` the windows
  tile the tail of the series without overlapping, so every test point is predicted exactly
  once and the aggregate metrics are not double-counting any observation.
* ``fit_end`` is the first validation origin (or the first test origin when no validation
  windows are requested). **Everything a run is allowed to learn from lives in
  ``values[:fit_end]``** -- model weights, scaler statistics and the MASE denominator.
* Contexts are read from the raw series, so the context of a later test window may include
  observations from the validation region or from an earlier test window's target span.
  That is not leakage: at forecast origin ``o`` those points are in the past and would be
  observed in production. Leakage would be using anything at index ``>= o`` to produce the
  forecast for ``o``, or fitting anything at all on data at or beyond ``fit_end``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_peft.data.dataset import TimeSeriesDataset
from tsfm_peft.metrics import seasonal_naive_scale

Array = NDArray[np.float64]


class BacktestProtocol(BaseModel):
    """Evaluation protocol parameters, validated straight from YAML.

    Attributes:
        horizon: Forecast horizon in steps.
        context_length: Number of past observations fed to the model at each origin.
        n_test_windows: Number of rolling-origin evaluation windows per series.
        n_val_windows: Number of windows carved out immediately before the test region,
            used for early stopping and checkpoint selection during fine-tuning. These sit
            outside the fit region, so tuning on them does not touch test data -- but it
            does shorten training. Zero disables validation entirely.
        stride: Step between consecutive origins. Defaults to ``horizon``, which gives
            non-overlapping windows. A smaller stride yields more windows from the same
            span at the cost of correlated (partially overlapping) errors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(gt=0)
    context_length: int = Field(gt=0)
    n_test_windows: int = Field(gt=0)
    n_val_windows: int = Field(default=0, ge=0)
    stride: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _warn_on_overlap(self) -> BacktestProtocol:
        """Reject strides that would make windows overlap silently past a full horizon."""
        if self.stride is not None and self.stride > self.horizon:
            raise ValueError(
                f"stride {self.stride} exceeds horizon {self.horizon}, which would leave "
                "gaps of unevaluated observations between windows"
            )
        return self

    @property
    def effective_stride(self) -> int:
        """The stride actually used: ``stride`` if set, otherwise ``horizon``."""
        return self.horizon if self.stride is None else self.stride

    @property
    def min_series_length(self) -> int:
        """Shortest series this protocol can be applied to.

        A series needs a fit region of at least ``context_length`` observations (so every
        origin has a full context and the model has something to train on) plus the span
        consumed by the validation and test windows.
        """
        return self.context_length + self.eval_span

    @property
    def eval_span(self) -> int:
        """Number of trailing observations reserved for validation and test windows."""
        stride = self.effective_stride
        span = self.horizon + (self.n_test_windows - 1) * stride
        if self.n_val_windows:
            span += self.horizon + (self.n_val_windows - 1) * stride
        return span


@dataclass(frozen=True, eq=False)
class SplitPlan:
    """Index boundaries for one series under a protocol. Pure arithmetic, no data.

    Attributes:
        series_id: The series these indices belong to.
        length: Total length of the series.
        fit_end: Exclusive end of the region the run may fit on.
        val_origins: First-forecast-step indices of the validation windows, ascending.
        test_origins: First-forecast-step indices of the test windows, ascending.
    """

    series_id: str
    length: int
    fit_end: int
    val_origins: tuple[int, ...]
    test_origins: tuple[int, ...]


def _origins(last_origin: int, count: int, stride: int) -> tuple[int, ...]:
    """Return ``count`` ascending origins ending at ``last_origin``, spaced by ``stride``."""
    return tuple(last_origin - (count - 1 - i) * stride for i in range(count))


def plan_split(series_id: str, length: int, protocol: BacktestProtocol) -> SplitPlan:
    """Compute the fit/validation/test boundaries for a single series.

    Args:
        series_id: Identifier, used only in error messages and the returned plan.
        length: Number of observations in the series.
        protocol: The evaluation protocol.

    Returns:
        The :class:`SplitPlan`.

    Raises:
        ValueError: If the series is too short to satisfy the protocol.
    """
    if length < protocol.min_series_length:
        raise ValueError(
            f"series {series_id!r} has {length} observations but the protocol needs at least "
            f"{protocol.min_series_length} (context_length {protocol.context_length} + "
            f"{protocol.eval_span} reserved for {protocol.n_val_windows} validation and "
            f"{protocol.n_test_windows} test windows at horizon {protocol.horizon}, "
            f"stride {protocol.effective_stride})"
        )
    stride = protocol.effective_stride
    test_origins = _origins(length - protocol.horizon, protocol.n_test_windows, stride)
    if protocol.n_val_windows:
        # The last validation target must end exactly where the test region begins.
        val_origins = _origins(test_origins[0] - protocol.horizon, protocol.n_val_windows, stride)
        fit_end = val_origins[0]
    else:
        val_origins = ()
        fit_end = test_origins[0]

    # Invariants worth asserting: the fit region is long enough to supply a full context,
    # and no window can see anything at or after its own origin.
    if fit_end < protocol.context_length:  # pragma: no cover - guarded by min_series_length
        raise AssertionError(f"fit_end {fit_end} < context_length {protocol.context_length}")
    return SplitPlan(
        series_id=series_id,
        length=length,
        fit_end=fit_end,
        val_origins=val_origins,
        test_origins=test_origins,
    )


@dataclass(frozen=True, eq=False)
class Window:
    """One forecast task: a context ending at ``origin`` and the target that follows it.

    Attributes:
        series_id: Series the window was cut from.
        origin: Index of the first forecast step. ``context`` is ``values[:origin]``'s tail
            and ``target`` is ``values[origin:origin + horizon]``.
        context: ``(context_length,)`` observations strictly before ``origin``.
        target: ``(horizon,)`` observations from ``origin`` onwards.
    """

    series_id: str
    origin: int
    context: Array
    target: Array


def cut_window(
    series_id: str, values: Array, origin: int, *, horizon: int, context_length: int
) -> Window:
    """Cut a single window out of a series.

    Args:
        series_id: Identifier for the resulting window.
        values: Full 1-D series.
        origin: Index of the first forecast step.
        horizon: Number of steps to forecast.
        context_length: Number of past observations to include.

    Returns:
        The :class:`Window`.

    Raises:
        ValueError: If the window would run off either end of the series.
    """
    if origin - context_length < 0:
        raise ValueError(
            f"series {series_id!r}: origin {origin} leaves only {origin} past observations, "
            f"fewer than the requested context_length {context_length}"
        )
    if origin + horizon > values.size:
        raise ValueError(
            f"series {series_id!r}: window at origin {origin} with horizon {horizon} runs "
            f"past the end of the series (length {values.size})"
        )
    return Window(
        series_id=series_id,
        origin=origin,
        context=values[origin - context_length : origin],
        target=values[origin : origin + horizon],
    )


def stack_windows(windows: Sequence[Window]) -> tuple[Array, Array, tuple[str, ...]]:
    """Stack windows into batched arrays.

    Args:
        windows: Windows sharing a context length and horizon.

    Returns:
        ``(contexts, targets, series_ids)`` with shapes ``(n, context_length)``,
        ``(n, horizon)`` and ``(n,)``.
    """
    if not windows:
        raise ValueError("cannot stack an empty sequence of windows")
    contexts = np.stack([w.context for w in windows])
    targets = np.stack([w.target for w in windows])
    return contexts, targets, tuple(w.series_id for w in windows)


def training_origins(
    fit_end: int, *, horizon: int, context_length: int, stride: int
) -> tuple[int, ...]:
    """Return the ascending forecast origins of the training windows of one series.

    Origins are laid out backwards from ``fit_end - horizon`` in steps of ``stride``, so the
    window nearest the fit boundary -- the one whose statistics most resemble the evaluation
    region -- is always included regardless of how the stride divides the region.

    Args:
        fit_end: Exclusive end of the fit region. Nothing at or beyond it is reachable.
        horizon: Forecast horizon in steps.
        context_length: Number of past observations per window.
        stride: Step between consecutive origins.

    Returns:
        The origins, ascending. Empty if the fit region cannot hold a single window.
    """
    last = fit_end - horizon
    if last < context_length:
        return ()
    count = 1 + (last - context_length) // stride
    return _origins(last, count, stride)


def make_training_windows(
    split: BacktestSplit, *, stride: int | None = None, max_per_series: int | None = None
) -> tuple[Window, ...]:
    """Cut the windows a fine-tuning run is allowed to train on.

    LEAKAGE BOUNDARY. Windows are cut from ``split.train[series_id]``, which is
    ``values[:fit_end]``, rather than from the full series. Slicing the array first means an
    origin arithmetic error cannot reach the validation or test region -- it raises out of
    :func:`cut_window` instead of silently training on data the model is scored against.

    Args:
        split: The backtest split. Only its fit regions are read.
        stride: Step between consecutive origins. Defaults to the protocol's horizon, giving
            non-overlapping targets so no observation is trained on twice per epoch.
        max_per_series: Keep at most this many windows per series, the ones closest to the
            fit boundary. ``None`` keeps all of them.

    Returns:
        The windows, series-major and ascending by origin within each series.

    Raises:
        ValueError: If ``stride`` or ``max_per_series`` is not positive, or if the fit region
            of any series is too short to hold a single window.
    """
    protocol = split.protocol
    step = protocol.horizon if stride is None else stride
    if step < 1:
        raise ValueError(f"stride must be >= 1; got {step}")
    if max_per_series is not None and max_per_series < 1:
        raise ValueError(f"max_per_series must be >= 1; got {max_per_series}")

    windows: list[Window] = []
    for series_id, fit_region in split.train.items():
        origins = training_origins(
            fit_region.size,
            horizon=protocol.horizon,
            context_length=protocol.context_length,
            stride=step,
        )
        if not origins:
            raise ValueError(
                f"series {series_id!r}: its {fit_region.size}-observation fit region cannot "
                f"hold a training window of context_length {protocol.context_length} plus "
                f"horizon {protocol.horizon}. Shorten the context, shorten the horizon, or "
                "reserve fewer evaluation windows."
            )
        if max_per_series is not None:
            origins = origins[-max_per_series:]
        windows.extend(
            cut_window(
                series_id,
                fit_region,
                origin,
                horizon=protocol.horizon,
                context_length=protocol.context_length,
            )
            for origin in origins
        )
    return tuple(windows)


@dataclass(frozen=True, eq=False)
class BacktestSplit:
    """A dataset partitioned into fit regions and rolling-origin evaluation windows.

    Attributes:
        dataset: The source dataset.
        protocol: The protocol used to cut the split.
        plans: Per-series index boundaries, keyed by series id.
        train: Per-series fit-region values, ``values[:fit_end]``. This is the *only* data a
            run may fit on.
        val_windows: Validation windows across all series, series-major.
        test_windows: Test windows across all series, series-major.
        mase_scales: Per-series seasonal-naive MAE, computed on ``train`` only.
    """

    dataset: TimeSeriesDataset
    protocol: BacktestProtocol
    plans: dict[str, SplitPlan]
    train: dict[str, Array]
    val_windows: tuple[Window, ...]
    test_windows: tuple[Window, ...]
    mase_scales: dict[str, float]

    def scales_for(self, windows: Iterable[Window]) -> Array:
        """Return the per-row MASE scale array aligned with ``windows``."""
        return np.array([self.mase_scales[w.series_id] for w in windows], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the split for the metrics artifact."""
        return {
            "dataset": self.dataset.describe(),
            "protocol": self.protocol.model_dump(),
            "effective_stride": self.protocol.effective_stride,
            "n_val_windows_total": len(self.val_windows),
            "n_test_windows_total": len(self.test_windows),
            "fit_end": {sid: plan.fit_end for sid, plan in self.plans.items()},
            "train_lengths": {sid: int(v.size) for sid, v in self.train.items()},
        }


def make_split(dataset: TimeSeriesDataset, protocol: BacktestProtocol) -> BacktestSplit:
    """Cut a dataset into fit regions and rolling-origin evaluation windows.

    Args:
        dataset: The dataset to split.
        protocol: The evaluation protocol.

    Returns:
        The :class:`BacktestSplit`.

    Raises:
        ValueError: If any series is too short for the protocol, or if a fit region is too
            short to define the MASE denominator at the dataset's seasonality.
    """
    plans: dict[str, SplitPlan] = {}
    train: dict[str, Array] = {}
    scales: dict[str, float] = {}
    val_windows: list[Window] = []
    test_windows: list[Window] = []

    for series in dataset:
        plan = plan_split(series.series_id, len(series), protocol)
        plans[series.series_id] = plan
        # LEAKAGE BOUNDARY. Slicing here, once, is what keeps training data separate from
        # evaluation data: the scaler and the MASE denominator below both derive from this
        # array and never see anything at or beyond fit_end.
        fit_region = series.values[: plan.fit_end]
        train[series.series_id] = fit_region
        try:
            scales[series.series_id] = seasonal_naive_scale(fit_region, dataset.seasonality)
        except ValueError as exc:
            raise ValueError(
                f"{dataset.name}: cannot compute the MASE scale for series "
                f"{series.series_id!r} from its {fit_region.size}-observation fit region: {exc}"
            ) from exc

        for origins, sink in ((plan.val_origins, val_windows), (plan.test_origins, test_windows)):
            for origin in origins:
                sink.append(
                    cut_window(
                        series.series_id,
                        series.values,
                        origin,
                        horizon=protocol.horizon,
                        context_length=protocol.context_length,
                    )
                )

    return BacktestSplit(
        dataset=dataset,
        protocol=protocol,
        plans=plans,
        train=train,
        val_windows=tuple(val_windows),
        test_windows=tuple(test_windows),
        mase_scales=scales,
    )
