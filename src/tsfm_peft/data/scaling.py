"""Per-series scaling, fitted on the training region only.

Scaling is where leakage most often enters a forecasting pipeline, because the natural
thing to write -- ``(x - x.mean()) / x.std()`` over a whole series -- makes every training
example depend on the values being predicted. The only public entry point that fits a
scaler here is :func:`fit_scalers`, which takes a :class:`~tsfm_peft.data.windows.BacktestSplit`
and reads exclusively from its ``train`` mapping (``values[:fit_end]``). Individual scalers
can still be fitted directly, but callers doing that are responsible for passing a training
slice.

Note that TimesFM applies its own per-context normalisation inside the model. These scalers
therefore matter mainly for the fine-tuning objective, where they set how much each series
contributes to the loss; they are not a substitute for the model's internal scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from tsfm_peft.data.windows import BacktestSplit

Array = NDArray[np.float64]

#: Guard against dividing by a degenerate spread (a constant training region).
_MIN_SCALE = 1e-8


@runtime_checkable
class Scaler(Protocol):
    """An affine, invertible per-series transform."""

    center: float
    scale: float

    def transform(self, values: Any) -> Array:
        """Map raw values into scaled space."""
        ...

    def inverse_transform(self, values: Any) -> Array:
        """Map scaled values back to raw space."""
        ...


@dataclass(frozen=True)
class AffineScaler:
    """``(x - center) / scale``, with the inverse used to return forecasts to raw units.

    Attributes:
        center: Location parameter subtracted before scaling.
        scale: Strictly positive spread parameter.
        kind: Name of the fitting rule that produced the parameters.
    """

    center: float
    scale: float
    kind: str = "affine"

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive parameters."""
        if not np.isfinite([self.center, self.scale]).all():
            raise ValueError(f"scaler parameters must be finite: {self}")
        if self.scale <= 0:
            raise ValueError(f"scaler scale must be strictly positive, got {self.scale}")

    def transform(self, values: Any) -> Array:
        """Map raw values into scaled space."""
        return (np.asarray(values, dtype=np.float64) - self.center) / self.scale

    def inverse_transform(self, values: Any) -> Array:
        """Map scaled values back to raw space.

        Forecasts must be inverse-transformed before any metric is computed: MASE and WQL
        are both scale-dependent, so evaluating in scaled space silently changes what is
        being measured.
        """
        return np.asarray(values, dtype=np.float64) * self.scale + self.center

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the fitted parameters."""
        return {"kind": self.kind, "center": self.center, "scale": self.scale}


def _as_train_array(values: Any) -> Array:
    """Coerce a training slice to a finite, non-empty 1-D array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"training values must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("training values are empty")
    if not np.isfinite(arr).all():
        raise ValueError("training values contain NaN or inf")
    return arr


def fit_identity(values: Any) -> AffineScaler:
    """Fit a no-op scaler, for running the pipeline in raw units."""
    _as_train_array(values)
    return AffineScaler(center=0.0, scale=1.0, kind="identity")


def fit_standard(values: Any) -> AffineScaler:
    """Fit a mean/standard-deviation scaler on a training slice.

    A constant training region has zero standard deviation; the scale falls back to 1.0 so
    the transform stays invertible instead of producing infinities.
    """
    arr = _as_train_array(values)
    scale = float(arr.std())
    return AffineScaler(center=float(arr.mean()), scale=max(scale, _MIN_SCALE), kind="standard")


def fit_mean_abs(values: Any) -> AffineScaler:
    """Fit a mean-absolute-value scaler (no centering).

    This is the normalisation family most time series foundation models use internally: it
    preserves the sign and the position of zero, which matters for count-like demand series
    such as NN5 where centering would make "no withdrawals" a negative quantity.
    """
    arr = _as_train_array(values)
    scale = float(np.abs(arr).mean())
    return AffineScaler(center=0.0, scale=max(scale, _MIN_SCALE), kind="mean_abs")


#: Fitting rules selectable by name from a config file.
SCALERS = {
    "identity": fit_identity,
    "standard": fit_standard,
    "mean_abs": fit_mean_abs,
}


def fit_scaler(values: Any, kind: str = "standard") -> AffineScaler:
    """Fit a named scaler on a training slice.

    Args:
        values: 1-D training values. LEAKAGE: pass the fit region, never a full series.
        kind: One of :data:`SCALERS`.

    Returns:
        The fitted scaler.
    """
    try:
        rule = SCALERS[kind]
    except KeyError:
        raise ValueError(f"unknown scaler {kind!r}; available: {sorted(SCALERS)}") from None
    return rule(values)


def fit_scalers(split: BacktestSplit, kind: str = "standard") -> dict[str, AffineScaler]:
    """Fit one scaler per series, reading only the split's training regions.

    This is the leak-free entry point: it takes its data from ``split.train``, which holds
    ``values[:fit_end]`` for each series, so no scaler can observe a validation or test
    observation regardless of what the caller does downstream.

    Args:
        split: The backtest split.
        kind: One of :data:`SCALERS`.

    Returns:
        Fitted scalers keyed by series id.
    """
    return {sid: fit_scaler(values, kind) for sid, values in split.train.items()}
