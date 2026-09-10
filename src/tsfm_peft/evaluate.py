"""The evaluation driver: run a model over a backtest split and score it.

This is the only place forecasts and targets meet, which is deliberate -- it means there is
exactly one function to audit for leakage. What it guarantees:

* Contexts come from :class:`~tsfm_peft.data.windows.Window` objects, which are cut as
  ``values[origin - context_length : origin]``. Nothing at or after an origin is ever handed
  to the model for that origin.
* Scalers, when a model needs them, are fitted through
  :func:`~tsfm_peft.data.scaling.fit_scalers`, which reads only ``split.train``.
* The MASE denominators come from ``split.mase_scales``, computed on the fit region only.
* Metrics are computed in raw units. A model given scaled inputs has its forecasts
  inverse-transformed *before* scoring, because MASE and weighted quantile loss are both
  scale-dependent and scoring in scaled space silently measures something else.

The model's quantile levels drive the metric rather than the other way round: asking a model
for levels it has no head for would mean interpolating a distribution it never learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tsfm_peft.data.scaling import AffineScaler, fit_scalers
from tsfm_peft.data.windows import BacktestSplit, Window, stack_windows
from tsfm_peft.metrics import ForecastMetrics, evaluate_forecasts
from tsfm_peft.models.base import DatasetContext, Forecast, ForecastModel
from tsfm_peft.runtime import ResourceLog

WindowSet = Literal["test", "val"]


@dataclass(frozen=True)
class EvaluationResult:
    """Everything one evaluation pass produced.

    Attributes:
        metrics: Accuracy metrics, aggregate and per series.
        window_set: Which windows were scored, ``"test"`` or ``"val"``.
        quantile_crossing_rate: Fraction of adjacent quantile pairs out of order. Not
            corrected, only reported: sorting crossings away would lower the pinball loss
            without the model having improved.
        scaler_kind: The external scaler applied, or ``None`` when the model normalises
            internally (the case for every v0.1 arm).
        forecast: The raw-unit forecasts, kept in memory for callers that want to save or
            plot them. Not written to the metrics artifact.
    """

    metrics: ForecastMetrics
    window_set: WindowSet
    quantile_crossing_rate: float
    scaler_kind: str | None
    forecast: Forecast

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation, excluding the forecast arrays."""
        return {
            "window_set": self.window_set,
            "scaler_kind": self.scaler_kind,
            "quantile_crossing_rate": self.quantile_crossing_rate,
            **self.metrics.to_dict(),
        }


def select_windows(split: BacktestSplit, window_set: WindowSet) -> tuple[Window, ...]:
    """Return the requested window set.

    Args:
        split: The backtest split.
        window_set: ``"test"`` or ``"val"``.

    Returns:
        The windows.

    Raises:
        ValueError: If the name is unknown, or the protocol produced no such windows.
    """
    if window_set == "test":
        windows = split.test_windows
    elif window_set == "val":
        windows = split.val_windows
    else:
        raise ValueError(f"window_set must be 'test' or 'val'; got {window_set!r}")
    if not windows:
        raise ValueError(
            f"the protocol produced no {window_set} windows; set n_{window_set}_windows > 0"
        )
    return windows


def _scale_contexts(
    contexts: np.ndarray, series_ids: tuple[str, ...], scalers: dict[str, AffineScaler]
) -> np.ndarray:
    """Apply each row's series scaler to its context."""
    centres = np.array([scalers[sid].center for sid in series_ids])[:, None]
    scales = np.array([scalers[sid].scale for sid in series_ids])[:, None]
    return (contexts - centres) / scales


def _unscale_forecast(
    forecast: Forecast, series_ids: tuple[str, ...], scalers: dict[str, AffineScaler]
) -> Forecast:
    """Return forecasts to raw units before any metric touches them."""
    centres = np.array([scalers[sid].center for sid in series_ids])[:, None]
    scales = np.array([scalers[sid].scale for sid in series_ids])[:, None]
    return Forecast(
        point=forecast.point * scales + centres,
        quantiles=forecast.quantiles * scales[:, :, None] + centres[:, :, None],
        levels=forecast.levels,
    )


def evaluate_model(
    model: ForecastModel,
    split: BacktestSplit,
    *,
    window_set: WindowSet = "test",
    scaler_kind: str | None = None,
    resources: ResourceLog | None = None,
) -> EvaluationResult:
    """Forecast every window in a split and score the result.

    Args:
        model: The adapter to evaluate.
        split: The backtest split, carrying the windows and the MASE denominators.
        window_set: Which windows to score.
        scaler_kind: Scaler to fit on the training regions when the model asks for external
            scaling. Ignored -- with a raised error if explicitly set -- when the model
            normalises internally, so a config cannot quietly double-normalise.
        resources: Optional log to time the prediction phase into.

    Returns:
        The :class:`EvaluationResult`.
    """
    windows = select_windows(split, window_set)
    dataset = split.dataset
    model.bind_dataset(
        DatasetContext(name=dataset.name, freq=dataset.freq, seasonality=dataset.seasonality)
    )

    contexts, targets, series_ids = stack_windows(windows)

    scalers: dict[str, AffineScaler] | None = None
    applied_scaler: str | None = None
    if model.requires_external_scaling:
        applied_scaler = scaler_kind or "standard"
        # LEAKAGE BOUNDARY: fit_scalers reads split.train, i.e. values[:fit_end], and
        # nothing else. Fitting on `contexts` here would be the classic mistake -- contexts
        # of later windows reach past fit_end into the evaluation region.
        scalers = fit_scalers(split, applied_scaler)
        contexts = _scale_contexts(contexts, series_ids, scalers)
    elif scaler_kind not in (None, "identity"):
        raise ValueError(
            f"{model.name} normalises contexts internally but the config asks for the "
            f"{scaler_kind!r} scaler. Applying both would double-normalise and change what "
            "the zero-shot number means. Remove the scaler or use an adapter that needs one."
        )

    device = getattr(model, "device", None)
    if resources is not None:
        with resources.phase(f"predict_{window_set}", device=device):
            forecast = model.predict(contexts, split.protocol.horizon)
    else:
        forecast = model.predict(contexts, split.protocol.horizon)

    if scalers is not None:
        forecast = _unscale_forecast(forecast, series_ids, scalers)

    metrics = evaluate_forecasts(
        y_true=targets,
        y_pred=forecast.point,
        y_pred_quantiles=forecast.quantiles,
        series_ids=series_ids,
        scales=split.scales_for(windows),
        levels=forecast.levels,
    )
    return EvaluationResult(
        metrics=metrics,
        window_set=window_set,
        quantile_crossing_rate=forecast.quantile_crossing_rate(),
        scaler_kind=applied_scaler,
        forecast=forecast,
    )
