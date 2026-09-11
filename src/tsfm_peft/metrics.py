"""Forecast accuracy metrics: MASE, sMAPE, wMAPE and weighted quantile loss.

Definitions follow the sources the numbers will be compared against:

* **MASE** -- Hyndman & Koehler (2006), as operationalised by the M4 competition. The
  denominator is the in-sample seasonal-naive MAE of the *training* series.
* **sMAPE** -- the M4 variant, ``200 * mean(|y - yhat| / (|y| + |yhat|))``, which is bounded
  in ``[0, 200]``. Note this is not the ``mean(2|y - yhat| / (y + yhat))`` form that can go
  negative; it is the one used for the M4 leaderboard.
* **wMAPE** -- ``100 * sum|y - yhat| / sum|y|``, pooled over every row and step. The
  weighted form, not the mean of per-row MAPEs: the normaliser is a sum over the whole set,
  so a near-zero observation cannot blow the metric up on its own. Reported in percentage
  points, like sMAPE.
* **Weighted quantile loss** -- the GluonTS / Chronos definition: pinball loss summed over
  levels and normalised by the total absolute magnitude of the targets.

Every metric returns per-row values as well as an aggregate so that the metrics artifact can
carry a per-series breakdown rather than a single opaque number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Deciles, matching the quantile heads TimesFM exposes and the Chronos benchmark protocol.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _as_2d(name: str, values: Any) -> Array:
    """Coerce to a finite 2-D ``(n_rows, horizon)`` float array or raise."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_rows, horizon); got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def seasonal_naive_scale(train: Any, seasonality: int) -> float:
    """Return the MASE denominator: in-sample seasonal-naive MAE.

    LEAKAGE: ``train`` must be the training portion of the series only -- never the full
    series, and never anything at or beyond the first forecast origin. Computing this scale
    over the whole series is a subtle and common way to leak test information into a metric,
    because it makes the denominator depend on the values being predicted.

    Args:
        train: 1-D training values for a single series.
        seasonality: Seasonal period ``m`` (24 for hourly, 7 for daily, 1 for non-seasonal).

    Returns:
        ``mean(|y_t - y_{t-m}|)`` over the training series.

    Raises:
        ValueError: If ``seasonality`` is not positive, the training series is too short, or
            the series is seasonally constant (the denominator would be zero and MASE
            undefined). The last case is raised rather than silently returning ``nan`` so a
            degenerate series is noticed at evaluation time instead of polluting an average.
    """
    if seasonality < 1:
        raise ValueError(f"seasonality must be >= 1; got {seasonality}")
    arr = np.asarray(train, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"train must be 1-D; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("train contains NaN or inf")
    if arr.size <= seasonality:
        raise ValueError(
            f"train has {arr.size} points, which is too short for seasonality {seasonality}"
        )
    scale = float(np.abs(arr[seasonality:] - arr[:-seasonality]).mean())
    if scale == 0.0:
        raise ValueError(
            "seasonal-naive MAE is zero (series is constant at this seasonal lag); "
            "MASE is undefined for this series"
        )
    return scale


def mase(y_true: Any, y_pred: Any, scales: Any) -> Array:
    """Mean absolute scaled error, one value per row.

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred: ``(n_rows, horizon)`` point forecasts.
        scales: ``(n_rows,)`` per-row seasonal-naive scales from :func:`seasonal_naive_scale`.
            Rows belonging to the same series share that series' scale.

    Returns:
        ``(n_rows,)`` array of MASE values.
    """
    truth = _as_2d("y_true", y_true)
    pred = _as_2d("y_pred", y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    scale = np.asarray(scales, dtype=np.float64)
    if scale.shape != (truth.shape[0],):
        raise ValueError(f"scales must have shape {(truth.shape[0],)}; got {scale.shape}")
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scales must be finite and strictly positive")
    return np.abs(truth - pred).mean(axis=1) / scale


def smape(y_true: Any, y_pred: Any) -> Array:
    """Symmetric mean absolute percentage error (M4 variant), one value per row.

    Rows where both the observation and the forecast are exactly zero contribute ``0`` for
    that step rather than ``nan``: a perfect prediction of zero is not an infinite error.

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred: ``(n_rows, horizon)`` point forecasts.

    Returns:
        ``(n_rows,)`` array of sMAPE values in ``[0, 200]``.
    """
    truth = _as_2d("y_true", y_true)
    pred = _as_2d("y_pred", y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    denom = np.abs(truth) + np.abs(pred)
    ratio = np.zeros_like(denom)
    np.divide(np.abs(truth - pred), denom, out=ratio, where=denom > 0)
    return 200.0 * ratio.mean(axis=1)


def wmape(y_true: Any, y_pred: Any) -> float:
    """Weighted MAPE: total absolute error as a percentage of total absolute magnitude.

    A single pooled number rather than one value per row, because that is what makes it
    robust: ``sum|y - yhat| / sum|y|`` normalises by the magnitude of the whole set, so an
    observation near zero contributes its own small share of the denominator instead of
    dividing its own error and dominating a per-row mean the way MAPE does.

    Being magnitude-weighted, it has the same caveat as pooled WQL: over series of very
    different scale the largest series decides most of the number, which is why
    :func:`evaluate_forecasts` reports a macro variant next to it.

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred: ``(n_rows, horizon)`` point forecasts.

    Returns:
        wMAPE in percentage points; ``0.0`` for an exactly-predicted all-zero target.

    Raises:
        ValueError: If the shapes disagree, or if the targets sum to zero magnitude while
            the forecasts do not -- a relative error against nothing, which has no finite
            value and is not worth publishing a stand-in for.
    """
    truth = _as_2d("y_true", y_true)
    pred = _as_2d("y_pred", y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    total = float(np.abs(truth).sum())
    error = float(np.abs(truth - pred).sum())
    if total == 0.0:
        # Mirrors smape's treatment of 0/0: predicting zero for zero is not an error.
        if error == 0.0:
            return 0.0
        raise ValueError(
            "wMAPE is undefined: the targets sum to zero magnitude but the forecasts do not"
        )
    return 100.0 * error / total


def _validate_quantiles(levels: Sequence[float]) -> Array:
    """Coerce quantile levels to a strictly increasing float array in ``(0, 1)``."""
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"quantile levels must be a non-empty 1-D sequence; got {levels!r}")
    if not ((arr > 0.0) & (arr < 1.0)).all():
        raise ValueError(f"quantile levels must lie strictly in (0, 1); got {levels!r}")
    if not (np.diff(arr) > 0).all():
        raise ValueError(f"quantile levels must be strictly increasing; got {levels!r}")
    return arr


def quantile_losses(
    y_true: Any,
    y_pred_quantiles: Any,
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> tuple[Array, float]:
    """Return per-level weighted quantile losses and the normalising denominator.

    The pinball (check) loss for level ``q`` is ``max(q * e, (q - 1) * e)`` with
    ``e = y - yhat_q``. Each level's loss is summed over every row and step, multiplied by 2
    and divided by ``sum(|y|)``. The factor of 2 is the GluonTS convention that makes the
    ``q = 0.5`` level equal to the normalised MAE.

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred_quantiles: ``(n_rows, horizon, n_levels)`` quantile forecasts.
        levels: Strictly increasing quantile levels in ``(0, 1)``.

    Returns:
        A ``(per_level_loss, denominator)`` pair, where ``per_level_loss`` has shape
        ``(n_levels,)`` and ``denominator`` is ``sum(|y_true|)``.
    """
    truth = _as_2d("y_true", y_true)
    quantiles = _validate_quantiles(levels)
    preds = np.asarray(y_pred_quantiles, dtype=np.float64)
    expected = (*truth.shape, quantiles.size)
    if preds.shape != expected:
        raise ValueError(f"y_pred_quantiles must have shape {expected}; got {preds.shape}")
    if not np.isfinite(preds).all():
        raise ValueError("y_pred_quantiles contains NaN or inf")

    denominator = float(np.abs(truth).sum())
    if denominator == 0.0:
        raise ValueError("weighted quantile loss is undefined: all targets are zero")

    error = truth[..., None] - preds
    pinball = np.maximum(quantiles * error, (quantiles - 1.0) * error)
    per_level = 2.0 * pinball.sum(axis=(0, 1)) / denominator
    return per_level, denominator


def weighted_quantile_loss(
    y_true: Any,
    y_pred_quantiles: Any,
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Weighted quantile loss: the mean of the per-level losses.

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred_quantiles: ``(n_rows, horizon, n_levels)`` quantile forecasts.
        levels: Strictly increasing quantile levels in ``(0, 1)``.

    Returns:
        The scalar WQL.
    """
    per_level, _ = quantile_losses(y_true, y_pred_quantiles, levels)
    return float(per_level.mean())


@dataclass(frozen=True)
class ForecastMetrics:
    """Aggregate and per-series accuracy metrics for one evaluation run.

    Attributes:
        mase: Mean MASE over every (series, window) row.
        smape: Mean sMAPE over every (series, window) row.
        wmape: wMAPE pooled across every row and step. Magnitude-weighted, so larger series
            dominate it, for the same reason and in the same way as ``wql``.
        wmape_macro: Mean of the per-series wMAPEs, each series counting once.
        wql: Weighted quantile loss pooled across all series. Series with larger magnitudes
            dominate this number by construction -- that is what "weighted" means here, and
            it is the definition the Chronos/GluonTS numbers use.
        wql_macro: Mean of the per-series weighted quantile losses. Each series counts once
            regardless of magnitude. Reported alongside ``wql`` because for datasets whose
            "series" are channels of very different scale (ETTh1) the pooled number is
            effectively a single-channel metric.
        per_quantile_wql: Pooled loss for each quantile level, keyed by level as a string.
        quantile_levels: The levels the quantile losses were computed at.
        n_series: Number of distinct series evaluated.
        n_rows: Number of (series, window) forecast rows.
        horizon: Forecast horizon in steps.
        per_series: Per-series ``{"mase", "smape", "wmape", "wql", "n_windows"}`` breakdown.
    """

    mase: float
    smape: float
    wmape: float
    wmape_macro: float
    wql: float
    wql_macro: float
    per_quantile_wql: dict[str, float]
    quantile_levels: tuple[float, ...]
    n_series: int
    n_rows: int
    horizon: int
    per_series: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the metrics artifact."""
        return {
            "mase": self.mase,
            "smape": self.smape,
            "wmape": self.wmape,
            "wmape_macro": self.wmape_macro,
            "wql": self.wql,
            "wql_macro": self.wql_macro,
            "per_quantile_wql": dict(self.per_quantile_wql),
            "quantile_levels": list(self.quantile_levels),
            "n_series": self.n_series,
            "n_rows": self.n_rows,
            "horizon": self.horizon,
            "per_series": {k: dict(v) for k, v in self.per_series.items()},
        }


def evaluate_forecasts(
    y_true: Any,
    y_pred: Any,
    y_pred_quantiles: Any,
    series_ids: Sequence[str],
    scales: Any,
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> ForecastMetrics:
    """Compute every metric for a set of backtest windows.

    Rows are (series, window) pairs in any order; ``series_ids`` maps each row back to its
    series so that the per-series breakdown and the macro-averaged WQL can be computed.
    MASE and sMAPE are averaged with equal weight per row, so a series contributes in
    proportion to how many evaluation windows it has (which is constant across series under
    the rolling-origin protocol in :mod:`tsfm_peft.data.windows`).

    Args:
        y_true: ``(n_rows, horizon)`` observed values.
        y_pred: ``(n_rows, horizon)`` point forecasts.
        y_pred_quantiles: ``(n_rows, horizon, n_levels)`` quantile forecasts.
        series_ids: ``n_rows`` series identifiers.
        scales: ``(n_rows,)`` seasonal-naive scales, computed on training data only.
        levels: Quantile levels matching the last axis of ``y_pred_quantiles``.

    Returns:
        The populated :class:`ForecastMetrics`.
    """
    truth = _as_2d("y_true", y_true)
    quantiles = _validate_quantiles(levels)
    if len(series_ids) != truth.shape[0]:
        raise ValueError(
            f"series_ids has {len(series_ids)} entries but y_true has {truth.shape[0]} rows"
        )

    row_mase = mase(truth, y_pred, scales)
    row_smape = smape(truth, y_pred)
    per_level, _ = quantile_losses(truth, y_pred_quantiles, quantiles)
    point = _as_2d("y_pred", y_pred)
    preds_q = np.asarray(y_pred_quantiles, dtype=np.float64)

    ids = np.asarray(series_ids, dtype=object)
    per_series: dict[str, dict[str, float]] = {}
    # dict.fromkeys preserves first-seen order, so the breakdown is deterministic.
    for series_id in dict.fromkeys(series_ids):
        rows = ids == series_id
        per_series[str(series_id)] = {
            "mase": float(row_mase[rows].mean()),
            "smape": float(row_smape[rows].mean()),
            "wmape": wmape(truth[rows], point[rows]),
            "wql": weighted_quantile_loss(truth[rows], preds_q[rows], quantiles),
            "n_windows": int(rows.sum()),
        }

    return ForecastMetrics(
        mase=float(row_mase.mean()),
        smape=float(row_smape.mean()),
        wmape=wmape(truth, point),
        wmape_macro=float(np.mean([m["wmape"] for m in per_series.values()])),
        wql=float(per_level.mean()),
        wql_macro=float(np.mean([m["wql"] for m in per_series.values()])),
        per_quantile_wql={
            f"{q:g}": float(loss) for q, loss in zip(quantiles, per_level, strict=True)
        },
        quantile_levels=tuple(float(q) for q in quantiles),
        n_series=len(per_series),
        n_rows=int(truth.shape[0]),
        horizon=int(truth.shape[1]),
        per_series=per_series,
    )
