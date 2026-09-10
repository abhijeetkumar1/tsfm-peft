"""Seasonal-naive baseline.

Two reasons this is in the library rather than in the tests:

* **It is the reference every MASE number is implicitly quoted against.** MASE divides by
  the *in-sample* seasonal-naive error, so a foundation model can score below 1.0 while
  still losing to an out-of-sample seasonal naive on the same windows. Publishing the
  baseline's out-of-sample row alongside the PEFT arms is what makes the comparison honest;
  omitting it is the most common way a foundation-model benchmark flatters itself.
* **It needs no weights and no torch,** so the whole pipeline -- windowing, prediction,
  metrics, artifact writing -- can be exercised end to end on CPU in CI.

The point forecast is the last observed seasonal cycle, tiled forward. The predictive
distribution is the empirical distribution of in-context seasonal-naive residuals, widened
by ``sqrt(k)`` at ``k`` cycles ahead, which is the seasonal random walk's variance growth.
Drift is deliberately not modelled: a drift term would make the median quantile diverge from
the point forecast, and this is meant to be a floor, not a competitor.

Everything is computed from the context alone, which is strictly pre-origin, so the baseline
cannot see a target it is being scored on.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsfm_peft.metrics import DEFAULT_QUANTILE_LEVELS
from tsfm_peft.models.base import Array, DatasetContext, Forecast, ForecastModel


class SeasonalNaiveOptions(BaseModel):
    """Configuration for :class:`SeasonalNaiveModel`.

    Attributes:
        seasonality: Seasonal period in steps. ``None`` means "take it from the dataset",
            which is the usual setting; an explicit value overrides it and is checked
            against the dataset so a config and a registry entry cannot silently disagree.
        quantile_levels: Levels the empirical predictive distribution is evaluated at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seasonality: int | None = Field(default=None, gt=0)
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS


class SeasonalNaiveModel(ForecastModel):
    """Seasonal-naive point forecasts with empirical residual quantiles."""

    name = "seasonal_naive"

    def __init__(self, options: SeasonalNaiveOptions) -> None:
        """Store options; seasonality may still arrive via :meth:`bind_dataset`."""
        self._options = options
        self._seasonality = options.seasonality

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Levels the empirical predictive distribution is evaluated at."""
        return tuple(float(q) for q in self._options.quantile_levels)

    @property
    def max_horizon(self) -> int | None:
        """Unbounded: tiling a seasonal cycle forward costs nothing."""
        return None

    @property
    def batch_size(self) -> int:
        """The whole batch at once; this is a handful of vectorised numpy operations."""
        return 4096

    @property
    def seasonality(self) -> int:
        """The seasonal period in use.

        Raises:
            ValueError: If neither the config nor :meth:`bind_dataset` supplied one.
        """
        if self._seasonality is None:
            raise ValueError(
                "seasonal_naive has no seasonality: set model.options.seasonality in the "
                "config, or evaluate through the driver, which supplies the dataset's."
            )
        return self._seasonality

    def bind_dataset(self, context: DatasetContext) -> None:
        """Adopt the dataset's seasonality, or verify an explicitly configured one."""
        configured = self._options.seasonality
        if configured is not None and configured != context.seasonality:
            raise ValueError(
                f"seasonal_naive is configured with seasonality {configured} but dataset "
                f"{context.name!r} declares {context.seasonality}. Remove the override to "
                "use the dataset's, or fix whichever one is wrong."
            )
        self._seasonality = context.seasonality

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description for the metrics artifact."""
        return {**super().describe(), "seasonality": self._seasonality}

    def _predict_batch(self, contexts: Array, horizon: int) -> Forecast:
        """Tile the last seasonal cycle and widen the empirical residual spread."""
        period = self.seasonality
        n_rows, context_length = contexts.shape
        if context_length <= period:
            raise ValueError(
                f"seasonal_naive needs a context longer than the seasonal period; got "
                f"context_length {context_length} with seasonality {period}"
            )

        step = np.arange(horizon)
        point = contexts[:, -period:][:, step % period]

        # Residuals of the seasonal naive *within the context*: y_t - y_{t-m}.
        residuals = contexts[:, period:] - contexts[:, :-period]
        centre = np.median(residuals, axis=1, keepdims=True)
        levels = np.asarray(self.quantile_levels, dtype=np.float64)
        spread = np.quantile(residuals - centre, levels, axis=1).T

        # k full cycles ahead: a seasonal random walk's variance grows linearly in k, so the
        # spread grows as sqrt(k). Step 0 is one cycle ahead, hence the +1.
        cycles = step // period + 1
        widening = np.sqrt(cycles, dtype=np.float64)[None, :, None]
        quantiles = point[:, :, None] + widening * spread[:, None, :]

        if quantiles.shape != (n_rows, horizon, levels.size):  # pragma: no cover - defensive
            raise AssertionError(f"unexpected quantile shape {quantiles.shape}")
        return Forecast(point=point, quantiles=quantiles, levels=self.quantile_levels)


def build_seasonal_naive(options: SeasonalNaiveOptions) -> SeasonalNaiveModel:
    """Build a :class:`SeasonalNaiveModel` from validated options."""
    return SeasonalNaiveModel(options)
