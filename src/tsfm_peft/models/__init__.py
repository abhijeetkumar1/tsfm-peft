"""Model adapters for time series foundation models."""

from tsfm_peft.models.base import DatasetContext, Forecast, ForecastModel
from tsfm_peft.models.naive import SeasonalNaiveModel, SeasonalNaiveOptions
from tsfm_peft.models.registry import (
    MODELS,
    ModelSpec,
    available_models,
    build_model,
    get_model_spec,
    validate_options,
)
from tsfm_peft.models.timesfm import TimesFmOptions, count_parameters

__all__ = [
    "MODELS",
    "DatasetContext",
    "Forecast",
    "ForecastModel",
    "ModelSpec",
    "SeasonalNaiveModel",
    "SeasonalNaiveOptions",
    "TimesFmOptions",
    "available_models",
    "build_model",
    "count_parameters",
    "get_model_spec",
    "validate_options",
]
