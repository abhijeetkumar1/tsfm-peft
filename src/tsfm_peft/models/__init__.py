"""Model adapters for time series foundation models."""

from tsfm_peft.models.base import DatasetContext, FineTunableModel, Forecast, ForecastModel
from tsfm_peft.models.lora import DEFAULT_TARGET_MODULES, PeftOptions, apply_peft
from tsfm_peft.models.naive import SeasonalNaiveModel, SeasonalNaiveOptions
from tsfm_peft.models.registry import (
    MODELS,
    ModelSpec,
    available_models,
    build_model,
    get_model_spec,
    is_finetunable,
    validate_options,
)
from tsfm_peft.models.timesfm import TimesFmOptions, TinyTimesFmOptions, count_parameters

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "MODELS",
    "DatasetContext",
    "FineTunableModel",
    "Forecast",
    "ForecastModel",
    "ModelSpec",
    "PeftOptions",
    "SeasonalNaiveModel",
    "SeasonalNaiveOptions",
    "TimesFmOptions",
    "TinyTimesFmOptions",
    "apply_peft",
    "available_models",
    "build_model",
    "count_parameters",
    "get_model_spec",
    "is_finetunable",
    "validate_options",
]
