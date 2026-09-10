"""Dataset loading, caching, windowing and scaling."""

from tsfm_peft.data.dataset import Series, TimeSeriesDataset
from tsfm_peft.data.registry import DATASETS, DatasetSpec, available_datasets, load_dataset
from tsfm_peft.data.scaling import AffineScaler, fit_scaler, fit_scalers
from tsfm_peft.data.windows import (
    BacktestProtocol,
    BacktestSplit,
    Window,
    make_split,
    plan_split,
    stack_windows,
)

__all__ = [
    "DATASETS",
    "AffineScaler",
    "BacktestProtocol",
    "BacktestSplit",
    "DatasetSpec",
    "Series",
    "TimeSeriesDataset",
    "Window",
    "available_datasets",
    "fit_scaler",
    "fit_scalers",
    "load_dataset",
    "make_split",
    "plan_split",
    "stack_windows",
]
