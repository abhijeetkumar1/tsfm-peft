"""Pydantic-validated configuration loaded from YAML.

Plain pydantic models rather than Hydra: an experiment config should be readable in one
screen and should fail loudly on a typo. ``extra="forbid"`` on every model is what turns a
misspelled key into an error instead of a silently ignored setting that makes two runs
differ for reasons nothing records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsfm_peft.data.registry import available_datasets, load_dataset
from tsfm_peft.data.scaling import SCALERS
from tsfm_peft.data.windows import BacktestProtocol, BacktestSplit, make_split

ModelT = TypeVar("ModelT", bound=BaseModel)


class DataConfig(BaseModel):
    """Which dataset to evaluate on, and how to cut it.

    Attributes:
        dataset: A key from :func:`~tsfm_peft.data.registry.available_datasets`.
        protocol: The rolling-origin backtest parameters.
        scaler: Fitting rule for the per-series scaler, from
            :data:`~tsfm_peft.data.scaling.SCALERS`. Fitted on the training region only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    protocol: BacktestProtocol
    scaler: str = Field(default="standard")

    @field_validator("dataset")
    @classmethod
    def _known_dataset(cls, value: str) -> str:
        """Fail at config-load time rather than after a download."""
        if value not in available_datasets():
            raise ValueError(
                f"unknown dataset {value!r}; available: {', '.join(available_datasets())}"
            )
        return value

    @field_validator("scaler")
    @classmethod
    def _known_scaler(cls, value: str) -> str:
        """Fail at config-load time rather than mid-training."""
        if value not in SCALERS:
            raise ValueError(f"unknown scaler {value!r}; available: {', '.join(sorted(SCALERS))}")
        return value

    def build_split(self, *, force_download: bool = False) -> BacktestSplit:
        """Download the dataset if needed and cut it into the configured split.

        Args:
            force_download: Re-download even if a valid cached copy exists.

        Returns:
            The :class:`~tsfm_peft.data.windows.BacktestSplit`.
        """
        dataset = load_dataset(self.dataset, force_download=force_download)
        return make_split(dataset, self.protocol)


def load_yaml(path: str | Path, model: type[ModelT]) -> ModelT:
    """Parse a YAML file into a pydantic model.

    Args:
        path: Path to the YAML file.
        model: The model class to validate against.

    Returns:
        The validated model instance.

    Raises:
        ValueError: If the file does not contain a mapping at the top level.
    """
    with open(path, encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(payload)}")
    return model.model_validate(payload)
