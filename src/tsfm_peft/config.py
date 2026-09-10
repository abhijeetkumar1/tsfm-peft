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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsfm_peft.data.registry import available_datasets, load_dataset
from tsfm_peft.data.scaling import SCALERS
from tsfm_peft.data.windows import BacktestProtocol, BacktestSplit, make_split
from tsfm_peft.models.base import ForecastModel
from tsfm_peft.models.registry import (
    available_models,
    build_model,
    is_finetunable,
    validate_options,
)
from tsfm_peft.training import TrainingConfig

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


class ModelConfig(BaseModel):
    """Which adapter to run, and how to configure it.

    ``options`` is validated against the adapter's own pydantic model rather than being
    passed through as a free dictionary, so a misspelled key fails when the config loads
    instead of after a GPU run that turns out to have configured nothing.

    Attributes:
        name: A key from :func:`~tsfm_peft.models.registry.available_models`.
        options: Adapter-specific options.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_model(cls, value: str) -> str:
        """Fail at config-load time rather than after the data is downloaded."""
        if value not in available_models():
            raise ValueError(f"unknown model {value!r}; available: {', '.join(available_models())}")
        return value

    @model_validator(mode="after")
    def _options_are_valid(self) -> ModelConfig:
        """Validate ``options`` against the adapter's options model."""
        validate_options(self.name, self.options)
        return self

    def resolved_options(self) -> BaseModel:
        """Return the validated adapter options."""
        return validate_options(self.name, self.options)

    def build(self) -> ForecastModel:
        """Construct the adapter. This is where heavy dependencies are first imported."""
        return build_model(self.name, self.resolved_options())


class ExperimentConfig(BaseModel):
    """One experiment: a dataset, a protocol, a model, and a seed.

    An arm of the benchmark is exactly one of these files. Adding a rank ablation is
    therefore adding config files, not writing code, which is what keeps the published
    table auditable: every row points at a file that produced it.

    Attributes:
        name: Identifies the experiment and names its metrics artifact. Must be filesystem
            safe, since it becomes a filename.
        data: Dataset and evaluation protocol.
        model: The adapter and its options.
        training: Fine-tuning hyperparameters. ``None`` is the zero-shot arm: the model is
            evaluated exactly as it was downloaded.
        seed: Seed for every RNG. Recorded in the artifact.
        deterministic: Select deterministic kernels. Costs a few percent of throughput, and
            the wall-clock figures in the artifact are only comparable at the same setting.
        notes: Free text carried into the artifact, for anything a reader would need.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig | None = None
    seed: int = Field(default=0, ge=0)
    deterministic: bool = True
    notes: str | None = None

    @model_validator(mode="after")
    def _training_can_run(self) -> ExperimentConfig:
        """Reject a training block the rest of the config cannot honour.

        All three of these are only discoverable at run time otherwise -- after a dataset
        download and a checkpoint load, which on a fresh machine is several minutes and a
        gigabyte before anything says the run was misconfigured.
        """
        if self.training is None:
            return self
        if not is_finetunable(self.model.name):
            raise ValueError(
                f"model {self.model.name!r} cannot be fine-tuned, but the config has a "
                "training block; remove it or choose a model that can"
            )
        options = self.model.resolved_options()
        if "peft" in type(options).model_fields and options.peft is None:
            raise ValueError(
                "the config asks to train but configures no peft block, so every weight "
                "would stay frozen and the run would change nothing. v0.1 fine-tunes with "
                "LoRA or DoRA only: add a model.options.peft block."
            )
        if self.training.eval_every and not self.data.protocol.n_val_windows:
            raise ValueError(
                f"training validates every {self.training.eval_every} steps but the "
                "protocol reserves no validation windows; set n_val_windows > 0, or "
                "eval_every: 0 to train for a fixed number of steps and keep the last one"
            )
        return self

    @field_validator("name")
    @classmethod
    def _filesystem_safe(cls, value: str) -> str:
        """Reject names that would not survive becoming a filename."""
        if not value or not all(c.isalnum() or c in "-_." for c in value):
            raise ValueError(
                f"experiment name {value!r} must be non-empty and contain only "
                "alphanumerics, '-', '_' and '.'; it is used as a filename"
            )
        return value


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment config, expanding a ``data:`` file reference if present.

    ``data`` may be an inline mapping or a path to a data config. The reference form is
    what keeps the three arms of a dataset honest: they share one protocol file, so a
    horizon cannot drift between the zero-shot and fine-tuned rows. Relative paths resolve
    against the experiment file's own directory.

    Args:
        path: Path to the experiment YAML.

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        ValueError: If the file is not a mapping, or a referenced data config is missing.
    """
    config_path = Path(path)
    with open(config_path, encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{config_path}: expected a YAML mapping at the top level, got {type(payload)}"
        )

    reference = payload.get("data")
    if isinstance(reference, str):
        data_path = Path(reference)
        if not data_path.is_absolute():
            data_path = (config_path.parent / data_path).resolve()
        if not data_path.is_file():
            raise ValueError(
                f"{config_path}: data references {reference!r}, which does not exist "
                f"(resolved to {data_path})"
            )
        with open(data_path, encoding="utf-8") as handle:
            payload = {**payload, "data": yaml.safe_load(handle)}

    return ExperimentConfig.model_validate(payload)
