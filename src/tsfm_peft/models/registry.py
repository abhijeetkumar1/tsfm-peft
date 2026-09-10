"""The model registry: the single place a new adapter is wired in.

Each entry pairs a pydantic options model with a builder. Validating options against a
per-adapter model rather than passing a free-form dictionary keeps the ``extra="forbid"``
guarantee that the rest of the configuration has: a misspelled ``target_moduls`` fails when
the config is loaded, not silently at the end of a GPU run that turns out to have fine-tuned
nothing.

Registering an adapter must not import torch. Adapter modules therefore keep their heavy
imports inside the builder, so ``tsfm-peft`` can validate configs, list models and run the
torch-free baselines in an environment with only the core dependencies installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from tsfm_peft.models.base import ForecastModel
from tsfm_peft.models.naive import SeasonalNaiveOptions, build_seasonal_naive
from tsfm_peft.models.timesfm import (
    TimesFmOptions,
    TinyTimesFmOptions,
    build_timesfm,
    build_tiny_timesfm,
)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registered model adapter.

    Attributes:
        name: Registry key, used in config files.
        options_model: Pydantic model the config's ``options`` block is validated against.
        builder: Callable turning validated options into a :class:`ForecastModel`.
        extra: Name of the ``pyproject`` optional-dependency group the adapter needs, or
            ``None`` if it runs on the core dependencies alone.
        description: One-line description for documentation.
        finetunable: Whether the adapter implements
            :class:`~tsfm_peft.models.base.FineTunableModel`. Checked when a config carries a
            ``training`` block, so asking to fine-tune a baseline fails at load time rather
            than after the data is downloaded.
        is_fixture: Whether the adapter exists only to exercise the pipeline. Fixtures carry
            no pretrained weights and their forecasts are meaningless; the flag is what keeps
            a smoke run from being mistaken for a benchmark row.
    """

    name: str
    options_model: type[BaseModel]
    builder: Callable[[Any], ForecastModel]
    extra: str | None
    description: str
    finetunable: bool = False
    is_fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the static fields as a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "extra": self.extra,
            "description": self.description,
            "finetunable": self.finetunable,
            "is_fixture": self.is_fixture,
        }


MODELS: dict[str, ModelSpec] = {
    "seasonal_naive": ModelSpec(
        name="seasonal_naive",
        options_model=SeasonalNaiveOptions,
        builder=build_seasonal_naive,
        extra=None,
        description="Seasonal-naive baseline with empirical residual quantiles; no weights.",
    ),
    "timesfm_2p5": ModelSpec(
        name="timesfm_2p5",
        options_model=TimesFmOptions,
        builder=build_timesfm,
        extra="models",
        description="TimesFM 2.5 (200M) via the transformers port; decoder-only, 9 quantiles.",
        finetunable=True,
    ),
    "timesfm_2p5_tiny": ModelSpec(
        name="timesfm_2p5_tiny",
        options_model=TinyTimesFmOptions,
        builder=build_tiny_timesfm,
        extra="models",
        description="Randomly initialised tiny TimesFM 2.5; a CPU smoke fixture, not a model.",
        finetunable=True,
        is_fixture=True,
    ),
}


def available_models() -> tuple[str, ...]:
    """Return the registered model names, sorted."""
    return tuple(sorted(MODELS))


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name``.

    Args:
        name: Registry key.

    Returns:
        The spec.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        ) from None


def validate_options(name: str, options: dict[str, Any] | None) -> BaseModel:
    """Validate a raw options mapping against the adapter's options model.

    Args:
        name: Registry key.
        options: Raw mapping from YAML, or ``None`` for defaults.

    Returns:
        The validated options instance.
    """
    return get_model_spec(name).options_model.model_validate(options or {})


def is_finetunable(name: str) -> bool:
    """Return whether the registered adapter supports fine-tuning."""
    return get_model_spec(name).finetunable


def build_model(name: str, options: BaseModel | dict[str, Any] | None = None) -> ForecastModel:
    """Construct a model adapter.

    Args:
        name: Registry key.
        options: Validated options, or a raw mapping to validate first.

    Returns:
        The constructed adapter.
    """
    spec = get_model_spec(name)
    validated = options if isinstance(options, BaseModel) else validate_options(name, options)
    model = spec.builder(validated)
    if model.name != spec.name:  # pragma: no cover - defensive
        raise AssertionError(f"adapter for {spec.name!r} reports name {model.name!r}")
    return model
