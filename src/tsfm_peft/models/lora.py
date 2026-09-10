"""LoRA and DoRA wrapping, driven entirely by config.

One options block covers both methods because in ``peft`` they are one implementation:
DoRA is ``LoraConfig(use_dora=True)``, which decomposes each adapted weight into a magnitude
vector and a direction and trains them separately. Registering a second adapter class for it
would duplicate everything the two share and let the two arms drift apart, which is exactly
what the benchmark is trying to measure.

That is also what makes the rank ablation a matter of adding YAML files: rank, alpha and the
target-module set are config, so an ablation row is a config file and the artifact it wrote,
not a code path.

Determinism note: ``get_peft_model`` initialises the ``lora_A`` matrices from the global
torch RNG (``lora_B`` starts at zero, so the wrapped model is initially identical to the
base model). Seeding therefore has to happen before the model is built, which is the order
:mod:`tsfm_peft.experiment` already runs in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Attention and MLP projections of every TimesFM 2.5 decoder layer. The default arm: the
#: usual LoRA surface, and it leaves the input embedding and the output projection heads
#: frozen so the horizon-to-quantile mapping the checkpoint was pretrained with is preserved.
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")

_MISSING_PEFT = (
    "LoRA and DoRA need the optional model dependencies. Install them with "
    "`uv sync --extra models` (or `pip install 'tsfm-peft[models]'`)."
)


class PeftOptions(BaseModel):
    """Parameter-efficient fine-tuning configuration.

    Attributes:
        method: ``"lora"`` or ``"dora"``. DoRA adds a per-output-channel magnitude vector on
            top of the same low-rank update, so it trains slightly more parameters than LoRA
            at the same rank and costs more per step.
        rank: Rank of the low-rank update. The ablation axis.
        alpha: LoRA scaling numerator; the update is scaled by ``alpha / rank``. Holding
            ``alpha / rank`` fixed across an ablation keeps the effective learning rate of
            the update comparable between ranks, which is what makes the rows comparable.
        dropout: Dropout on the LoRA input. Non-zero draws from the torch RNG every step;
            still reproducible under a fixed seed, but it is another thing that has to match
            for two runs to agree.
        target_modules: Module name suffixes to adapt.
        modules_to_save: Modules trained in full and saved alongside the adapter. Empty by
            default: anything listed here is a full-rank update whose parameters count
            against the "parameter-efficient" claim, and the artifact reports them as
            trainable so that shows up in the table.
        bias: Which biases to train, passed through to ``peft``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["lora", "dora"] = "lora"
    rank: int = Field(default=16, gt=0)
    alpha: float = Field(default=32.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    modules_to_save: tuple[str, ...] = ()
    bias: Literal["none", "all", "lora_only"] = "none"

    @field_validator("target_modules")
    @classmethod
    def _non_empty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an empty or duplicated target set.

        An empty set would produce a model with no trainable parameters that trains happily
        for hours and changes nothing -- the failure mode this repo can least afford.
        """
        if not value:
            raise ValueError(
                "target_modules is empty, which would train nothing; name at least one "
                f"module (the TimesFM 2.5 default is {', '.join(DEFAULT_TARGET_MODULES)})"
            )
        if len(set(value)) != len(value):
            raise ValueError(f"target_modules contains duplicates: {value}")
        return value

    @property
    def use_dora(self) -> bool:
        """Whether this configuration selects DoRA."""
        return self.method == "dora"

    def to_peft_config(self) -> Any:
        """Build the ``peft.LoraConfig`` this options block describes.

        ``task_type`` is left unset: TimesFM is not a sequence-classification or causal-LM
        head, and the generic wrapper is what applies cleanly to an arbitrary module.

        Returns:
            The ``LoraConfig``.

        Raises:
            ImportError: If the optional model dependencies are not installed.
        """
        try:
            from peft import LoraConfig
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MISSING_PEFT) from exc

        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            modules_to_save=list(self.modules_to_save) or None,
            bias=self.bias,
            use_dora=self.use_dora,
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the metrics artifact."""
        return {
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.alpha / self.rank,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "modules_to_save": list(self.modules_to_save),
            "bias": self.bias,
        }


def apply_peft(module: Any, options: PeftOptions) -> Any:
    """Wrap a torch module with LoRA or DoRA adapters.

    Args:
        module: The base ``nn.Module``. Its parameters are frozen in place by ``peft``.
        options: The validated options.

    Returns:
        The wrapped ``PeftModel``.

    Raises:
        ImportError: If the optional model dependencies are not installed.
        ValueError: If no target module matched, or if the wrap left nothing trainable.
    """
    try:
        from peft import get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING_PEFT) from exc

    try:
        wrapped = get_peft_model(module, options.to_peft_config())
    except ValueError as exc:
        # peft's own message names the config but not the model, which is the half a reader
        # needs to fix a typo in target_modules.
        raise ValueError(
            f"could not attach {options.method} adapters to {type(module).__name__} with "
            f"target_modules {list(options.target_modules)}: {exc}"
        ) from exc

    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    if not trainable:  # pragma: no cover - defensive; peft raises first in practice
        raise ValueError(
            f"attaching {options.method} adapters left no trainable parameters; check "
            f"target_modules {list(options.target_modules)}"
        )
    return wrapped


def adapter_state(module: Any) -> dict[str, Any]:
    """Return a detached CPU copy of the adapter weights.

    Used to hold the best-scoring checkpoint in memory during training. Only the adapter
    tensors are copied -- a few megabytes at typical ranks -- so keeping the best step costs
    nothing next to a full-model snapshot.

    Args:
        module: A wrapped ``PeftModel``.

    Returns:
        The state dict, detached and on CPU.
    """
    from peft import get_peft_model_state_dict

    return {
        key: value.detach().to("cpu").clone()
        for key, value in get_peft_model_state_dict(module).items()
    }


def load_adapter_state(module: Any, state: dict[str, Any]) -> None:
    """Load adapter weights previously captured by :func:`adapter_state`.

    Args:
        module: A wrapped ``PeftModel``.
        state: The state dict to restore.

    Raises:
        ValueError: If any adapter tensor in the module was left unset.
    """
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(module, state)
    missing = getattr(result, "missing_keys", ())
    if missing:  # pragma: no cover - defensive
        raise ValueError(f"adapter state is missing {len(missing)} keys, e.g. {missing[:3]}")


def save_adapter(module: Any, path: str | Path) -> Path:
    """Write the adapter weights and their config to a directory.

    Only the adapter is written, not the base checkpoint: reloading needs the base weights
    from HuggingFace plus this directory, which is what keeps a saved arm a few megabytes
    instead of a gigabyte.

    Args:
        module: A wrapped ``PeftModel``.
        path: Destination directory, created if needed.

    Returns:
        The directory written.
    """
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    module.save_pretrained(str(destination))
    return destination
