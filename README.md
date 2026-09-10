# tsfm-peft

Parameter-efficient fine-tuning (LoRA, DoRA) for time series foundation models, benchmarked
honestly against zero-shot baselines.

> **Status: v0.1 in progress.** This README is a placeholder. The real one — opening with a
> results table generated from run artifacts by `scripts/build_readme_table.py` — lands with
> milestone 6. No benchmark numbers appear here until they come from a real run.

## Scope of v0.1

| | |
|---|---|
| Model | TimesFM 2.5 (200M), `google/timesfm-2.5-200m-transformers`, Apache-2.0 |
| Arms | zero-shot, LoRA, DoRA (+ a config-driven rank ablation) |
| Datasets | ETTh1; NN5 Daily (Monash) |
| Protocol | rolling-origin backtest, single fit on the training prefix |
| Metrics | MASE, sMAPE, weighted quantile loss, plus trainable-parameter count/%, peak memory, wall-clock |

Explicit non-goals: no web UI, no multi-GPU, no new architectures, no Moirai/Chronos
implementations (the adapter interface is designed for them), no experiment-tracking service
as a hard dependency.

## Development

```bash
uv sync --extra dev                  # core + test tooling, no torch
uv sync --extra dev --extra models   # adds torch/transformers/peft
uv run pytest
uvx ruff check . && uvx ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE). Dataset licenses are documented separately in the datasets
section of the final README.
