# tsfm-peft

Parameter-efficient fine-tuning (LoRA, DoRA) for time series foundation models, benchmarked
honestly against zero-shot baselines.

> **Status: v0.1 in progress.** This README is a placeholder; the real one lands with
> milestone 6. The table below is generated from run artifacts by
> `scripts/build_readme_table.py` and is never hand-edited, so no benchmark number appears
> here until it comes from a real run.

## Results

<!-- BEGIN RESULTS TABLE -->

### etth1

Test windows: horizon 96, context 512, 8 rolling origins per series.

| Arm | MASE | sMAPE | WQL | WQL (macro) | Trained params | Peak GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 zero-shot *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + LoRA r4 *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + LoRA r8 *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + LoRA r16 *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + LoRA r32 *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + DoRA r16 *(not run)* | -- | -- | -- | -- | -- | -- | -- |

### nn5_daily

Test windows: horizon 56, context 256, 3 rolling origins per series.

| Arm | MASE | sMAPE | WQL | WQL (macro) | Trained params | Peak GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 zero-shot *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + LoRA r16 *(not run)* | -- | -- | -- | -- | -- | -- | -- |
| TimesFM 2.5 + DoRA r16 *(not run)* | -- | -- | -- | -- | -- | -- | -- |

- `--` marks an arm whose config exists but has not been run yet.

<!-- END RESULTS TABLE -->

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

## Datasets

Downloaded on first use into `~/.cache/tsfm_peft` (override with `TSFM_PEFT_CACHE`) and
pinned by SHA-256, so an upstream edit fails loudly instead of silently changing results.
No data is committed to this repository.

| Dataset | Content | License | Source |
|---|---|---|---|
| `etth1` | Electricity Transformer Temperature, hourly, 7 channels x 17420 steps, each channel treated as an independent univariate series | CC BY-ND 4.0 | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| `nn5_daily` | Daily ATM cash withdrawals, 111 series x 791 steps, Monash "without missing values" variant | CC BY 4.0 | [Zenodo 4656117](https://zenodo.org/records/4656117) |

ETTh1 is CC BY-ND: it is downloaded at runtime and never redistributed or shipped in
modified form. NN5's missing values were imputed upstream by the Monash authors (median of
the same weekday); those imputed points are still scored here.

## Development

```bash
uv sync --extra dev                  # core + test tooling, no torch
uv sync --extra dev --extra models   # adds torch/transformers/peft
uv run pytest
uvx ruff check . && uvx ruff format --check .
```

## License

Apache-2.0, see [LICENSE](LICENSE). Dataset licenses are documented separately in the datasets
section of the final README.
