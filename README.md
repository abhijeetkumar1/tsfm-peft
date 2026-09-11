# tsfm-peft

[![CI](https://github.com/abhijeetkumar1/tsfm-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeetkumar1/tsfm-peft/actions/workflows/ci.yml)

Does parameter-efficient fine-tuning actually earn its keep on a time series foundation
model? This repo answers that for TimesFM 2.5 with LoRA and DoRA, on public data, under a
protocol you can read in one sitting and reproduce with one command.

## Results

<!-- BEGIN RESULTS TABLE -->

### etth1

Test windows: horizon 96, context 512, 8 rolling origins per series, 7 series, 56 forecast windows.

| Arm | MASE | sMAPE | wMAPE | WQL | WQL (macro) | Trained params | Peak GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive | 1.074 | 33.574 | 29.823 | 0.253 | 0.230 | n/a | n/a | n/a |
| TimesFM 2.5 zero-shot | 0.905 | 29.357 | 24.417 | 0.193 | 0.180 | 0 | 1.06 GB | n/a |
| TimesFM 2.5 + LoRA r16 | 0.910 | 30.004 | 24.242 | 0.192 | 0.182 | 4.92M (2.08%) | 1.45 GB | 8m 07s |
| TimesFM 2.5 + LoRA r32 | 0.886 | 29.189 | 24.019 | 0.189 | 0.176 | 9.83M (4.08%) | 1.51 GB | 4m 36s |
| TimesFM 2.5 + LoRA r64 | 0.909 | 30.181 | 24.671 | 0.195 | 0.181 | 19.66M (7.83%) | 1.64 GB | 4m 33s |
| TimesFM 2.5 + LoRA r128 | 0.956 | 31.706 | 26.046 | 0.205 | 0.190 | 39.32M (14.53%) | 1.88 GB | 4m 32s |
| TimesFM 2.5 + DoRA r16 | 0.917 | 30.262 | 24.365 | 0.193 | 0.184 | 5.07M (2.14%) | 1.78 GB | 11m 33s |
| TimesFM 2.5 + DoRA r32 | 0.884 | 29.126 | 24.009 | 0.189 | 0.176 | 9.98M (4.14%) | 1.84 GB | 6m 14s |
| TimesFM 2.5 + DoRA r64 | 0.909 | 30.180 | 24.657 | 0.195 | 0.181 | 19.81M (7.89%) | 1.96 GB | 6m 21s |
| TimesFM 2.5 + DoRA r128 | 0.957 | 31.765 | 26.051 | 0.205 | 0.190 | 39.48M (14.58%) | 2.21 GB | 6m 18s |

### nn5_daily

Test windows: horizon 56, context 256, 3 rolling origins per series, 111 series, 333 forecast windows.

| Arm | MASE | sMAPE | wMAPE | WQL | WQL (macro) | Trained params | Peak GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive | 1.243 | 30.127 | 26.520 | 0.236 | 0.242 | n/a | n/a | n/a |
| TimesFM 2.5 zero-shot | 0.806 | 20.000 | 17.337 | 0.141 | 0.143 | 0 | 1.01 GB | n/a |
| TimesFM 2.5 + LoRA r16 | 0.821 | 20.322 | 17.639 | 0.143 | 0.145 | 4.92M (2.08%) | 1.25 GB | 1m 55s |
| TimesFM 2.5 + DoRA r16 | 0.821 | 20.325 | 17.642 | 0.143 | 0.145 | 5.07M (2.14%) | 1.43 GB | 2m 43s |

### Aggregate

Unweighted mean over 2 datasets (etth1, nn5_daily), for arms scored on all of them.

| Arm | MASE | sMAPE | wMAPE (macro) | WQL (macro) |
|---|---:|---:|---:|---:|
| Seasonal naive | 1.159 | 31.851 | 27.245 | 0.236 |
| TimesFM 2.5 zero-shot | 0.855 | 24.678 | 20.153 | 0.161 |
| TimesFM 2.5 + LoRA r16 | 0.865 | 25.163 | 20.423 | 0.164 |
| TimesFM 2.5 + DoRA r16 | 0.869 | 25.293 | 20.511 | 0.165 |

- Rows come from more than one commit; re-run the whole set from one tree before publishing.
- At least one row was produced from a dirty working tree and cannot be tied to a commit.
- Rows were produced on more than one machine, so the peak-memory and train-time columns are not comparable between them.
- Ran on a kernel with no deterministic implementation, so these rows reproduce only to within floating-point accumulation order, not bit-exactly: `etth1-timesfm-dora`, `etth1-timesfm-lora`, `etth1-timesfm-lora-r32`, `nn5_daily-timesfm-dora`, `nn5_daily-timesfm-lora`. The artifact names the kernel.

<!-- END RESULTS TABLE -->

Every number above is generated from run artifacts by `scripts/build_readme_table.py` and is
never hand-edited. A number with no artifact behind it cannot appear in this table, and CI
fails if the table and the artifacts disagree.

## Reproduce

```bash
uv sync --extra models                                                    # install
uv run tsfm-peft run configs/experiments/{etth1,nn5_daily}-*.yaml && uv run tsfm-peft table --configs configs/experiments --write
```

The first command installs torch, transformers and peft alongside the package. The second
runs all fourteen arms and regenerates the table above from the artifacts they write. Datasets
and checkpoint weights download on first use into `~/.cache/tsfm_peft` and
`~/.cache/huggingface`; nothing is committed to this repository.

Expect this to want a GPU. The zero-shot and seasonal-naive arms run on CPU in minutes; the
ten fine-tuning arms are 1000 steps each against a 231M-parameter checkpoint.

To run a single arm, or to see what a config resolves to without downloading anything:

```bash
uv run tsfm-peft run configs/experiments/etth1-timesfm-lora.yaml
uv run tsfm-peft run --check configs/experiments/*.yaml
uv run tsfm-peft list
```

### More than one GPU

The arms are independent and one arm uses a fraction of a card, so the parallelism worth
having is across arms, not inside one. `scripts/run_all.sh` runs the whole set with one
process per visible GPU, each pinned with `CUDA_VISIBLE_DEVICES` and calling the same
`tsfm-peft run` a serial invocation would:

```bash
scripts/run_all.sh                        # one process per visible GPU
TSFM_PEFT_PARALLEL=1 scripts/run_all.sh   # force serial
```

An arm that already has an artifact is skipped, so rerunning resumes; a failing arm does not
stop the others; each writes its own log under `logs/`, since concurrent arms cannot share a
terminal legibly. The script refuses to start if a config exists that its arm list does not
mention, so a new arm cannot be silently left out of the table.

**This trades the cost columns for wall clock.** Concurrent arms compete for CPU, disk and
host memory, so their train-time and peak-memory figures include contention a serial run does
not have. Each artifact records `environment.scheduling.concurrent_runs`, and the generated
table names the rows it applies to, so the two kinds of measurement cannot be quietly averaged
together. Run with `TSFM_PEFT_PARALLEL=1` when the cost columns are the point of the exercise.

Distributed training is a different thing and is deliberately not here: splitting a batch
across devices changes the effective batch size and therefore the numbers, which would make
the rows incomparable with every row already published.

### Docker

There is a Docker image if you would rather not install anything:

```bash
docker build -t tsfm-peft .
docker run --rm -v tsfm-peft-cache:/cache tsfm-peft run configs/experiments/etth1-timesfm-zeroshot.yaml
```

## Method

**Rolling-origin backtesting, never a single split.** Each series contributes several
evaluation windows rather than one. Test origins are the last `n_test_windows` positions
spaced one horizon apart, so the windows are non-overlapping and each scores a distinct
span. A window's context is `values[origin - context_length : origin]` and its target is
`values[origin : origin + horizon]`; nothing at or after an origin reaches the model for
that origin.

**Three regions per series, in order: fit, validation, test.** Validation windows sit
immediately before the test region and are used only for checkpoint selection and early
stopping. The fit region is everything before the first validation origin. Because test
origins are computed from the end of each series, reserving validation windows shortens
training without moving a single test target — which is what lets the zero-shot arm (no
validation windows) and the fine-tuned arms share a table.

**Leakage boundaries are marked in the source.** Scalers are fitted on the fit region only
(`evaluate.py`), MASE denominators and training windows both come from the fit region only
(`data/windows.py`), and the trainer sees only the fit region and the validation windows
(`experiment.py`). Each of those places carries a `LEAKAGE BOUNDARY` comment saying what
would go wrong with the obvious shortcut. All
forecasts are returned to raw units before any metric sees them: MASE and weighted quantile
loss are both scale-dependent, so scoring in normalised space would silently measure
something else.

**Metrics follow the sources they would be compared against.** MASE per Hyndman & Koehler
as operationalised by M4, with the in-sample seasonal-naive MAE of the fit region as the
denominator. sMAPE in the M4 form, bounded in [0, 200]. wMAPE as `100 * sum|y - yhat| /
sum|y|` pooled over every row and step — the weighted form, not a mean of per-row MAPEs, so
an observation near zero contributes its share of the denominator instead of dividing its own
error; in percentage points, like sMAPE. Weighted quantile loss in the GluonTS/Chronos form —
pinball loss over the nine deciles, normalised by the total absolute magnitude of the targets.
Pooled WQL and pooled wMAPE are each reported alongside a macro average because for a dataset
whose series are channels of very different scale (ETTh1) the pooled number is effectively a
single-channel metric.

**Cost is reported next to accuracy, always.** Trained parameter count and share, peak GPU
memory, and wall-clock training time sit in the same table as MASE, because an accuracy
number on its own does not tell you whether PEFT was worth it. The trained-parameter column
counts what a run actually updated, not what `requires_grad` says: an untrained base model
reports all 231M of its weights as trainable, and crediting the zero-shot arm with that
would invert the whole comparison.

**Fine-tuning.** LoRA and DoRA are attached via `peft` to the attention and MLP projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc1`, `fc2`) of all 20 decoder layers. The input
embedding and the output quantile heads stay frozen, so the pretrained horizon-to-quantile
mapping is preserved and the trainable count stays honest. Checkpoints are selected on
validation MASE with early stopping; only the adapter is saved.

**Determinism.** Every run seeds Python, NumPy and torch, selects deterministic kernels, and
records the seed, the package versions, the git commit and the hardware into its artifact.
Deterministic kernels are requested with `warn_only=True` — a benchmark that refuses to run
measures nothing — so a fallback is silent by design. Rather than trust that, each run
collects the warnings torch emits when one happens into `seed.nondeterministic_kernels`, and
the table names any row that has them: the claim is one the artifacts can contradict.

The one fallback that did happen is closed at the source. Attention's backward pass has no
deterministic implementation in the flash, memory-efficient or cuDNN kernels, so a
deterministic run restricts scaled-dot-product attention to the math backend, which does.
The usual objection — that materialising the attention matrix costs memory — does not apply
at this size: TimesFM 2.5 has `patch_length` 32, so a 512-step context is 16 tokens and the
matrix is 16×16 per head.

**The eleven rows in the table above predate that change** and were produced with the fused
kernels, which is why the notes under the table name the seven fine-tuned arms as reproducing
only to within floating-point accumulation order. Runs from this commit onward should record
no fallback at all; if one ever does, the artifact and the table will say so. Wall-clock and
memory figures are only comparable within one machine and one dtype.

**Artifacts.** Every run writes one JSON file holding the full config, the seed record, the
dataset and protocol provenance, the metrics with a per-series breakdown, the training
curves, the resource log and the environment. The table is a pure function of those files.

## Limitations

Read this section before believing the table.

- **Two datasets, one model family.** ETTh1 and NN5 Daily do not stand in for time series
  generally. Nothing here has been tested on high-frequency, intermittent, hierarchical or
  very long series, and no model outside TimesFM 2.5 has been run.
- **ETTh1's seven channels are not seven independent series.** They are correlated
  measurements from one transformer, treated as univariate series because that is what the
  model interface takes. Per-series metrics are therefore less independent than the series
  count suggests.
- **The evaluation windows are few.** Eight non-overlapping test windows per ETTh1 channel
  (56 forecast rows) and three per NN5 series (333 rows). Differences smaller than the
  spread across windows are not meaningful, and no confidence intervals are computed.
- **One seed per arm.** Results do not separate the effect of the method from run-to-run
  variance. A small gap between two arms may be seed noise.
- **The hyperparameters are documented starting points, not tuned values.** Learning rate,
  step budget, warmup and batch size were not swept. If a LoRA arm fails to beat zero-shot,
  that is evidence about these hyperparameters at least as much as about LoRA.
- **This is not the LTSF leaderboard protocol.** Published ETTh1 tables generally slide a
  stride-1 window over a 20% test split. These numbers are internally comparable across arms
  and are *not* comparable to those tables.
- **Only one adapter target set was tested.** Attention plus MLP projections, all layers.
  Adapting fewer layers, or the embedding, or the output heads, is untested.
- **The rank ablation holds alpha/rank at 2.0** so that capacity is the only thing varying.
  The more common convention of fixing alpha instead would change capacity and update scale
  together and would produce a different curve.
- **Horizons are single-decode only.** TimesFM 2.5 emits 128 steps per forward pass, and the
  horizons here (96 and 56) fit inside that. Autoregressive rollout to longer horizons is
  not implemented, so nothing beyond 128 steps has been measured.
- **NN5's missing values were imputed upstream** by the Monash authors, and those imputed
  points are still scored here.
- **A workaround sits in the training loss.** Upstream's `TimesFm2_5ModelForPrediction`
  computes its `future_values` loss against misaligned quantile heads (see below), so this
  repo computes its own. If upstream fixes that differently, fine-tuned numbers may move.
- **Out of scope entirely:** multi-GPU and distributed training, quantized training,
  Moirai/Chronos, and any hyperparameter search.

### An upstream bug worth knowing about

`TimesFm2_5ModelForPrediction.forward` returns `mse + quantile_loss` when handed
`future_values`, and that quantile term is misaligned. It drops the `decode_index` column
from the `[point, q_0.1 … q_0.9]` output and then zips what remains against
`config.quantiles` positionally, so level 0.1 is scored against the point head, 0.2 against
the 0.1 head, and so on — while the median head, the one this adapter reports as its point
forecast, is excluded from the pinball term entirely. The failure is silent: the loss still
falls and the model still trains, it just learns the wrong quantiles.

`aligned_forecast_loss` in `src/tsfm_peft/models/timesfm.py` replaces it. This has not been
reported upstream yet.

## Adding a dataset

1. Write a loader returning a `TimeSeriesDataset` (see `src/tsfm_peft/data/loaders.py`).
   Downloads go through `data/download.py`, which pins the file by SHA-256 so an upstream
   edit fails loudly instead of silently changing results.
2. Add a `DatasetSpec` to `DATASETS` in `src/tsfm_peft/data/registry.py`, giving the
   frequency, the MASE seasonal lag, the license and the source URL.
3. Add a data config under `configs/data/` with the horizon, context length and window
   counts. Every arm of that dataset references this one file, which is what stops the
   horizon from drifting between the zero-shot row and the fine-tuned rows.
4. Add one experiment config per arm under `configs/experiments/`, each pointing at the
   data config from step 3.

The datasets table below and the results table both read from the registry, so a new entry
appears in the documentation without anything being written by hand.

## Adding a model

1. Subclass `ForecastModel` in `src/tsfm_peft/models/` and implement `_predict_batch`:
   raw contexts in, point and quantile forecasts in the same raw units out. If the model
   needs an external scaler rather than normalising internally, set
   `requires_external_scaling`.
2. To make it fine-tunable, subclass `FineTunableModel` instead and add `module` and
   `training_loss`. The trainer owns batching, the optimiser, the schedule and checkpoint
   selection; the adapter owns only the loss, because the loss depends on what space the
   model normalises in and which output head means what.
3. Add a `ModelSpec` to `MODELS` in `src/tsfm_peft/models/registry.py` with a pydantic
   options model. Config `options` blocks are validated against it, so a misspelled key
   fails when the config loads rather than after an hour on a GPU.
4. Add experiment configs. No evaluation code changes.

## Datasets

Downloaded on first use into `~/.cache/tsfm_peft` (override with `TSFM_PEFT_CACHE`) and
pinned by SHA-256. No data is committed to this repository.

| Dataset | Content | Horizon | License | Source |
|---|---|---|---|---|
| `etth1` | Electricity Transformer Temperature, hourly, 7 channels x 17420 steps, each channel treated as an independent univariate series | 96 | CC BY-ND 4.0 | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| `nn5_daily` | Daily ATM cash withdrawals, 111 series x 791 steps, Monash "without missing values" variant | 56 | CC BY 4.0 | [Zenodo 4656117](https://zenodo.org/records/4656117) |

ETTh1 is CC BY-ND: it is downloaded at runtime and never redistributed or shipped in
modified form. NN5's missing values were imputed upstream by the Monash authors (median of
the same weekday); those imputed points are still scored here.

The model checkpoint is `google/timesfm-2.5-200m-transformers` (Apache-2.0). A published
number should name the revision it was produced against; the configs carry a `revision`
field for that purpose.

## Development

```bash
uv sync --extra dev                  # core + test tooling, no torch
uv sync --extra dev --extra models   # adds torch/transformers/peft
uv run pytest                        # the torch-free suite runs in seconds
uv run ruff check . && uv run ruff format --check .
uv run tsfm-peft table --configs configs/experiments --check
```

Tests are marked `slow`, `gpu` and `network`; CI deselects all three. The CPU job runs the
full end-to-end pipeline against a generated dataset and a randomly initialised tiny
checkpoint, so the path a real run takes is exercised on every push without downloading
anything. Those fixture runs are excluded from the results table by construction.

## License

Apache-2.0, see [LICENSE](LICENSE). Dataset licenses are listed in the datasets section
above; the data itself is never redistributed here.
