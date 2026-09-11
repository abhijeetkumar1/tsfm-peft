#!/usr/bin/env bash
# Run every benchmark arm, one process per visible GPU.
#
# The arms are independent, and one arm uses a fraction of a card, so the parallelism worth
# having is across arms rather than inside one. Each process is pinned to a single device
# with CUDA_VISIBLE_DEVICES and runs the same `tsfm-peft run` a serial invocation would; no
# distributed training is involved and nothing in the package knows this script exists.
#
#   scripts/run_all.sh                # one process per visible GPU (serial if there is one)
#   TSFM_PEFT_PARALLEL=1 scripts/run_all.sh   # force serial, e.g. to measure timings cleanly
#
# An arm that already has an artifact is skipped, so rerunning after a lost session picks up
# where it stopped. A failing arm does not stop the others; it is reported at the end.
#
# COST COLUMNS: concurrent arms share CPU, disk and host memory, so their train-time and
# peak-memory figures include contention that a serial run does not have. That is recorded in
# each artifact (environment.scheduling.concurrent_runs) and the generated table says which
# rows it applies to. Run with TSFM_PEFT_PARALLEL=1 when the cost columns are the point.
set -uo pipefail

cd "$(dirname "$0")/.."

# Ordered so that a session cut short still leaves something publishable: the baselines and
# zero-shot rows are what every fine-tuned arm is compared against, and the rank ablation is
# the part the table is complete without.
ARMS=(
  etth1-seasonal-naive
  nn5_daily-seasonal-naive
  etth1-timesfm-zeroshot
  nn5_daily-timesfm-zeroshot
  etth1-timesfm-lora
  etth1-timesfm-dora
  nn5_daily-timesfm-lora
  nn5_daily-timesfm-dora
  etth1-timesfm-lora-r4
  etth1-timesfm-lora-r8
  etth1-timesfm-lora-r32
)

# Synthetic arms are CI smoke fixtures against a randomly initialised checkpoint and are
# excluded from the table by construction, so running them here would prove nothing. Any
# other config on disk that is missing from ARMS is a mistake: fail rather than quietly
# leave a row out of the table.
missing=()
for path in configs/experiments/*.yaml; do
  name=$(basename "$path" .yaml)
  case "$name" in synthetic-*) continue ;; esac
  case " ${ARMS[*]} " in *" $name "*) continue ;; esac
  missing+=("$name")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "configs not listed in ARMS: ${missing[*]}" >&2
  exit 1
fi

gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
[ "$gpus" -gt 0 ] 2>/dev/null || gpus=0
parallel=${TSFM_PEFT_PARALLEL:-$gpus}
[ "$parallel" -ge 1 ] 2>/dev/null || parallel=1

results=${TSFM_PEFT_RESULTS:-results}
logs=${TSFM_PEFT_LOGS:-logs}
mkdir -p "$results" "$logs"

echo "$gpus GPU(s) visible; running $parallel arm(s) at a time"
[ "$parallel" -gt 1 ] && echo "cost columns will be flagged as contended in the table"

pending=()
for name in "${ARMS[@]}"; do
  if [ -f "$results/$name.json" ]; then
    echo "== $name: already done, skipping"
  else
    pending+=("$name")
  fi
done

index=0
for name in "${pending[@]}"; do
  # Block until a slot frees. Polled rather than `wait -n`, which bash 3.2 does not have --
  # there it is not an error but a spin loop, and an arm takes minutes, so a one-second poll
  # costs nothing and behaves the same everywhere.
  while [ "$(jobs -rp | wc -l)" -ge "$parallel" ]; do sleep 1; done

  if [ "$gpus" -gt 0 ]; then
    device=$((index % gpus))
    echo "== $name: running on GPU $device (log: $logs/$name.log)"
    CUDA_VISIBLE_DEVICES=$device TSFM_PEFT_CONCURRENCY=$parallel \
      tsfm-peft run "configs/experiments/$name.yaml" > "$logs/$name.log" 2>&1 &
  else
    echo "== $name: running on CPU (log: $logs/$name.log)"
    TSFM_PEFT_CONCURRENCY=$parallel \
      tsfm-peft run "configs/experiments/$name.yaml" > "$logs/$name.log" 2>&1 &
  fi
  index=$((index + 1))
done
wait

failed=()
for name in "${ARMS[@]}"; do
  [ -f "$results/$name.json" ] || failed+=("$name")
done

echo
echo "complete: $(( ${#ARMS[@]} - ${#failed[@]} ))/${#ARMS[@]}"
if [ ${#failed[@]} -gt 0 ]; then
  echo "no artifact for: ${failed[*]} (see $logs/<arm>.log; rerun this script to retry)"
fi

tsfm-peft table --configs configs/experiments --write
