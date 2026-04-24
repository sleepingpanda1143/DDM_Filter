#!/usr/bin/env bash
set -euo pipefail

# Repo root (this script lives in code/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Always use verltool5090 env (not base). Override only if you must: PYTHON=/other/bin/python
PY="${PYTHON:-/home/xiongmaoren/miniconda3/envs/verltool5090/bin/python}"

ROOT="${1:-dataset/exp_blair_5090_meas_ablation_20260419_232945}"
NS="${NUM_SAMPLES:-100}"
MS="${MAX_SERIES:-0}"
BW="${BATCH_WINDOWS:-1024}"
PBS="${PREDICTOR_BATCH_SIZE:-512}"
LEW="${LOG_EVERY_WINDOWS:-1000}"
PAR="${PARALLEL:-1}"
MEAS_NPZ="${MEASUREMENT_NPZ:-}"
KF_CV_Q="${KF_CV_Q_ACC:-10.0}"
KF_CA_Q="${KF_CA_Q_JERK:-1.0}"
KF_R_VAR="${KF_R_VAR:--1}"
PREDICTOR_DEVICE="${PREDICTOR_DEVICE:-cuda}"

# Per-run-dir GPU balance: split test series across GPUs (shard 0..N-1), then merge.
# Set GPU_SHARDS=1 to restore "one full eval per GPU" (baseline on GPU0 + meas on GPU1 when PAR=1).
GPU_SHARDS="${GPU_SHARDS:-2}"
NUM_GPUS="${NUM_GPUS:-2}"

export PYTHONPATH=.

BASE_ARGS=(
  --num-samples "$NS"
  --max-series "$MS"
  --batch-windows "$BW"
  --log-every-windows "$LEW"
  --kf-cv-q-acc "$KF_CV_Q"
  --kf-ca-q-jerk "$KF_CA_Q"
  --kf-r-var "$KF_R_VAR"
  --predictor-device "$PREDICTOR_DEVICE"
  --predictor-batch-size "$PBS"
)
if [[ -n "$MEAS_NPZ" ]]; then
  BASE_ARGS+=(--measurement-npz "$MEAS_NPZ")
fi

run_sharded_one_subdir() {
  local SUB="$1"
  local RD="$ROOT/$SUB"
  mkdir -p "$RD"
  local LOG="$RD/filter_eval.log"
  {
    echo "=== sharded eval GPU_SHARDS=$GPU_SHARDS NUM_GPUS=$NUM_GPUS subdir=$SUB ==="
    local -a pids=()
    local k g
    for ((k = 0; k < GPU_SHARDS; k++)); do
      g=$((k % NUM_GPUS))
      CUDA_VISIBLE_DEVICES="$g" "$PY" code/eval_timegrad_sliding_window.py --run-dir "$RD" "${BASE_ARGS[@]}" \
        --shard-id "$k" --num-shards "$GPU_SHARDS" &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || exit 1; done
    "$PY" code/eval_timegrad_sliding_window.py --merge-shards --run-dir "$RD" --num-shards "$GPU_SHARDS"
  } &> "$LOG"
}

if [[ "$GPU_SHARDS" -ge 2 ]]; then
  run_sharded_one_subdir baseline_no_meas
  run_sharded_one_subdir meas_conditioning
elif [[ "$PAR" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" code/eval_timegrad_sliding_window.py --run-dir "$ROOT/baseline_no_meas" "${BASE_ARGS[@]}" > "$ROOT/baseline_no_meas/filter_eval.log" 2>&1 &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" code/eval_timegrad_sliding_window.py --run-dir "$ROOT/meas_conditioning" "${BASE_ARGS[@]}" > "$ROOT/meas_conditioning/filter_eval.log" 2>&1 &
  PID1=$!

  E0=0; E1=0
  wait "$PID0" || E0=$?
  wait "$PID1" || E1=$?
  if [[ "$E0" -ne 0 || "$E1" -ne 0 ]]; then
    echo "Filter eval failed (baseline=$E0, meas=$E1). Check logs:"
    echo "  $ROOT/baseline_no_meas/filter_eval.log"
    echo "  $ROOT/meas_conditioning/filter_eval.log"
    exit 1
  fi
else
  "$PY" code/eval_timegrad_sliding_window.py --run-dir "$ROOT/baseline_no_meas" "${BASE_ARGS[@]}"
  "$PY" code/eval_timegrad_sliding_window.py --run-dir "$ROOT/meas_conditioning" "${BASE_ARGS[@]}"
fi

echo "Done. Reports:"
echo "  $ROOT/baseline_no_meas/filter_eval_report.json"
echo "  $ROOT/meas_conditioning/filter_eval_report.json"
echo "Done. Plots:"
echo "  $ROOT/baseline_no_meas/filter_eval_timeseries.png"
echo "  $ROOT/baseline_no_meas/filter_eval_timeseries_xyz.png"
echo "  $ROOT/meas_conditioning/filter_eval_timeseries.png"
echo "  $ROOT/meas_conditioning/filter_eval_timeseries_xyz.png"
echo "Logs:"
echo "  $ROOT/baseline_no_meas/filter_eval.log"
echo "  $ROOT/meas_conditioning/filter_eval.log"
