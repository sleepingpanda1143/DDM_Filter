#!/usr/bin/env bash
# Radar TimeGrad experiment pipeline (dual RTX 5090 friendly).
#
# Order:
#   0) One-time: ensure default trajectory npz exists (see Radar/generate_trajectory_data.py).
#   1) GPU smoke test (minutes).
#   2) Pre-generate low-scale npz if missing (avoids parallel jobs racing on generation).
#   3) Full comparison: either sequential on one GPU, or parallel across two GPUs.
#
# Usage:
#   chmod +x run_radar_pipeline_2gpu.sh
#   KALMAN_ROOT=$HOME/kalman_net ./run_radar_pipeline_2gpu.sh smoke
#   KALMAN_ROOT=$HOME/kalman_net ./run_radar_pipeline_2gpu.sh prepare_low_npz
#   KALMAN_ROOT=$HOME/kalman_net ./run_radar_pipeline_2gpu.sh compare_all
#   KALMAN_ROOT=$HOME/kalman_net ./run_radar_pipeline_2gpu.sh compare_parallel
#
set -euo pipefail

KALMAN_ROOT="${KALMAN_ROOT:-$HOME/kalman_net}"
DDM="$KALMAN_ROOT/DDM_Timeseries_Forecast"
RADAR="$KALMAN_ROOT/Radar"
CODE="$DDM/code"
PY="${PYTHON:-python}"
DEFAULT_NPZ="$RADAR/data/trajectory_data.npz"
LOW_NPZ="$RADAR/data/trajectory_low_scale_data.npz"

export PYTHONPATH="$DDM${PYTHONPATH:+:$PYTHONPATH}"

die() { echo "ERROR: $*" >&2; exit 1; }

check_default_npz() {
  [[ -f "$DEFAULT_NPZ" ]] || die "Missing $DEFAULT_NPZ — generate with Radar/generate_trajectory_data.py first."
}

run_smoke() {
  check_default_npz
  echo "=== Smoke: 1 GPU, short epochs ==="
  "$PY" "$CODE/radar_timegrad_experiment.py" \
    --input-npz "$DEFAULT_NPZ" \
    --output-dir "$DDM/dataset/radar_smoke_gpu" \
    --device cuda \
    --epochs 1 \
    --num-batches-per-epoch 5 \
    --num-samples 20 \
    --max-train-trajectories 50 \
    --max-test-trajectories 10 \
    --no-timegrad-scaling \
    --suppress-warnings
  echo "OK: see $DDM/dataset/radar_smoke_gpu/metrics.json"
}

run_prepare_low_npz() {
  [[ -f "$LOW_NPZ" ]] && echo "Already exists: $LOW_NPZ" && return 0
  echo "=== Generating low distance/speed dataset ==="
  (cd "$RADAR" && exec "$PY" generate_trajectory_data.py \
    --num_trajectories 2000 \
    --min_length 80 \
    --max_length 220 \
    --output_dir "$RADAR/data" \
    --prefix trajectory_low_scale \
    --seed 43 \
    --init-x-range -2500 2500 \
    --init-y-range -2500 2500 \
    --cv-speed-range 5 55)
  [[ -f "$LOW_NPZ" ]] || die "Expected $LOW_NPZ after generation."
  echo "OK: $LOW_NPZ"
}

run_compare_all() {
  check_default_npz
  echo "=== Full matrix (sequential, single GPU, long) ==="
  "$PY" "$CODE/run_radar_compare_experiments.py" --device cuda --python "$PY" \
    --input-npz "$DEFAULT_NPZ" \
    --out-root "$DDM/dataset/radar_compare_runs"
  echo "OK: $DDM/dataset/radar_compare_runs/comparison_report.md"
}

run_compare_parallel() {
  check_default_npz
  run_prepare_low_npz
  echo "=== Parallel split: GPU0 runs 01–03, GPU1 runs 04–05 (different --out-root) ==="
  local LOGDIR="$DDM/dataset/radar_compare_parallel_logs"
  mkdir -p "$LOGDIR"
  (
    CUDA_VISIBLE_DEVICES=0 exec "$PY" "$CODE/run_radar_compare_experiments.py" --device cuda --python "$PY" \
      --input-npz "$DEFAULT_NPZ" \
      --out-root "$DDM/dataset/radar_compare_runs_gpu0" \
      --only "01_baseline_short,02_long_train_double_scale,03_long_train_zscore_only" \
      >"$LOGDIR/gpu0.log" 2>&1
  ) &
  PID0=$!
  (
    CUDA_VISIBLE_DEVICES=1 exec "$PY" "$CODE/run_radar_compare_experiments.py" --device cuda --python "$PY" \
      --input-npz "$DEFAULT_NPZ" \
      --out-root "$DDM/dataset/radar_compare_runs_gpu1" \
      --only "04_long_train_wider_rnn,05_low_distance_speed_npz" \
      >"$LOGDIR/gpu1.log" 2>&1
  ) &
  PID1=$!
  E0=0; wait "$PID0" || E0=$?
  E1=0; wait "$PID1" || E1=$?
  echo "Logs: $LOGDIR/gpu0.log $LOGDIR/gpu1.log"
  if [[ "$E0" -ne 0 || "$E1" -ne 0 ]]; then
    echo "One or both parallel jobs failed (exit $E0, $E1). See logs." >&2
    exit 1
  fi
  echo "Summaries: $DDM/dataset/radar_compare_runs_gpu0/comparison_summary.json"
  echo "           $DDM/dataset/radar_compare_runs_gpu1/comparison_summary.json"
  echo "Merge manually for one table, or run:"
  echo "  \"$PY\" -c \"import json, pathlib; r=json.loads(pathlib.Path('$DDM/dataset/radar_compare_runs_gpu0/comparison_summary.json').read_text())+json.loads(pathlib.Path('$DDM/dataset/radar_compare_runs_gpu1/comparison_summary.json').read_text()); pathlib.Path('$DDM/dataset/radar_compare_runs_merged.json').write_text(json.dumps(r, indent=2, ensure_ascii=False))\""
}

case "${1:-}" in
  smoke) run_smoke ;;
  prepare_low_npz) run_prepare_low_npz ;;
  compare_all) run_compare_all ;;
  compare_parallel) run_compare_parallel ;;
  *)
    echo "Usage: $0 [smoke|prepare_low_npz|compare_all|compare_parallel]" >&2
    exit 1
    ;;
esac
