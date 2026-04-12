#!/usr/bin/env bash
# ~20 分钟 GPU 训练（5090 量级粗估：12×45≈540 optimizer steps + 评估；实际随机器负载浮动）
#
# 用法（在仓库外先 export PYTHONPATH 或下面自动设）:
#   chmod +x train_radar_20min.sh
#   KALMAN_ROOT=$HOME/kalman_net ./train_radar_20min.sh main
#   KALMAN_ROOT=$HOME/kalman_net ./train_radar_20min.sh lowscale
#   KALMAN_ROOT=$HOME/kalman_net ./train_radar_20min.sh baseline_abs
#
set -euo pipefail

KALMAN_ROOT="${KALMAN_ROOT:-$HOME/kalman_net}"
DDM="$KALMAN_ROOT/DDM_Timeseries_Forecast"
CODE="$DDM/code"
PY="${PYTHON:-python}"

export PYTHONPATH="$DDM${PYTHONPATH:+:$PYTHONPATH}"

# 可调：略增/减 epochs 或 num-batches 以卡 20min
EPOCHS=50
NUM_BATCHES=100
BATCH=1024
NUM_SAMPLES=50
MAX_TRAIN=800
MAX_TEST=120

common() {
  echo "Using: EPOCHS=$EPOCHS NUM_BATCHES=$NUM_BATCHES BATCH=$BATCH MAX_TRAIN=$MAX_TRAIN MAX_TEST=$MAX_TEST"
}

run_main() {
  common
  "$PY" "$CODE/radar_timegrad_experiment.py" \
    --input-npz "$KALMAN_ROOT/Radar/data/trajectory_data.npz" \
    --output-dir "$DDM/dataset/radar_train_20min_origin_delta" \
    --device cuda \
    --position-origin first \
    --state-repr delta \
    --no-timegrad-scaling \
    --denorm-metrics \
    --save-forecast-samples \
    --suppress-warnings \
    --epochs "$EPOCHS" \
    --num-batches-per-epoch "$NUM_BATCHES" \
    --batch-size "$BATCH" \
    --num-samples "$NUM_SAMPLES" \
    --max-train-trajectories "$MAX_TRAIN" \
    --max-test-trajectories "$MAX_TEST" \
    --learning-rate 1e-3 \
    --diff-steps 100
  echo "Done. Plots: $PY $CODE/visualize_radar_timegrad_run.py --run-dir $DDM/dataset/radar_train_20min_origin_delta"
}

run_lowscale() {
  common
  "$PY" "$CODE/radar_timegrad_experiment.py" \
    --input-npz "$KALMAN_ROOT/Radar/data/trajectory_low_scale_data.npz" \
    --output-dir "$DDM/dataset/radar_train_20min_lowscale_origin_delta" \
    --device cuda \
    --position-origin first \
    --state-repr delta \
    --no-timegrad-scaling \
    --denorm-metrics \
    --save-forecast-samples \
    --suppress-warnings \
    --epochs "$EPOCHS" \
    --num-batches-per-epoch "$NUM_BATCHES" \
    --batch-size "$BATCH" \
    --num-samples "$NUM_SAMPLES" \
    --max-train-trajectories "$MAX_TRAIN" \
    --max-test-trajectories "$MAX_TEST" \
    --learning-rate 1e-3 \
    --diff-steps 100
}

run_baseline_abs() {
  common
  "$PY" "$CODE/radar_timegrad_experiment.py" \
    --input-npz "$KALMAN_ROOT/Radar/data/trajectory_data.npz" \
    --output-dir "$DDM/dataset/radar_train_20min_absolute_zscore" \
    --device cuda \
    --position-origin none \
    --state-repr absolute \
    --no-timegrad-scaling \
    --denorm-metrics \
    --save-forecast-samples \
    --suppress-warnings \
    --epochs "$EPOCHS" \
    --num-batches-per-epoch "$NUM_BATCHES" \
    --batch-size "$BATCH" \
    --num-samples "$NUM_SAMPLES" \
    --max-train-trajectories "$MAX_TRAIN" \
    --max-test-trajectories "$MAX_TEST" \
    --learning-rate 1e-3 \
    --diff-steps 100
}

case "${1:-}" in
  main) run_main ;;
  lowscale) run_lowscale ;;
  baseline_abs) run_baseline_abs ;;
  *)
    echo "Usage: $0 [main|lowscale|baseline_abs]" >&2
    exit 1
    ;;
esac
