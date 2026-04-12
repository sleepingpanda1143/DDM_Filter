#!/usr/bin/env bash
# TimeGrad radar experiments for six motion datasets on two GPUs (three waves × 2 jobs).
# Requires: datasets from Radar/generate_six_motion_datasets.sh
# Requires: conda env verltool5090 (override with TIMEGRAD_PY).
#
#   KALMAN_ROOT=$HOME/kalman_net ./run_six_motion_timegrad.sh
#
set -euo pipefail

KALMAN_ROOT="${KALMAN_ROOT:-$HOME/kalman_net}"
DDM="$KALMAN_ROOT/DDM_Timeseries_Forecast"
# Default: your verltool5090 interpreter (avoid relying on activated conda).
TIMEGRAD_PY="${TIMEGRAD_PY:-$HOME/miniconda3/envs/verltool5090/bin/python}"

if [[ ! -x "$TIMEGRAD_PY" ]]; then
  echo "Set TIMEGRAD_PY to your env's python (e.g. verltool5090)." >&2
  echo "Tried: $TIMEGRAD_PY" >&2
  exit 1
fi

export PYTHONPATH="$DDM"

COMMON=(
  --device cuda --freq 1s
  --context-length 20 --prediction-length 1
  --position-origin first --state-repr delta --no-timegrad-scaling
  --denorm-metrics --save-forecast-samples --suppress-warnings
  --epochs 80 --num-batches-per-epoch 150 --batch-size 512
  --max-train-trajectories 0 --max-test-trajectories 0
  --learning-rate 1e-3 --diff-steps 200 --num-samples 100
  --num-layers 3 --num-cells 128 --num-workers 6
)

run_one() {
  local gpu=$1 npz_stem=$2 out_slug=$3
  echo ">>> GPU $gpu  $out_slug"
  # generate_trajectory_data.py saves as {prefix}_data.npz (not {prefix}.npz)
  CUDA_VISIBLE_DEVICES="$gpu" "$TIMEGRAD_PY" "$DDM/code/radar_timegrad_experiment.py" \
    --input-npz "$KALMAN_ROOT/Radar/data/${npz_stem}.npz" \
    --output-dir "$DDM/dataset/${out_slug}" \
    "${COMMON[@]}"
}

echo "Wave 1: CV (GPU0) + Singer (GPU1)"
run_one 0 trajectory_cv_only_data exp_motion_cv_gpu0 &
pid1=$!
run_one 1 trajectory_singer_only_data exp_motion_singer_gpu1 &
pid2=$!
wait "$pid1" "$pid2"

echo "Wave 2: CA (GPU0) + Jerk (GPU1)"
run_one 0 trajectory_ca_only_data exp_motion_ca_gpu0 &
pid1=$!
run_one 1 trajectory_jerk_only_data exp_motion_jerk_gpu1 &
pid2=$!
wait "$pid1" "$pid2"

echo "Wave 3: CT (GPU0) + VRW (GPU1)"
run_one 0 trajectory_ct_only_data exp_motion_ct_gpu0 &
pid1=$!
run_one 1 trajectory_vrw_only_data exp_motion_vrw_gpu1 &
pid2=$!
wait "$pid1" "$pid2"

echo "All six runs finished."
