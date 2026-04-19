#!/usr/bin/env bash
# Two-GPU A/B: GluonTS MeanScaler (per-channel over context time) vs fixed JSON z-score per channel.
# Run from repo root: ./scripts/run_timegrad_norm_ablation_2gpu.sh
#
# GPU0: --timegrad-scaling mean  (no metadata z-score in loader)
# GPU1: --timegrad-scaling none --dataset-target-metadata-zscore
#
# Same GPU, two concurrent jobs: only if both fit in VRAM — halve --batch-size per job or run sequentially.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

COMMON=(
  train
  --preset radar_train_20min_origin_delta
  --encoder-type transformer
  --hidden-size 128 --num-layers 3
  --transformer-nhead 8 --transformer-dim-feedforward 512
  --dropout-rate 0.1 --lr 0.002
  --lags-preset metadata
  --max-epochs 40 --num-batches-per-epoch 150 --batch-size 256
  --accumulate-grad-batches 2
  --num-inference-steps 80 --eval-num-samples 80
  --num-parallel-samples 100
)

echo "Starting GPU0: GluonTS mean scaling only -> dataset/exp_ablate_2gpu/gpu0_tg_mean"
TIMEGRAD_CUDA_DEVICE=0 ./scripts/run_timegrad.sh "${COMMON[@]}" \
  --timegrad-scaling mean \
  --output-dir dataset/exp_ablate_2gpu/gpu0_tg_mean &
PID0=$!

echo "Starting GPU1: metadata z-score loader + scaling none -> dataset/exp_ablate_2gpu/gpu1_md_zscore"
TIMEGRAD_CUDA_DEVICE=1 ./scripts/run_timegrad.sh "${COMMON[@]}" \
  --timegrad-scaling none \
  --dataset-target-metadata-zscore \
  --output-dir dataset/exp_ablate_2gpu/gpu1_md_zscore &
PID1=$!

wait "${PID0}"
echo "GPU0 job finished."
wait "${PID1}"
echo "GPU1 job finished."
echo "Done."
