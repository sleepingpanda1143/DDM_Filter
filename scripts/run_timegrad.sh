#!/usr/bin/env bash
# TimeGrad 训练 / 测试入口。默认激活 conda 环境 verltool5090（可用环境变量覆盖）。
# 脚本位于 scripts/，避免与 Python 标准库模块 ``code`` 同名目录冲突。
#
# 用法:
#   ./scripts/run_timegrad.sh                    # 默认: smoke + 三个预设数据集
#   ./scripts/run_timegrad.sh smoke --encoder-type transformer
#   ./scripts/run_timegrad.sh train --preset radar_viz_debug --max-epochs 30
#   TIMEGRAD_CONDA_ENV=ddm ./scripts/run_timegrad.sh smoke
#
# 双卡（例如 2×5090）并行跑两条命令时，用 TIMEGRAD_CUDA_DEVICE 绑定物理 GPU，
# 使 Lightning 与 GluonTS PyTorchPredictor 都落在同一张卡上（进程内逻辑 cuda:0）:
#   TIMEGRAD_CUDA_DEVICE=0 ./scripts/run_timegrad.sh train ... --output-dir .../run_A
#   TIMEGRAD_CUDA_DEVICE=1 ./scripts/run_timegrad.sh train ... --output-dir .../run_B
#
# 默认 train 的 --batch-size 为 128（可用更小值避免 OOM）；短序列 + 小模型时 GPU 利用率仍可能偏低，
# 增大 batch 与 shuffle_buffer 可摊薄 kernel 启动与数据管道开销。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${TIMEGRAD_CONDA_ENV:=verltool5090}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "WARN: conda.sh not found; assuming current shell already has env ${TIMEGRAD_CONDA_ENV}" >&2
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${TIMEGRAD_CONDA_ENV}"
fi

if [[ -n "${TIMEGRAD_CUDA_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${TIMEGRAD_CUDA_DEVICE}"
fi

if [[ $# -ge 1 && ( "$1" == "smoke" || "$1" == "train" ) ]]; then
  set -- --mode "$1" "${@:2}"
fi

exec python "${SCRIPT_DIR}/timegrad_train_eval.py" "$@"
