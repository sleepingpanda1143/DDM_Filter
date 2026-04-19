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

if [[ $# -ge 1 && ( "$1" == "smoke" || "$1" == "train" ) ]]; then
  set -- --mode "$1" "${@:2}"
fi

exec python "${SCRIPT_DIR}/timegrad_train_eval.py" "$@"
