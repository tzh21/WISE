#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-flagscale-robo}"
PYTHON_BIN="${PYTHON_BIN:-/share/project/zhangqq/envs_conda/flagscale-robo/bin/python}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_wmdec_wise.py" "$@"
