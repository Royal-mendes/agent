#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${QWEN_VL_ENV:-/root/autodl-tmp/envs/qwen_vl}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
PYTHON_VERSION="${QWEN_VL_PYTHON_VERSION:-3.11}"
PIP_INDEX_URL="${QWEN_VL_PIP_INDEX_URL:-https://pypi.org/simple}"
VLLM_VERSION="${QWEN_VL_VLLM_VERSION:-0.10.2}"
TRANSFORMERS_VERSION="${QWEN_VL_TRANSFORMERS_VERSION:-4.56.2}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "conda_not_found=$CONDA_BIN" >&2
  exit 2
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION"
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

python -m pip install -U -i "$PIP_INDEX_URL" pip setuptools wheel
python -m pip install -U -i "$PIP_INDEX_URL" --only-binary=:all: \
  "vllm==$VLLM_VERSION" "qwen-vl-utils[decord]" "openai"
python -m pip install -U -i "$PIP_INDEX_URL" "transformers==$TRANSFORMERS_VERSION"

python - <<'PY'
import importlib.metadata as md
for name in ("vllm", "torch", "transformers", "qwen-vl-utils", "openai"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not_installed")
PY

echo "qwen_vl_env=$ENV_PREFIX"
