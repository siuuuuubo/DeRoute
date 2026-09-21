#!/usr/bin/env bash
set -euo pipefail
DAGLAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DAGLAB_ROOT/.tmp"
export TMPDIR="$DAGLAB_ROOT/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$DAGLAB_ROOT/.cache/huggingface"
export TORCH_HOME="$DAGLAB_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$DAGLAB_ROOT/.cache/triton"
export XDG_CACHE_HOME="$DAGLAB_ROOT/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$DAGLAB_ROOT"
trap 'rmdir "$DAGLAB_ROOT/.tmp" 2>/dev/null || true' EXIT
# 运行环境：默认 rag310（Python 3.10，含 torch，可跑 small.provider=local_transformers）。
# 纯 API 模式不需要 torch；如需换环境用 DEROUTE_CONDA_ENV 覆盖。
conda run --no-capture-output -n "${DEROUTE_CONDA_ENV:-rag310}" python "$@"
