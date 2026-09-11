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
conda run --no-capture-output -n agent python "$@"
