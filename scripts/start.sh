#!/usr/bin/env bash
set -euo pipefail
studio_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$studio_root"
export GEMINI_STUDIO_ROOT="$studio_root"
export PYTHONPATH="$studio_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-$HOME/miniconda3/envs/bridgedepth}"
export PATH="$studio_root/.venv/bin:$CUDA_HOME/bin:$PATH"
if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="$studio_root/3rdparty/torch_extensions"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$studio_root/.venv/bin/python" scripts/doctor.py
exec "$studio_root/build/studio_server" "$studio_root" "${1:-8787}"
