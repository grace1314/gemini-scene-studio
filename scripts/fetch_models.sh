#!/usr/bin/env bash
set -euo pipefail

studio_root="$(cd "$(dirname "$0")/.." && pwd)"
model_dir="$studio_root/models"
torch_cache="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
mkdir -p "$model_dir" "$torch_cache"

fetch_checked() {
  local url="$1" destination="$2" expected_sha256="${3:-}"
  if [ ! -s "$destination" ]; then
    curl -fL --retry 3 --connect-timeout 20 --max-time 7200 "$url" -o "$destination.partial"
    mv "$destination.partial" "$destination"
  fi
  if [ -n "$expected_sha256" ]; then
    printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --check --status || {
      echo "Checksum mismatch: $destination" >&2
      exit 1
    }
  fi
}

# Official RVC-pretrain checkpoint. It is about 1.44 GB and is intentionally
# excluded from Git. The checksum was recorded from the public release used by
# the verified example run.
fetch_checked \
  https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_rvc_pretrain.pth \
  "$model_dir/bridge_rvc_pretrain.pth" \
  4cb8d0395e6733a92f1d998077dc08b4e6b17a1b4f9cd814593cf8cf8c8842f9

# Pre-warm the exact files requested by ALIKED + LightGlue so later runs do not
# need network access. torch.hub recognizes these filenames in its cache.
fetch_checked \
  https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth \
  "$torch_cache/aliked-n16.pth"
fetch_checked \
  https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth \
  "$torch_cache/aliked_lightglue_v0-1_arxiv.pth"

echo "Model cache ready. BridgeDepth and ALIKED/LightGlue can now run offline."
