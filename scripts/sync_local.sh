#!/usr/bin/env bash
# Development sync into this user's requested project directory. No deletions.
set -euo pipefail
studio_source="$(cd "$(dirname "$0")/.." && pwd)"
studio_destination="${GEMINI_STUDIO_DEPLOY_DIR:-$HOME/Tars_Project/gemini-scene-studio}"
if [ "$studio_source" != "$studio_destination" ]; then
  # Runtime data, downloaded models, third-party source/binaries and build output
  # have independent lifecycles.  Synchronize only the maintained project tree;
  # in particular, never replace a populated checkpoint directory with a
  # development-machine symlink.
  rsync -a \
    --exclude=/.venv \
    --exclude=/build \
    --exclude=/data \
    --exclude=/models \
    --exclude=/3rdparty \
    --exclude=/.git \
    --exclude=__pycache__ \
    "$studio_source/" "$studio_destination/"
  mkdir -p "$studio_destination/data" "$studio_destination/models"
  cp "$studio_source/data/.gitkeep" "$studio_destination/data/.gitkeep"
  cp "$studio_source/models/README.md" "$studio_destination/models/README.md"
fi
bash "$studio_destination/scripts/build.sh"
if [ ! -d "$studio_destination/.git" ]; then git -C "$studio_destination" init -b main; fi
