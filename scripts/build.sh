#!/usr/bin/env bash
set -euo pipefail
studio_root="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$studio_root" -B "$studio_root/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$studio_root/build" -j 2
ctest --test-dir "$studio_root/build" --output-on-failure
