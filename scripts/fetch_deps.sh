#!/usr/bin/env bash
set -euo pipefail
studio_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$studio_root/3rdparty/include/nlohmann" "$studio_root/3rdparty/licenses" "$studio_root/3rdparty/downloads" "$studio_root/3rdparty/three"
fetch() {
  if [ ! -s "$2" ]; then
    curl -fL --retry 2 --connect-timeout 20 --max-time 300 "$1" -o "$2.partial"
    mv "$2.partial" "$2"
  fi
}
clone_at() {
  local url="$1" revision="$2" destination="$3"
  if [ ! -d "$destination/.git" ]; then
    if [ -e "$destination" ]; then
      echo "Refusing to replace existing non-Git directory: $destination" >&2
      echo "Move it aside explicitly, then rerun this script." >&2
      return 1
    fi
    git clone --filter=blob:none "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
}
fetch https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.20.0/httplib.h "$studio_root/3rdparty/include/httplib.h"
fetch https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.20.0/LICENSE "$studio_root/3rdparty/licenses/cpp-httplib.txt"
fetch https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp "$studio_root/3rdparty/include/nlohmann/json.hpp"
fetch https://raw.githubusercontent.com/nlohmann/json/v3.11.3/LICENSE.MIT "$studio_root/3rdparty/licenses/nlohmann-json.txt"
fetch https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz "$studio_root/3rdparty/downloads/open3d-0.19.0.tar.xz"
if [ ! -d "$studio_root/3rdparty/open3d" ]; then
  mkdir -p "$studio_root/3rdparty/open3d"
  tar -xJf "$studio_root/3rdparty/downloads/open3d-0.19.0.tar.xz" -C "$studio_root/3rdparty/open3d" --strip-components=1
fi
fetch https://unpkg.com/three@0.170.0/build/three.module.js "$studio_root/3rdparty/three/three.module.js"
fetch https://unpkg.com/three@0.170.0/examples/jsm/controls/OrbitControls.js "$studio_root/3rdparty/three/OrbitControls.js"
fetch https://unpkg.com/three@0.170.0/LICENSE "$studio_root/3rdparty/licenses/three.txt"
clone_at https://github.com/aeolusguan/BridgeDepth.git 9a231601e6981d3536fa6a78ebaccebc1edb99a4 "$studio_root/3rdparty/BridgeDepth"
clone_at https://github.com/cvg/LightGlue.git 5a650c444718caffc2669dc0094cde908f42dd73 "$studio_root/3rdparty/LightGlue"

echo "Dependencies fetched. Run scripts/fetch_models.sh once before the first GPU reconstruction."
