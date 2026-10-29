#!/usr/bin/env bash
# Install and run the privacy-redacted 20-frame example through the local server.
set -euo pipefail

studio_root="$(cd "$(dirname "$0")/.." && pwd)"
scene_id=a11ce5afe001
scene_source="$studio_root/examples/desk-scene/scene"
scene_destination="$studio_root/data/scenes/$scene_id"
studio_port="${1:-8787}"
base_url="http://127.0.0.1:$studio_port"
skip_train="${GEMINI_DEMO_SKIP_TRAIN:-0}"

for dependency in curl jq; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done

if [ ! -f "$scene_source/manifest.json" ]; then
  echo "public example is missing: $scene_source" >&2
  exit 1
fi

if [ ! -e "$scene_destination" ]; then
  mkdir -p "$studio_root/data/scenes"
  cp -a "$scene_source" "$scene_destination"
  echo "installed privacy-redacted scene: $scene_destination"
elif [ ! -f "$scene_destination/manifest.json" ]; then
  echo "destination exists but is not a scene: $scene_destination" >&2
  exit 1
else
  echo "using existing scene: $scene_destination"
fi

curl -fsS "$base_url/api/health" >/dev/null || {
  echo "Gemini Scene Studio is not running on $base_url" >&2
  echo "start it first with: bash scripts/start.sh $studio_port" >&2
  exit 1
}

submit() {
  local stage="$1"
  local config="$2"
  curl -fsS -X POST \
    -H 'X-Studio-Request: 1' \
    -H 'Content-Type: application/json' \
    "$base_url/api/scenes/$scene_id/jobs" \
    -d "$(jq -cn --arg stage "$stage" --argjson config "$config" '{stage:$stage,config:$config}')" |
    jq -r '.id'
}

wait_for_job() {
  local run_id="$1"
  while true; do
    local run
    run="$(curl -fsS "$base_url/api/scenes/$scene_id" | jq -c --arg id "$run_id" '.runs[] | select(.id==$id)')"
    local state message
    state="$(jq -r '.state' <<<"$run")"
    message="$(jq -r '.progress_message // .message // ""' <<<"$run")"
    printf '\r%-9s %s' "$state" "$message"
    case "$state" in
      succeeded) printf '\n'; return 0 ;;
      failed|cancelled) printf '\n'; return 1 ;;
    esac
    sleep 2
  done
}

run_stage() {
  local stage="$1"
  local config="$2"
  echo "==> $stage"
  wait_for_job "$(submit "$stage" "$config")"
}

run_stage depth '{"backend":"bridgedepth","width":1280,"min_depth":0.04,"max_depth":5.0,"lr_threshold":1.5}'
run_stage poses '{"backend":"colmap_rgb"}'
run_stage cloud '{"crop_min_depth":0.04,"crop_max_depth":0.3,"roi_x":0,"roi_y":0,"roi_w":1,"roi_h":1,"voxel_size":0.002}'
run_stage tsdf '{"voxel_size":0.003,"sdf_trunc":0.012,"min_depth":0.04,"max_depth":0.3}'

if [ "$skip_train" = 1 ]; then
  echo "skipping 3DGS because GEMINI_DEMO_SKIP_TRAIN=1"
else
  run_stage train '{"quality_mode":false,"adaptive_density":false,"iterations":3000,"initial_gaussians":80000,"max_gaussians":80000,"sh_degree":0,"ssim_weight":0,"width":1280,"depth_weight":0.2,"surface_weight":0.1,"dense_match_weight":0.1,"crop_min_depth":0.04,"crop_max_depth":0.3,"roi_x":0,"roi_y":0,"roi_w":1,"roi_h":1}'
fi

echo "done: $base_url/?scene=$scene_id#/region"
