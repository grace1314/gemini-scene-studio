"""Low-VRAM OmniVGGT RGB-D pose backend.

The model receives every BridgeDepth RGB-D frame, but only the first camera is
used as a coordinate-system anchor.  This prevents the existing (possibly bad)
trajectory from being baked back into the prediction.  The predicted trajectory
is rescaled to metres from the supplied stereo depth before the normal native
point-cloud/TSDF stages consume it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scipy.spatial import cKDTree


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def _progress(out: Path, value: float, message: str) -> None:
    _write(out / "progress.json", {
        "value": max(0.0, min(1.0, value)), "message": message, "phase": "poses"
    })


def _homogeneous(matrix_3x4: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3] = matrix_3x4
    return result


def _sample_world_cloud(depth: np.ndarray, quality: np.ndarray, k: np.ndarray,
                        pose: np.ndarray, max_depth: float, stride: int = 6) -> np.ndarray:
    yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    q = quality[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0.04) & (z < max_depth) & (q >= 0.25)
    z, xx, yy = z[valid], xx[valid], yy[valid]
    local = np.column_stack(((xx - k[0, 2]) * z / k[0, 0],
                             (yy - k[1, 2]) * z / k[1, 1], z))
    return local @ pose[:3, :3].T + pose[:3, 3]


def _adjacent_geometry_check(scene: Path, depth_root: Path, manifest: dict,
                             poses: list[np.ndarray], max_depth: float) -> list[dict]:
    clouds = []
    for frame, pose in zip(manifest["frames"], poses):
        ident = frame["id"]
        depth = np.load(depth_root / ident / "depth.npy").astype(np.float32)
        quality = np.load(depth_root / ident / "quality.npy").astype(np.float32)
        clouds.append(_sample_world_cloud(depth, quality, np.asarray(frame["K"]),
                                          pose, max_depth))
    checks = []
    threshold = 0.010
    for index, (source, target) in enumerate(zip(clouds[:-1], clouds[1:])):
        if len(source) < 100 or len(target) < 100:
            checks.append({"pair": [index, index + 1], "overlap": 0.0,
                           "rmse_m": None, "passed": False})
            continue
        forward = cKDTree(target).query(source, workers=-1)[0]
        reverse = cKDTree(source).query(target, workers=-1)[0]
        inliers = np.concatenate((forward[forward < threshold], reverse[reverse < threshold]))
        overlap = float(min(np.mean(forward < threshold), np.mean(reverse < threshold)))
        rmse = float(np.sqrt(np.mean(inliers * inliers))) if inliers.size else None
        checks.append({"pair": [index, index + 1], "overlap": overlap,
                       "rmse_m": rmse, "passed": overlap >= 0.12 and rmse is not None and rmse <= 0.007})
    return checks


def main() -> None:
    scene = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    root = Path(__file__).resolve().parents[1]
    omni = root / "3rdparty" / "OmniVGGT"
    checkpoint = omni / "checkpoints" / "OmniVGGT.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 4_800_000_000:
        raise RuntimeError("OmniVGGT 权重未完整下载（需要约 4.87 GB）")

    sys.path.insert(0, str(omni))
    os.environ["OMNIVGGT_SKIP_DINOV2_INIT"] = "1"
    from omnivggt.models.omnivggt import OmniVGGT
    from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

    request = json.loads((out / "request.json").read_text())
    manifest = json.loads((scene / "manifest.json").read_text())
    depth_id = request["dependencies"]["depth"]
    depth_root = scene / "runs" / depth_id
    frame_ids = [frame["id"] for frame in manifest["frames"]]
    target_width = int(request["config"].get("target_width", 224))
    max_depth = float(request["config"].get("max_depth", 1.5))
    min_quality = float(request["config"].get("min_quality", 0.25))

    _progress(out, 0.03, "准备全帧 RGB-D 与相机锚点")
    images, depths, masks, intrinsics = [], [], [], []
    original_depths = []
    final_height = None
    for frame in manifest["frames"]:
        ident = frame["id"]
        rgb = np.asarray(Image.open(depth_root / ident / "rgb.png").convert("RGB"))
        depth = np.load(depth_root / ident / "depth.npy").astype(np.float32)
        quality = np.load(depth_root / ident / "quality.npy").astype(np.float32)
        height, width = rgb.shape[:2]
        resized_height = max(14, round((height * target_width / width) / 14) * 14)
        final_height = resized_height
        rgb_small = cv2.resize(rgb, (target_width, resized_height), interpolation=cv2.INTER_AREA)
        depth_small = cv2.resize(depth, (target_width, resized_height), interpolation=cv2.INTER_NEAREST)
        quality_small = cv2.resize(quality, (target_width, resized_height), interpolation=cv2.INTER_NEAREST)
        valid = (depth_small > 0.04) & (depth_small < max_depth) & (quality_small >= min_quality)
        depth_small = np.where(valid, depth_small, 0).astype(np.float32)
        k = np.asarray(frame["K"], dtype=np.float32).copy()
        k[0, :] *= target_width / width
        k[1, :] *= resized_height / height
        images.append(torch.from_numpy(rgb_small).permute(2, 0, 1).float() / 255.0)
        depths.append(depth_small[..., None])
        masks.append(valid.astype(np.float32))
        intrinsics.append(k)
        original_depths.append(depth_small)

    image_tensor = torch.stack(images)[None]
    depth_tensor = torch.from_numpy(np.asarray(depths))[None]
    mask_tensor = torch.from_numpy(np.asarray(masks))[None]
    intrinsic_tensor = torch.from_numpy(np.asarray(intrinsics))[None]
    extrinsic_tensor = torch.zeros((1, len(frame_ids), 3, 4), dtype=torch.float32)
    extrinsic_tensor[0, 0, :3, :3] = torch.eye(3)

    _progress(out, 0.10, "加载 OmniVGGT 权重")
    model = OmniVGGT()
    state = load_file(str(checkpoint))
    model.load_state_dict(state, strict=True)
    del state
    # The downstream fusion deliberately uses the original full-resolution
    # BridgeDepth maps, so the dense point-map head would only consume VRAM.
    model.point_head = None
    model.depth_head = None
    device = torch.device("cuda")
    model = model.half().to(device).eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = {
        "images": image_tensor.half().to(device),
        "extrinsics": extrinsic_tensor.half().to(device),
        "intrinsics": intrinsic_tensor.half().to(device),
        "depth": depth_tensor.half().to(device),
        "mask": mask_tensor.half().to(device),
        "depth_gt_index": list(range(len(frame_ids))),
        "camera_gt_index": [0],
    }
    _progress(out, 0.25, f"OmniVGGT 全局推理：{len(frame_ids)} 帧，{target_width}×{final_height}")
    started = time.time()
    with torch.inference_mode():
        prediction = model.inference(**inputs)
        world_to_camera, predicted_k = pose_encoding_to_extri_intri(
            prediction["pose_enc"].float(), image_tensor.shape[-2:]
        )
    inference_seconds = time.time() - started
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    w2c = world_to_camera.float().cpu().numpy()[0]
    predicted_k_np = predicted_k.float().cpu().numpy()[0]
    del prediction, inputs, model
    torch.cuda.empty_cache()

    # OmniVGGT's GeoAdapter divides all supplied valid depths by their batch
    # mean (see ZeroAggregator.normalize_depth).  Camera translations therefore
    # use the same normalized unit; multiplying by that exact mean restores metres.
    valid_depth_values = np.concatenate([d[d > 0.04] for d in original_depths])
    metric_scale = float(np.mean(valid_depth_values)) if valid_depth_values.size else 1.0
    per_frame_ratios = [float(np.mean(d[d > 0.04])) if np.any(d > 0.04) else 0.0
                        for d in original_depths]

    camera_to_world = []
    for matrix in w2c:
        camera_to_world.append(np.linalg.inv(_homogeneous(matrix)))
    anchor_inverse = np.linalg.inv(camera_to_world[0])
    camera_to_world = [anchor_inverse @ matrix for matrix in camera_to_world]
    for matrix in camera_to_world:
        matrix[:3, 3] *= metric_scale

    poses = {ident: matrix.tolist() for ident, matrix in zip(frame_ids, camera_to_world)}
    _write(out / "poses.json", poses)
    np.savez_compressed(
        out / "omnivggt_diagnostics.npz",
        predicted_intrinsics=predicted_k_np.astype(np.float32),
        supplied_depth_mean_m=np.asarray(per_frame_ratios, dtype=np.float32),
    )

    centers = np.asarray([matrix[:3, 3] for matrix in camera_to_world])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    rotations = []
    for first, second in zip(camera_to_world[:-1], camera_to_world[1:]):
        relative = first[:3, :3].T @ second[:3, :3]
        rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))))

    _progress(out, 0.92, "验证相邻 RGB-D 几何重叠")
    geometry_checks = _adjacent_geometry_check(scene, depth_root, manifest,
                                               camera_to_world, max_depth)
    valid_rmse = [item["rmse_m"] for item in geometry_checks if item["rmse_m"] is not None]
    failed_pairs = [item["pair"] for item in geometry_checks if not item["passed"]]

    result = {
        "backend": "OmniVGGT RGB-D global pose (first-camera anchor, metric stereo rescale)",
        "frames": len(frame_ids),
        "registered_frames": len(frame_ids),
        "resolution": [target_width, final_height],
        "depth_frames_supplied": len(frame_ids),
        "camera_frames_supplied": 1,
        "metric_scale": metric_scale,
        "per_frame_depth_scale_median": per_frame_ratios,
        "inference_seconds": inference_seconds,
        "peak_allocated_mib": peak_mib,
        "median_camera_step_m": float(np.median(steps)) if steps.size else 0.0,
        "max_camera_step_m": float(np.max(steps)) if steps.size else 0.0,
        "median_rotation_step_deg": float(np.median(rotations)) if rotations else 0.0,
        "max_rotation_step_deg": float(np.max(rotations)) if rotations else 0.0,
        "median_adjacent_overlap": float(np.median([item["overlap"] for item in geometry_checks])),
        "median_adjacent_rmse_m": float(np.median(valid_rmse)) if valid_rmse else None,
        "failed_geometry_pairs": failed_pairs,
        "adjacent_geometry": geometry_checks,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "limitations": [
            "OmniVGGT runs at reduced inference resolution for 8GB VRAM; final cloud uses full-resolution BridgeDepth",
            "only the first camera anchors coordinates so the previous bad trajectory is not reused",
        ],
    }
    _write(out / "result.json", result)
    _progress(out, 1.0, "OmniVGGT 米制全局位姿完成")


if __name__ == "__main__":
    main()
