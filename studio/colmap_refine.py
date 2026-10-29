"""COLMAP visual-track triangulation and bundle adjustment from metric RGB-D poses."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .core import read_json, write_json


def _progress(out: Path, value: float, message: str) -> None:
    print(f"PROGRESS {value:.3f} {message}", flush=True)
    write_json(out / "progress.json", {
        "value": max(0.0, min(1.0, value)), "message": message, "phase": "poses"
    })


def _run(arguments: list[str]) -> None:
    print("COLMAP " + " ".join(arguments[1:3]), flush=True)
    subprocess.run(arguments, check=True)


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _parse_images(path: Path) -> dict[str, np.ndarray]:
    cameras = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 10 or fields[0].startswith("#") or not fields[9].lower().endswith(".png"):
            continue
        values = [float(x) for x in fields[1:8]]
        rotation = _quaternion_rotation(*values[:4])
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = values[4:]
        cameras[Path(fields[9]).stem] = np.linalg.inv(world_to_camera)
    return cameras


def _latest_metric_pose_run(scene: Path, current: str, depth_run: str) -> Path:
    candidates = []
    for status_path in (scene / "runs").glob("*/status.json"):
        status = read_json(status_path)
        if (status.get("id") == current or status.get("stage") != "poses" or status.get("state") != "succeeded"):
            continue
        if status.get("dependencies", {}).get("depth") != depth_run:
            continue
        run = status_path.parent
        if not (run / "poses.json").is_file() or not (run / "colmap/sparse/0/images.txt").is_file():
            continue
        request = read_json(run / "request.json")
        if request.get("config", {}).get("backend", "opencv_rgbd") == "opencv_rgbd":
            candidates.append((float(status.get("created_at", 0)), run))
    if not candidates:
        raise RuntimeError("请先运行全分辨率 RGB-D 鲁棒配准，再执行 COLMAP BA 精修")
    return max(candidates)[1]


def refine_colmap(scene: Path, out: Path, request: dict, manifest: dict) -> dict:
    depth_run_id = request["dependencies"]["depth"]
    depth = scene / "runs" / depth_run_id
    baseline = _latest_metric_pose_run(scene, out.name, depth_run_id)
    baseline_poses = {k: np.asarray(v, dtype=np.float64)
                      for k, v in read_json(baseline / "poses.json").items()}
    frame_ids = [frame["id"] for frame in manifest["frames"]]
    if set(frame_ids) != set(baseline_poses):
        raise RuntimeError("米制初始轨迹未覆盖全部采集帧")

    work = out / "colmap"
    images = work / "images"
    initial = work / "initial_metric"
    triangulated = work / "triangulated"
    adjusted = work / "adjusted"
    text_model = work / "sparse" / "0"
    for folder in (images, initial, triangulated, adjusted, text_model):
        folder.mkdir(parents=True, exist_ok=True)
    for ident in frame_ids:
        shutil.copy2(depth / ident / "rgb.png", images / f"{ident}.png")
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(baseline / "colmap/sparse/0" / name, initial / name)

    first_meta = read_json(depth / "result.json")["frames"][frame_ids[0]]
    K = first_meta["K"]
    camera = f"{K[0][0]},{K[1][1]},{K[0][2]},{K[1][2]}"
    database = work / "database.db"
    _progress(out, 0.04, "COLMAP 提取全分辨率 SIFT 特征")
    _run(["colmap", "feature_extractor", "--database_path", str(database),
          "--image_path", str(images), "--ImageReader.single_camera", "1",
          "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", camera,
          "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_num_features", "8192",
          "--SiftExtraction.peak_threshold", "0.002", "--SiftExtraction.edge_threshold", "20",
          "--SiftExtraction.estimate_affine_shape", "1",
          "--SiftExtraction.domain_size_pooling", "1"])
    _progress(out, 0.27, "COLMAP 全图像对引导匹配")
    _run(["colmap", "exhaustive_matcher", "--database_path", str(database),
          "--SiftMatching.use_gpu", "0", "--SiftMatching.guided_matching", "1",
          "--SiftMatching.max_ratio", "0.9", "--SiftMatching.max_num_matches", "65536"])
    _progress(out, 0.54, "按米制初始轨迹三角化视觉轨迹")
    _run(["colmap", "point_triangulator", "--database_path", str(database),
          "--image_path", str(images), "--input_path", str(initial),
          "--output_path", str(triangulated), "--Mapper.ba_refine_focal_length", "0",
          "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0"])
    _progress(out, 0.76, "COLMAP 全局 Bundle Adjustment")
    _run(["colmap", "bundle_adjuster", "--input_path", str(triangulated),
          "--output_path", str(adjusted), "--BundleAdjustment.refine_focal_length", "0",
          "--BundleAdjustment.refine_principal_point", "0",
          "--BundleAdjustment.refine_extra_params", "0",
          "--BundleAdjustment.max_num_iterations", "100"])
    _run(["colmap", "model_converter", "--input_path", str(adjusted),
          "--output_path", str(text_model), "--output_type", "TXT"])

    cameras = _parse_images(text_model / "images.txt")
    if set(cameras) != set(frame_ids):
        raise RuntimeError(f"COLMAP BA 仅注册 {len(cameras)}/{len(frame_ids)} 帧，拒绝生成不完整轨迹")
    reference = cameras[frame_ids[0]]
    relative = {ident: np.linalg.inv(reference) @ cameras[ident] for ident in frame_ids}
    ratios = []
    for i, a in enumerate(frame_ids):
        for b in frame_ids[i + 1:]:
            metric_distance = np.linalg.norm(baseline_poses[a][:3, 3] - baseline_poses[b][:3, 3])
            colmap_distance = np.linalg.norm(relative[a][:3, 3] - relative[b][:3, 3])
            if metric_distance > 0.005 and colmap_distance > 1e-8:
                ratios.append(metric_distance / colmap_distance)
    if len(ratios) < len(frame_ids):
        raise RuntimeError("COLMAP BA 无法恢复可靠的米制尺度")
    scale = float(np.median(ratios))
    output = {}
    shifts = []
    for ident in frame_ids:
        pose = relative[ident].copy()
        pose[:3, 3] *= scale
        if not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-3:
            raise RuntimeError(f"COLMAP BA 输出非法位姿: {ident}")
        shifts.append(float(np.linalg.norm(pose[:3, 3] - baseline_poses[ident][:3, 3])))
        output[ident] = pose.tolist()
    if max(shifts) > 0.30:
        raise RuntimeError(f"COLMAP BA 相对米制初值偏移过大 ({max(shifts):.3f} m)，已拒绝覆盖")
    write_json(out / "poses.json", output)

    point_errors = []
    point_count = 0
    for line in (text_model / "points3D.txt").read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#") and len(fields) >= 8:
            point_count += 1
            point_errors.append(float(fields[7]))
    _progress(out, 1.0, "COLMAP 三角化与全局 BA 完成")
    return {
        "backend": "COLMAP full-resolution guided SIFT + metric initialization + global BA",
        "frames": len(frame_ids), "registered_frames": len(cameras),
        "sparse_points": point_count,
        "mean_reprojection_px": float(np.mean(point_errors)) if point_errors else None,
        "metric_scale_correction": scale,
        "median_pose_shift_m": float(np.median(shifts)), "max_pose_shift_m": max(shifts),
        "initial_pose_run": baseline.name,
        "resolution": [first_meta["width"], first_meta["height"]],
        "edges": [],
        "notes": "All 20 cameras initialized in metric RGB-D coordinates; COLMAP triangulates visual tracks and jointly optimizes camera extrinsics and sparse points."
    }
