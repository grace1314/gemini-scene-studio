#!/usr/bin/env python3
"""Run RGB-only COLMAP SfM/BA, recover metric scale from depth, then fuse RGB-D."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def parse_images(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    images = {}
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            continue
        image_id = int(fields[0])
        rotation = quaternion_rotation(*map(float, fields[1:5]))
        translation = np.asarray(fields[5:8], dtype=np.float64)
        observations = []
        obs = lines[index + 1].split() if index + 1 < len(lines) else []
        for offset in range(0, len(obs) - 2, 3):
            point_id = int(obs[offset + 2])
            if point_id >= 0:
                observations.append((float(obs[offset]), float(obs[offset + 1]), point_id))
        images[Path(fields[9]).stem] = {
            "image_id": image_id, "R": rotation, "t": translation,
            "observations": observations,
        }
    return images


def parse_points(path: Path):
    points = {}
    errors = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 8:
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
            errors.append(float(fields[7]))
    return points, errors


def recover_scale(images, points, depth_root: Path) -> tuple[float, dict]:
    ratios = []
    per_frame = {}
    for ident, image in images.items():
        depth = np.load(depth_root / ident / "depth.npy").astype(np.float64)
        quality = np.load(depth_root / ident / "quality.npy").astype(np.float64)
        values = []
        for x, y, point_id in image["observations"]:
            px, py = int(round(x)), int(round(y))
            if point_id not in points or not (0 <= px < depth.shape[1] and 0 <= py < depth.shape[0]):
                continue
            measured = depth[py, px]
            z_colmap = (image["R"] @ points[point_id] + image["t"])[2]
            if quality[py, px] >= 0.25 and 0.04 <= measured <= 1.5 and z_colmap > 1e-8:
                values.append(float(measured / z_colmap))
        if values:
            per_frame[ident] = {"samples": len(values), "median_scale": float(np.median(values))}
            ratios.extend(values)
    if len(ratios) < 100:
        raise RuntimeError(f"Only {len(ratios)} valid RGB-D scale observations")
    ratios = np.asarray(ratios)
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    robust_sigma = max(1.4826 * mad, median * 0.01)
    kept = ratios[np.abs(ratios - median) <= 3.0 * robust_sigma]
    scale = float(np.median(kept))
    return scale, {
        "raw_samples": int(len(ratios)), "inlier_samples": int(len(kept)),
        "median": median, "mad": mad, "scale": scale, "per_frame": per_frame,
    }


def metric_poses(images, frame_ids: list[str], scale: float):
    camera_to_world = {}
    for ident in frame_ids:
        image = images[ident]
        pose = np.eye(4)
        pose[:3, :3] = image["R"].T
        pose[:3, 3] = -image["R"].T @ image["t"]
        camera_to_world[ident] = pose
    first_inverse = np.linalg.inv(camera_to_world[frame_ids[0]])
    result = {}
    for ident in frame_ids:
        pose = first_inverse @ camera_to_world[ident]
        pose[:3, 3] *= scale
        result[ident] = pose
    return result


def roi_mask(shape, polygon) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if len(polygon or []) >= 3:
        mask[:] = False
        points = np.asarray([[round(x * (shape[1] - 1)), round(y * (shape[0] - 1))]
                             for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(mask.view(np.uint8), [points], 1)
    return mask


def fuse(manifest, depth_root: Path, poses, output: Path) -> dict:
    processing = manifest.get("processing", {})
    minimum = float(processing.get("crop_min_depth", 0.04))
    maximum = float(processing.get("crop_max_depth", 0.3))
    rois = processing.get("frame_rois", {})
    combined = o3d.geometry.PointCloud()
    frame_counts = {}
    for frame in manifest["frames"]:
        ident = frame["id"]
        depth = np.load(depth_root / ident / "depth.npy").astype(np.float32)
        quality = np.load(depth_root / ident / "quality.npy").astype(np.float32)
        rgb = cv2.cvtColor(cv2.imread(str(depth_root / ident / "rgb.png")), cv2.COLOR_BGR2RGB)
        keep = roi_mask(depth.shape, rois.get(ident, []))
        stride = 2
        yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[::stride, ::stride]
        valid = (keep[::stride, ::stride] & np.isfinite(z) & (z >= minimum) &
                 (z <= maximum) & (quality[::stride, ::stride] >= 0.25))
        z, x, y = z[valid], xx[valid], yy[valid]
        k = np.asarray(frame["K"], dtype=np.float64)
        xyz = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z))
        colors = rgb[::stride, ::stride][valid].astype(np.float64) / 255.0
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud = cloud.voxel_down_sample(0.0015)
        frame_counts[ident] = len(cloud.points)
        combined += cloud.transform(poses[ident])
    before = len(combined.points)
    combined = combined.voxel_down_sample(0.002)
    if len(combined.points) > 100:
        combined, _ = combined.remove_statistical_outlier(20, 2.0)
    combined.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    o3d.io.write_point_cloud(str(output), combined, write_ascii=False, compressed=False)
    extent = combined.get_axis_aligned_bounding_box().get_extent()
    return {"points_before_final_voxel": before, "points": len(combined.points),
            "voxel_size_m": 0.002, "extent_m": extent.tolist(),
            "depth_range_m": [minimum, maximum], "per_frame_points": frame_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("depth_run", type=str)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scene, output = args.scene.resolve(), args.output.resolve()
    manifest = load_json(scene / "manifest.json")
    frames = manifest["frames"]
    frame_ids = [frame["id"] for frame in frames]
    depth_root = scene / "runs" / args.depth_run
    if len(frame_ids) != 20:
        raise RuntimeError(f"Expected 20 frames, got {len(frame_ids)}")
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    images_dir = output / "images"
    sparse_dir = output / "sparse"
    text_dir = output / "model_text"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir()
    text_dir.mkdir()
    for ident in frame_ids:
        shutil.copy2(scene / "frames" / ident / "left.png", images_dir / f"{ident}.png")

    k = np.asarray(frames[0]["K"], dtype=np.float64)
    camera_params = f"{k[0,0]},{k[1,1]},{k[0,2]},{k[1,2]}"
    database = output / "database.db"
    run(["colmap", "feature_extractor", "--database_path", str(database),
         "--image_path", str(images_dir), "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", camera_params,
         "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_num_features", "16384",
         "--SiftExtraction.peak_threshold", "0.002", "--SiftExtraction.edge_threshold", "20",
         "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"])
    run(["colmap", "exhaustive_matcher", "--database_path", str(database),
         "--SiftMatching.use_gpu", "0", "--SiftMatching.guided_matching", "1",
         "--SiftMatching.max_ratio", "0.85", "--SiftMatching.max_num_matches", "65536"])
    run(["colmap", "mapper", "--database_path", str(database), "--image_path", str(images_dir),
         "--output_path", str(sparse_dir), "--Mapper.multiple_models", "1",
         "--Mapper.min_model_size", "3", "--Mapper.min_num_matches", "20",
         "--Mapper.init_min_num_inliers", "80", "--Mapper.init_min_tri_angle", "4",
         "--Mapper.abs_pose_min_num_inliers", "20", "--Mapper.abs_pose_min_inlier_ratio", "0.15",
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_global_max_num_iterations", "100"])

    models = [folder for folder in sparse_dir.iterdir() if folder.is_dir()]
    if not models:
        raise RuntimeError("COLMAP mapper produced no model")
    best = None
    for folder in models:
        candidate = output / f"model_text_{folder.name}"
        candidate.mkdir()
        run(["colmap", "model_converter", "--input_path", str(folder),
             "--output_path", str(candidate), "--output_type", "TXT"])
        parsed = parse_images(candidate / "images.txt")
        if best is None or len(parsed) > len(best[2]):
            best = folder, candidate, parsed
    model, candidate, images = best
    if set(images) != set(frame_ids):
        missing = sorted(set(frame_ids) - set(images))
        raise RuntimeError(f"Largest model registered {len(images)}/20 frames; missing {missing}")
    shutil.rmtree(text_dir)
    shutil.copytree(candidate, text_dir)
    points, errors = parse_points(text_dir / "points3D.txt")
    scale, scale_diagnostics = recover_scale(images, points, depth_root)
    poses = metric_poses(images, frame_ids, scale)
    save_json(output / "poses_colmap_metric.json", {key: value.tolist() for key, value in poses.items()})
    cloud_stats = fuse(manifest, depth_root, poses, output / "fused_cloud_colmap_rgb_only.ply")
    steps = [float(np.linalg.norm(poses[frame_ids[i]][:3, 3] - poses[frame_ids[i-1]][:3, 3]))
             for i in range(1, len(frame_ids))]
    report = {
        "method": "RGB-only COLMAP exhaustive SIFT + incremental SfM + global BA; metric scale recovered after SfM from BridgeDepth observations",
        "input_images": len(frame_ids), "registered_images": len(images),
        "sparse_points": len(points),
        "mean_reprojection_error_px": float(np.mean(errors)) if errors else None,
        "selected_model": model.name, "metric_scale": scale,
        "scale_diagnostics": scale_diagnostics,
        "median_camera_step_m": float(np.median(steps)), "max_camera_step_m": max(steps),
        "cloud": cloud_stats,
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
