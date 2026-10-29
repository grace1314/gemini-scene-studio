"""Integrated left-RGB ALIKED/LightGlue + COLMAP SfM/BA pose backend."""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

from .core import read_json, write_json


def progress(out: Path, value: float, message: str) -> None:
    print(f"PROGRESS {value:.3f} {message}", flush=True)
    write_json(out / "progress.json", {
        "value": max(0.0, min(1.0, value)), "message": message, "phase": "poses"
    })


def run(arguments: list[str]) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def create_database(database: Path, manifest: dict) -> None:
    run(["colmap", "database_creator", "--database_path", str(database)])
    first = manifest["frames"][0]
    width, height = first["size"]
    k = np.asarray(first["K"], dtype=np.float64)
    params = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) "
        "VALUES(1,1,?,?,?,1)", (width, height, params.tobytes())
    )
    for image_id, frame in enumerate(manifest["frames"], 1):
        connection.execute("INSERT INTO images(image_id,name,camera_id) VALUES(?,?,1)",
                           (image_id, f"{frame['id']}.png"))
    connection.commit()
    connection.close()


def rotation_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    scene = out.parent.parent
    root = Path(__file__).resolve().parents[1]
    request = read_json(out / "request.json")
    manifest = read_json(scene / "manifest.json")
    depth_root = scene / "runs" / request["dependencies"]["depth"]
    frame_ids = [frame["id"] for frame in manifest["frames"]]
    if len(frame_ids) < 3:
        raise RuntimeError("COLMAP SfM requires at least three frames")

    from scripts.colmap_rgb_only_fusion import (
        metric_poses, parse_images, parse_points, recover_scale,
    )
    from scripts.lightglue_colmap_matches import generate_matches

    work = out / "colmap"
    images_dir = work / "images"
    raw_models = work / "sparse_raw"
    adjusted = work / "adjusted"
    text_model = work / "sparse" / "0"
    for folder in (images_dir, raw_models, adjusted, text_model):
        folder.mkdir(parents=True, exist_ok=True)
    for ident in frame_ids:
        shutil.copy2(scene / "frames" / ident / "left.png", images_dir / f"{ident}.png")

    database = work / "database.db"
    matches = work / "matches_aliked_lightglue.txt"
    progress(out, 0.02, "建立 COLMAP 单相机数据库")
    create_database(database, manifest)
    matching = generate_matches(
        database, images_dir, matches,
        lambda value, message: progress(out, value, message),
    )
    progress(out, 0.55, "COLMAP 几何验证")
    run(["colmap", "matches_importer", "--database_path", str(database),
         "--match_list_path", str(matches), "--match_type", "raw",
         "--SiftMatching.use_gpu", "0", "--SiftMatching.guided_matching", "1",
         "--TwoViewGeometry.min_num_inliers", "12",
         "--TwoViewGeometry.max_error", "4",
         "--TwoViewGeometry.min_inlier_ratio", "0.1",
         "--TwoViewGeometry.compute_relative_pose", "1"])

    progress(out, 0.63, "COLMAP 增量 SfM")
    run(["colmap", "mapper", "--database_path", str(database),
         "--image_path", str(images_dir), "--output_path", str(raw_models),
         "--Mapper.multiple_models", "1", "--Mapper.min_model_size", "3",
         "--Mapper.min_num_matches", "12", "--Mapper.init_min_num_inliers", "50",
         "--Mapper.init_min_tri_angle", "2", "--Mapper.abs_pose_min_num_inliers", "15",
         "--Mapper.abs_pose_min_inlier_ratio", "0.08", "--Mapper.abs_pose_max_error", "12",
         "--Mapper.max_reg_trials", "10", "--Mapper.ba_refine_focal_length", "0",
         "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_global_max_num_iterations", "100"])

    candidates = []
    for model in raw_models.iterdir():
        if not model.is_dir():
            continue
        converted = work / f"candidate_{model.name}"
        converted.mkdir()
        run(["colmap", "model_converter", "--input_path", str(model),
             "--output_path", str(converted), "--output_type", "TXT"])
        cameras = parse_images(converted / "images.txt")
        candidates.append((len(cameras), model, cameras))
    if not candidates:
        raise RuntimeError("COLMAP did not produce a sparse model")
    registered, selected, cameras = max(candidates, key=lambda item: item[0])
    if set(cameras) != set(frame_ids):
        missing = sorted(set(frame_ids) - set(cameras))
        raise RuntimeError(f"COLMAP registered {registered}/{len(frame_ids)} frames; missing {missing}")

    progress(out, 0.82, "固定内参全局 Bundle Adjustment")
    run(["colmap", "bundle_adjuster", "--input_path", str(selected),
         "--output_path", str(adjusted), "--BundleAdjustment.refine_focal_length", "0",
         "--BundleAdjustment.refine_principal_point", "0",
         "--BundleAdjustment.refine_extra_params", "0",
         "--BundleAdjustment.max_num_iterations", "150"])
    run(["colmap", "model_converter", "--input_path", str(adjusted),
         "--output_path", str(text_model), "--output_type", "TXT"])

    cameras = parse_images(text_model / "images.txt")
    points, errors = parse_points(text_model / "points3D.txt")
    progress(out, 0.92, "BridgeDepth 观测恢复米制尺度")
    scale, scale_diagnostics = recover_scale(cameras, points, depth_root)
    poses = metric_poses(cameras, frame_ids, scale)
    write_json(out / "poses.json", {ident: poses[ident].tolist() for ident in frame_ids})
    write_json(out / "poses_colmap_unscaled.json", {
        ident: metric_poses(cameras, frame_ids, 1.0)[ident].tolist() for ident in frame_ids
    })

    steps = [float(np.linalg.norm(poses[frame_ids[index]][:3, 3] -
                                  poses[frame_ids[index - 1]][:3, 3]))
             for index in range(1, len(frame_ids))]
    rotations = [rotation_degrees(poses[frame_ids[index - 1]], poses[frame_ids[index]])
                 for index in range(1, len(frame_ids))]
    progress(out, 1.0, "COLMAP SfM / BA 米制位姿完成")
    result = {
        "backend": "Left RGB ALIKED + LightGlue + COLMAP incremental SfM / global BA",
        "frames": len(frame_ids), "registered_frames": len(cameras),
        "input_modality": "left RGB only for feature matching and SfM",
        "depth_usage": "BridgeDepth is used only after BA for metric scale recovery",
        "sparse_points": len(points),
        "mean_reprojection_px": float(np.mean(errors)) if errors else None,
        "metric_scale": scale,
        "scale_samples": scale_diagnostics["raw_samples"],
        "scale_inliers": scale_diagnostics["inlier_samples"],
        "matched_pairs": matching["matched_pairs"],
        "raw_matches": matching["raw_matches"],
        "peak_allocated_mib": matching["peak_allocated_mib"],
        "median_camera_step_m": float(np.median(steps)),
        "max_camera_step_m": max(steps),
        "median_rotation_step_deg": float(np.median(rotations)),
        "max_rotation_step_deg": max(rotations),
        "resolution": manifest["frames"][0]["size"],
        "colmap_model": "colmap/sparse/0",
        "notes": "No RGB-D pose initialization and no ICP. COLMAP optimizes rotations, translations and sparse points; depth supplies only one robust global metric scale after BA.",
    }
    write_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
