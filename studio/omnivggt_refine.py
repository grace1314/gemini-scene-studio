"""Conservative full-resolution geometry refinement for OmniVGGT poses."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .core import read_json, write_json


def _cloud(depth_path: Path, quality_path: Path, k: np.ndarray,
           min_depth: float, max_depth: float) -> o3d.geometry.PointCloud:
    depth = np.load(depth_path).astype(np.float32)
    quality = np.load(quality_path).astype(np.float32)
    stride = 2
    yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    q = quality[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth) & (q >= 0.25)
    z, xx, yy = z[valid], xx[valid], yy[valid]
    points = np.column_stack(((xx - k[0, 2]) * z / k[0, 0],
                              (yy - k[1, 2]) * z / k[1, 1], z))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(0.0035)
    if len(cloud.points) > 100:
        cloud, _ = cloud.remove_statistical_outlier(16, 2.0)

    # A tabletop can make translation parallel to the plane and yaw weakly
    # observable.  Keep all non-planar object geometry and only a sparse sample
    # of the dominant plane for normal/height support.
    if len(cloud.points) > 500:
        try:
            _, plane = cloud.segment_plane(0.003, 3, 120)
            plane_set = set(plane)
            keep = [idx for idx in range(len(cloud.points))
                    if idx not in plane_set or idx % 8 == 0]
            cloud = cloud.select_by_index(keep)
        except RuntimeError:
            pass
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.014, max_nn=32))
    return cloud


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def refine_omnivggt(scene: Path, out: Path) -> dict:
    manifest = read_json(scene / "manifest.json")
    request = read_json(out / "request.json")
    result = read_json(out / "result.json")
    depth_root = scene / "runs" / request["dependencies"]["depth"]
    processing = manifest.get("processing", {})
    min_depth = float(processing.get("crop_min_depth", 0.04))
    max_depth = min(float(request["config"].get("max_depth", 1.5)),
                    max(0.35, float(processing.get("crop_max_depth", 0.3))))
    frame_ids = [frame["id"] for frame in manifest["frames"]]
    original = read_json(out / "poses.json")
    write_json(out / "poses_omnivggt_raw.json", original)
    poses = [np.asarray(original[ident], dtype=np.float64) for ident in frame_ids]

    clouds = []
    for frame in manifest["frames"]:
        ident = frame["id"]
        clouds.append(_cloud(depth_root / ident / "depth.npy",
                             depth_root / ident / "quality.npy",
                             np.asarray(frame["K"], dtype=np.float64),
                             min_depth, max_depth))

    refined = [poses[0]]
    world_clouds = [copy.deepcopy(clouds[0]).transform(refined[0])]
    diagnostics = []
    accepted = 0
    for index in range(1, len(clouds)):
        target = o3d.geometry.PointCloud()
        for item in world_clouds[max(0, len(world_clouds) - 3):]:
            target += item
        target = target.voxel_down_sample(0.0035)
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.014, max_nn=32))
        initial = poses[index]
        coarse = o3d.pipelines.registration.registration_generalized_icp(
            clouds[index], target, 0.015, initial,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
                o3d.pipelines.registration.HuberLoss(0.005)),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 1e-6, 35))
        fine = o3d.pipelines.registration.registration_icp(
            clouds[index], target, 0.0075, coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(0.0035)),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, 50))
        delta = fine.transformation @ np.linalg.inv(initial)
        translation = float(np.linalg.norm(delta[:3, 3]))
        rotation = _rotation_degrees(delta)
        use = (np.isfinite(fine.transformation).all() and fine.fitness >= 0.16 and
               fine.inlier_rmse <= 0.0075 and translation <= 0.035 and rotation <= 10.0)
        pose = fine.transformation if use else initial
        accepted += int(use)
        refined.append(pose)
        world_clouds.append(copy.deepcopy(clouds[index]).transform(pose))
        diagnostics.append({"pair": [index - 1, index], "accepted": use,
                            "fitness": float(fine.fitness),
                            "rmse_m": float(fine.inlier_rmse),
                            "translation_delta_m": translation,
                            "rotation_delta_deg": rotation})

    write_json(out / "poses.json", {ident: pose.tolist()
                                     for ident, pose in zip(frame_ids, refined)})
    result["backend"] += " + full-resolution geometry GICP/point-to-plane refinement"
    result["geometry_refinement"] = {
        "accepted_frames": accepted,
        "rejected_frames": len(diagnostics) - accepted,
        "registration_max_depth_m": max_depth,
        "dominant_plane_keep_fraction": 0.125,
        "frames": diagnostics,
    }
    write_json(out / "result.json", result)
    return result
