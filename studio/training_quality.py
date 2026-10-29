"""Quality-oriented small-scene 3DGS for an 8 GB GPU.

Uses gsplat's adaptive density control, SH appearance, SSIM, metric RGB-D
geometry constraints, and exports both training and interpolated novel views.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .core import read_json, write_json

C0 = 0.28209479177387814


def _ssim(x, y, mask=None):
    """Small differentiable SSIM; inputs are HWC RGB in [0, 1]."""
    x = x.permute(2, 0, 1)[None]
    y = y.permute(2, 0, 1)[None]
    mu_x = F.avg_pool2d(x, 7, 1, 3)
    mu_y = F.avg_pool2d(y, 7, 1, 3)
    sigma_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2) *
        (sigma_x + sigma_y + 0.03**2)
    )
    score = score.clamp(-1, 1)
    if mask is not None and mask.any():
        selected = score[0].permute(1, 2, 0)[mask].reshape(-1)
        return selected.mean()
    return score.mean()


def _polygon_mask(manifest, ident, width, height):
    points = manifest.get("processing", {}).get("frame_rois", {}).get(ident, [])
    if len(points) < 3:
        return np.ones((height, width), dtype=bool)
    polygon = np.rint(np.asarray(points, dtype=np.float32) * [width - 1, height - 1]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def _load_views(deps, cfg, manifest):
    poses = read_json(deps["poses"] / "poses.json")
    meta = read_json(deps["depth"] / "result.json")["frames"]
    views = []
    for ident, pose in poses.items():
        source = deps["depth"] / ident
        bgr = cv2.imread(str(source / "rgb.png"))
        depth = cv2.imread(str(source / "depth.tiff"), cv2.IMREAD_UNCHANGED)
        quality = cv2.imread(str(source / "quality.tiff"), cv2.IMREAD_UNCHANGED)
        h0, w0 = depth.shape
        width = min(cfg["width"], w0)
        height = round(h0 * width / w0)
        K = np.array(meta[ident]["K"], dtype=np.float32)
        K[0] *= width / w0
        K[1] *= height / h0
        rgb = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy() / 255.0
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        quality = cv2.resize(quality, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = _polygon_mask(manifest, ident, width, height)
        mask &= np.isfinite(depth) & (depth >= cfg["crop_min_depth"]) & (depth <= cfg["crop_max_depth"])
        depth = np.where(mask, depth, 0).astype(np.float32)
        quality = np.where(mask, quality, 0).astype(np.float32)
        views.append((ident, rgb, depth, quality, K, np.linalg.inv(np.array(pose, dtype=np.float32)), mask))
    return views


def _dense_pairs(views, out):
    pairs = []
    for index in range(len(views) - 1):
        a, b = views[index], views[index + 1]
        height, width = a[2].shape
        gray_a = cv2.cvtColor((a[1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor((b[1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        forward = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, .5, 3, 21, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, .5, 3, 21, 3, 5, 1.2, 0)
        yy, xx = np.mgrid[8:height-8:12, 8:width-8:12]
        flow = forward[yy, xx]
        qx = np.rint(xx + flow[..., 0]).astype(np.int32)
        qy = np.rint(yy + flow[..., 1]).astype(np.int32)
        inside = (qx >= 2) & (qx < width - 2) & (qy >= 2) & (qy < height - 2)
        xx, yy, qx, qy, flow = xx[inside], yy[inside], qx[inside], qy[inside], flow[inside]
        if not len(xx):
            continue
        fb_error = np.linalg.norm(flow + backward[qy, qx], axis=1)
        za, zb = a[2][yy, xx], b[2][qy, qx]
        qa, qb = a[3][yy, xx], b[3][qy, qx]
        valid = np.isfinite(fb_error) & (fb_error < 1.25) & (za > 0) & (zb > 0) & (qa > .25) & (qb > .25)
        xx, yy, qx, qy, za, zb = xx[valid], yy[valid], qx[valid], qy[valid], za[valid], zb[valid]
        if not len(xx):
            continue

        def world(view, x, y, z):
            K = view[4]
            camera = np.column_stack(((x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)))
            return (np.linalg.inv(view[5]) @ camera.T).T[:, :3]

        consistent = np.linalg.norm(world(a, xx, yy, za) - world(b, qx, qy, zb), axis=1) < .02
        if consistent.sum() >= 8:
            pairs.append({"a": index, "b": index + 1,
                          "pixels_a": np.column_stack((xx[consistent], yy[consistent])).astype(np.int16),
                          "pixels_b": np.column_stack((qx[consistent], qy[consistent])).astype(np.int16)})
    summary = [{"frame_a": views[p["a"]][0], "frame_b": views[p["b"]][0], "matches": len(p["pixels_a"])} for p in pairs]
    write_json(out / "dense_matches.json", {"method": "Farneback forward/backward + RGB-D world consistency",
                                             "pairs": summary, "max_world_error_m": .02, "stride_px": 12})
    return pairs, summary


def _camera_path(views, samples_per_pair=3):
    path = []
    for index in range(len(views) - 1):
        a, b = views[index], views[index + 1]
        c2w_a, c2w_b = np.linalg.inv(a[5]), np.linalg.inv(b[5])
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]])))
        for sample in range(1, samples_per_pair + 1):
            t = sample / (samples_per_pair + 1)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = slerp([t]).as_matrix()[0]
            c2w[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
            K = ((1 - t) * a[4] + t * b[4]).astype(np.float32)
            path.append({"W": np.linalg.inv(c2w), "K": K, "source_a": a[0], "source_b": b[0], "t": t})
    return path


def train_quality(scene: Path, out: Path, request, manifest):
    from gsplat import rasterization
    from gsplat.exporter import export_splats
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import remove

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Quality Gaussian training requires CUDA.")
    cfg = request["config"]
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    write_json(out / "progress.json", {"value": 0, "message": "加载质量模式训练数据", "phase": "train"})
    deps = {key: scene / "runs" / value for key, value in request["dependencies"].items()}
    views = _load_views(deps, cfg, manifest)
    pairs, pair_summary = _dense_pairs(views, out)

    cloud = o3d.io.read_point_cloud(str(deps["cloud"] / "fused_cloud.ply"))
    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors)
    normals = np.asarray(cloud.normals)
    if len(xyz) < 100:
        raise ValueError("Need at least 100 fused points")
    max_gaussians = int(cfg["max_gaussians"])
    initial_count = min(int(cfg.get("initial_gaussians", min(30000, max_gaussians))), max_gaussians, len(xyz))
    take = np.random.default_rng(42).choice(len(xyz), initial_count, replace=False)
    xyz, rgb = xyz[take], rgb[take]
    normals = normals[take] if len(normals) else np.tile([0., 0., 1.], (len(xyz), 1))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    reference = np.tile([1., 0., 0.], (len(xyz), 1))
    reference[np.abs(normals[:, 0]) > .9] = [0., 1., 0.]
    tangent = np.cross(normals, reference)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    second = np.cross(normals, tangent)
    rot = Rotation.from_matrix(np.stack([tangent, second, normals], axis=2)).as_quat()
    quats = np.column_stack([rot[:, 3], rot[:, :3]])
    spacing = np.clip(cKDTree(xyz).query(xyz, k=2)[0][:, 1], .0005, .025)
    scales = np.column_stack([spacing, spacing, spacing * .25])
    sh_degree = int(cfg.get("sh_degree", 2))
    coefficients = (sh_degree + 1) ** 2

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous()

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(tensor(xyz)),
        "scales": torch.nn.Parameter(tensor(np.log(scales))),
        "quats": torch.nn.Parameter(tensor(quats)),
        "opacities": torch.nn.Parameter(torch.full((len(xyz),), float(torch.logit(torch.tensor(.1))), device="cuda")),
        "sh0": torch.nn.Parameter(tensor((rgb - .5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((len(xyz), coefficients - 1, 3), device="cuda")),
        "anchors": torch.nn.Parameter(tensor(xyz), requires_grad=False),
        "anchor_normals": torch.nn.Parameter(tensor(normals), requires_grad=False),
    })
    adaptive_density = bool(cfg.get("adaptive_density", False))
    rates = ({"means": 8e-5, "scales": 3e-3, "quats": 1e-3, "opacities": 2e-2, "sh0": 2.5e-3, "shN": 1.25e-4}
             if adaptive_density else
             {"means": 1e-5, "scales": 5e-4, "quats": 3e-4, "opacities": 1e-2, "sh0": 2e-3, "shN": 5e-5})
    optimizers = {name: torch.optim.Adam([params[name]], lr=rate, eps=1e-15) for name, rate in rates.items()}
    iterations = int(cfg["iterations"])
    strategy = DefaultStrategy(prune_opa=.01, grow_grad2d=.00035, grow_scale3d=.012,
                               refine_start_iter=min(400, max(50, iterations // 10)),
                               refine_stop_iter=max(1, int(iterations * .82)),
                               reset_every=max(1200, iterations // 2), refine_every=100,
                               pause_refine_after_reset=len(views), verbose=True)
    strategy.check_sanity(params, optimizers)
    robust_low, robust_high = np.percentile(xyz, [2, 98], axis=0)
    scene_scale = max(float(np.linalg.norm(robust_high - robust_low)), .1)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    def render_camera(W, K, width, height, active_degree):
        sh = torch.cat([params["sh0"], params["shN"]], dim=1)
        rendered, alpha, info = rasterization(
            means=params["means"], quats=F.normalize(params["quats"], dim=-1),
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=sh, sh_degree=active_degree, viewmats=tensor(W)[None], Ks=tensor(K)[None],
            width=width, height=height, render_mode="RGB+ED", packed=True,
            near_plane=.005, rasterize_mode="antialiased")
        return rendered[0], alpha[0, ..., 0], info

    history = []
    for step in range(iterations):
        view_index = step % len(views)
        view = views[view_index]
        height, width = view[2].shape
        active_degree = min(sh_degree, step // 1000)
        rendered, alpha, info = render_camera(view[5], view[4], width, height, active_degree)
        target, depth, quality = tensor(view[1]), tensor(view[2]), tensor(view[3])
        train_mask = torch.as_tensor(view[6], dtype=torch.bool, device="cuda")
        l1 = (rendered[..., :3][train_mask] - target[train_mask]).abs().mean() if train_mask.any() else rendered.sum() * 0
        ssim_loss = 1 - _ssim(rendered[..., :3].clamp(0, 1), target, train_mask)
        rgb_loss = (1 - cfg.get("ssim_weight", .2)) * l1 + cfg.get("ssim_weight", .2) * ssim_loss
        mask = (depth > 0) & (quality > .25) & (alpha.detach() > .1)
        depth_loss = ((rendered[..., 3][mask].clamp_min(1e-5).log() - depth[mask].log()).abs() * quality[mask]).mean() if mask.any() else rendered.sum() * 0
        offset = params["means"] - params["anchors"]
        surface_loss = (offset * params["anchor_normals"]).sum(-1).abs().mean() / .01 + offset.square().sum(-1).mean() / .01**2 * .03
        dense_loss = rendered.sum() * 0
        match = next((pair for pair in pairs if pair["a"] == view_index or pair["b"] == view_index), None)
        if cfg.get("dense_match_weight", 0) > 0 and match is not None:
            ia, ib = match["a"], match["b"]
            other_index = ib if view_index == ia else ia
            other = views[other_index]
            uv0 = match["pixels_a"] if view_index == ia else match["pixels_b"]
            uv1 = match["pixels_b"] if view_index == ia else match["pixels_a"]
            sample = np.random.default_rng(step + 812).choice(len(uv0), min(256, len(uv0)), replace=False)
            uv0, uv1 = uv0[sample], uv1[sample]
            rd0 = rendered[..., 3][uv0[:, 1], uv0[:, 0]]
            rendered_other, alpha_other, _ = render_camera(other[5], other[4], width, height, active_degree)
            rd1 = rendered_other[..., 3][uv1[:, 1], uv1[:, 0]]
            visible = (rd0 > 0) & (rd1 > 0) & (alpha.detach()[uv0[:, 1], uv0[:, 0]] > .1) & (alpha_other.detach()[uv1[:, 1], uv1[:, 0]] > .1)
            if visible.any():
                def world(view_data, uv, z):
                    K = view_data[4]
                    pixel = tensor(uv.astype(np.float32))
                    camera = torch.stack(((pixel[:, 0] - float(K[0, 2])) * z / float(K[0, 0]),
                                          (pixel[:, 1] - float(K[1, 2])) * z / float(K[1, 1]), z, torch.ones_like(z)), dim=-1)
                    return (tensor(np.linalg.inv(view_data[5])) @ camera.T).T[:, :3]
                dense_loss = F.smooth_l1_loss(world(view, uv0, rd0)[visible], world(other, uv1, rd1)[visible], beta=.01, reduction="none").sum(-1).mean() / .01
        loss = rgb_loss + cfg["depth_weight"] * depth_loss + cfg["surface_weight"] * surface_loss + cfg.get("dense_match_weight", 0) * dense_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if adaptive_density:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if adaptive_density:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
        if adaptive_density and len(params["means"]) > max_gaussians:
            excess = len(params["means"]) - max_gaussians
            discard = torch.topk(params["opacities"].detach(), excess, largest=False).indices
            mask_remove = torch.zeros(len(params["means"]), dtype=torch.bool, device="cuda")
            mask_remove[discard] = True
            remove(params, optimizers, strategy_state, mask_remove)
        with torch.no_grad():
            params["scales"].clamp_(np.log(.0001), np.log(.06))
            params["opacities"].clamp_(-8, 8)
        if step % 25 == 0 or step + 1 == iterations:
            entry = {"step": step + 1, "loss": float(loss.detach()), "rgb": float(rgb_loss.detach()),
                     "l1": float(l1.detach()), "ssim": float(ssim_loss.detach()), "depth": float(depth_loss.detach()),
                     "surface": float(surface_loss.detach()), "dense_match": float(dense_loss.detach()),
                     "gaussians": len(params["means"]), "sh_degree": active_degree}
            history.append(entry)
            message = f"质量训练 {step + 1}/{iterations} gaussians={entry['gaussians']} loss={entry['loss']:.6f}"
            print(f"PROGRESS {(step + 1) / iterations:.3f} {message}", flush=True)
            write_json(out / "progress.json", {"value": (step + 1) / iterations, "message": message, "phase": "train"})
            write_json(out / "loss.json", history)

    cpu = {key: value.detach().cpu() for key, value in params.items() if key not in ("anchors", "anchor_normals")}
    torch.save({"parameters": cpu, "config": cfg, "dependencies": request["dependencies"], "format": f"SH{sh_degree} adaptive RGB-D geometry"}, out / "checkpoint.pt")
    export_splats(cpu["means"], cpu["scales"], F.normalize(cpu["quats"], dim=-1), cpu["opacities"], cpu["sh0"], cpu["shN"], format="ply", save_to=str(out / "gaussians.ply"))
    export_splats(cpu["means"], cpu["scales"], F.normalize(cpu["quats"], dim=-1), cpu["opacities"], cpu["sh0"], cpu["shN"], format="splat", save_to=str(out / "gaussians.splat"))

    psnrs = []
    with torch.no_grad():
        for view in views:
            height, width = view[2].shape
            image, _, _ = render_camera(view[5], view[4], width, height, sh_degree)
            rgb_image = image[..., :3].clamp(0, 1)
            mse = (rgb_image - tensor(view[1])).square().mean().clamp_min(1e-10)
            psnrs.append(float(-10 * torch.log10(mse)))
            cv2.imwrite(str(out / f"render_{view[0]}.png"), (rgb_image.cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
        novel = _camera_path(views)
        novel_meta = []
        video = None
        for index, camera in enumerate(novel):
            height, width = views[0][2].shape
            image, _, _ = render_camera(camera["W"], camera["K"], width, height, sh_degree)
            bgr = (image[..., :3].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
            name = f"novel_{index:04d}.png"
            cv2.imwrite(str(out / name), bgr)
            if video is None:
                video = cv2.VideoWriter(str(out / "novel_view.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
            video.write(bgr)
            novel_meta.append({"file": name, "source_a": camera["source_a"], "source_b": camera["source_b"], "t": camera["t"]})
        if video is not None:
            video.release()
        write_json(out / "novel_views.json", novel_meta)

    means = cpu["means"].numpy()
    colors = np.clip(cpu["sh0"].squeeze(1).numpy() * C0 + .5, 0, 1)
    indices = np.linspace(0, len(means) - 1, min(25000, len(means))).astype(int)
    write_json(out / "preview.json", {"positions": means[indices].ravel().tolist(), "colors": colors[indices].ravel().tolist(),
                                      "total_points": len(means), "sampled_points": len(indices), "label": "Gaussian centres diagnostic"})
    density_label = "adaptive" if adaptive_density else "fixed-density conservative"
    return {"backend": f"gsplat 1.5.3 SH{sh_degree} {density_label} RGB-D geometry",
            "gaussians": len(means), "initial_gaussians": initial_count, "iterations": iterations,
            "seconds": time.time() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "final_loss": history[-1], "mean_train_psnr": float(np.mean(psnrs)),
            "dense_match_pairs": pair_summary, "novel_views": novel_meta,
            "limitations": ["interpolated novel views stay near the captured camera path", "fixed input camera poses", "browser centre preview is diagnostic; PNG/MP4 outputs use true Gaussian rasterization"]}
