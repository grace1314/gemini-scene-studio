from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .core import ROOT, read_json, write_json


_progress_path = None


def set_progress_path(path):
    global _progress_path
    _progress_path = path


def progress(value, message):
    print(f"PROGRESS {value:.3f} {message}", flush=True)
    if _progress_path is not None:
        write_json(_progress_path, {"value": max(0.0, min(1.0, float(value))), "message": message, "phase": "depth"})


def upstream(scene, request, stage):
    return scene / "runs" / request["dependencies"][stage]


def depth_frame(folder, ident):
    meta = read_json(folder / "result.json")["frames"][ident]
    return np.load(folder / ident / "depth.npy"), np.array(meta["K"]), cv2.imread(str(folder / ident / "rgb.png")), meta


def run_depth(scene, out, request, manifest):
    set_progress_path(out / "progress.json")
    cfg = request["config"]
    model = None
    if cfg["backend"] == "bridgedepth":
        import sys
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用；BridgeDepth需要GPU。可选择SGBM完成CPU基线。")
        repo = ROOT / "3rdparty" / "BridgeDepth"
        weight = ROOT / "models" / "bridge_rvc_pretrain.pth"
        if not repo.exists() or not weight.exists():
            raise RuntimeError("BridgeDepth代码或权重缺失，请运行资源归档脚本")
        sys.path.insert(0, str(repo))
        from bridgedepth.bridgedepth import BridgeDepth
        model = BridgeDepth.from_pretrained(str(weight)).cuda().eval()
        torch.cuda.reset_peak_memory_stats()

    def infer(left, right):
        if model is not None:
            import torch
            def tensor(x):
                return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().cuda()
            with torch.inference_mode():
                return model({"img1": tensor(left), "img2": tensor(right)})["disp_pred"].squeeze().cpu().numpy()
        disparities = min(128, max(16, (left.shape[1] // 3 // 16) * 16))
        matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=disparities, blockSize=5, P1=8*25, P2=32*25, uniquenessRatio=8, speckleWindowSize=80, speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return matcher.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16

    results = {}
    for index, frame in enumerate(manifest["frames"]):
        dest = out / frame["id"]
        dest.mkdir()
        source = scene / "frames" / frame["id"]
        left, right = [cv2.imread(str(source / (side + ".png"))) for side in ("left", "right")]
        h0, w0 = left.shape[:2]
        # Formal RGB-D processing uses the captured resolution. ``width`` is an
        # upper bound for imported datasets, never a preview-resolution knob.
        width = min(cfg["width"], w0)
        height = round(h0 * width / w0)
        left, right = [cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA) for x in (left, right)]
        K = np.array(frame["K"], dtype=float)
        K[0] *= width / w0
        K[1] *= height / h0
        disparity = infer(left, right).astype(np.float32)
        # Flip and swap to estimate right-view positive disparity.
        d_right = np.fliplr(infer(np.ascontiguousarray(np.fliplr(right)), np.ascontiguousarray(np.fliplr(left))))
        yy, xx = np.indices(disparity.shape)
        finite = np.isfinite(disparity)
        xr = np.rint(xx - np.where(finite, disparity, 0)).astype(int)
        inside = (xr >= 0) & (xr < width)
        sampled = d_right[yy, np.clip(xr, 0, width-1)]
        lr_error = np.abs(disparity - sampled)
        valid = finite & (disparity > 0) & inside & np.isfinite(sampled) & (sampled > 0) & (lr_error <= cfg["lr_threshold"])
        depth = np.zeros_like(disparity)
        depth[valid] = K[0, 0] * frame["baseline_m"] / disparity[valid]
        valid &= (depth >= cfg["min_depth"]) & (depth <= cfg["max_depth"])
        depth[~valid] = 0
        quality = np.where(valid, np.exp(-np.nan_to_num(lr_error, nan=100)), 0).astype(np.float32)
        np.save(dest / "depth.npy", depth)
        np.save(dest / "disparity.npy", disparity)
        np.save(dest / "quality.npy", quality)
        cv2.imwrite(str(dest / "depth.tiff"), depth)
        cv2.imwrite(str(dest / "quality.tiff"), quality)
        cv2.imwrite(str(dest / "valid.png"), valid.astype(np.uint8)*255)
        cv2.imwrite(str(dest / "rgb.png"), left)
        colored = cv2.applyColorMap(np.clip((depth-cfg["min_depth"])/(cfg["max_depth"]-cfg["min_depth"])*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(str(dest / "preview.png"), colored)
        results[frame["id"]] = {"K": K.tolist(), "width": width, "height": height,
                                "source_width": w0, "source_height": h0,
                                "inference_width": width, "inference_height": height,
                                "full_resolution": width == w0 and height == h0,
                                "valid_fraction": float(valid.mean()),
                                "median_depth_m": float(np.median(depth[valid])) if valid.any() else None}
        if not valid.any():
            print(f"WARNING {frame['id']} 没有有效深度", flush=True)
        progress((index+1)/len(manifest["frames"]), f"深度 {index+1}/{len(manifest['frames'])}")
    result = {"backend": cfg["backend"], "frames": results,
              "resolution_policy": "formal RGB-D uses captured resolution; only live preview is compressed",
              "quality_definition": "exp(-left-right disparity error); not calibrated probability"}
    if model is not None:
        import torch
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
    return result
