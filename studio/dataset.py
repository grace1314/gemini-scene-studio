from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .core import DATA, ROOT, read_json, sha256, write_json


def make_demo(scene):
    """Analytic textured sphere + plane, cameras translating along x; not real capture."""
    manifest = read_json(scene / "manifest.json")
    if manifest["frames"]:
        raise ValueError("请在空场景生成合成数据")
    w, h, focal, baseline = 320, 240, 280.0, 0.018
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([(xx - w / 2) / focal, (yy - h / 2) / focal, np.ones_like(xx)], -1)
    rng = np.random.default_rng(42)
    texture = rng.integers(30, 225, (768, 1024, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)

    def render(cx):
        origin = np.array([cx, 0, 0])
        center = np.array([0.025, 0, 0.43])
        oc = origin - center
        a = (rays * rays).sum(-1)
        b = 2 * (rays * oc).sum(-1)
        c = (oc * oc).sum() - 0.065 ** 2
        disc = b * b - 4 * a * c
        z = np.full((h, w), 0.60)
        hit = disc > 0
        z[hit] = (-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit])
        points = rays * z[..., None] + origin
        tx = np.clip(((points[..., 0] + 0.4) * 1200).astype(int), 0, 1023)
        ty = np.clip(((points[..., 1] + 0.3) * 1200).astype(int), 0, 767)
        rgb = texture[ty, tx].copy()
        rgb[hit, 1] = np.minimum(rgb[hit, 1].astype(int) + 40, 255)
        return rgb, z.astype(np.float32)

    frames, truth = [], {}
    for i, cx in enumerate([-0.02, 0, 0.02, 0.04]):
        ident = f"{i:04d}"
        dest = scene / "frames" / ident
        dest.mkdir(parents=True, exist_ok=True)
        for side, offset in (("left", 0), ("right", baseline)):
            rgb, z = render(cx + offset)
            cv2.imwrite(str(dest / f"{side}.png"), rgb)
            if side == "left":
                np.save(dest / "ground_truth_depth.npy", z)
        (dest / "K.txt").write_text(" ".join(map(str, K.ravel())) + f"\n{baseline}\n")
        T = np.eye(4)
        T[0, 3] = cx + 0.02
        truth[ident] = T.tolist()
        frames.append({"id": ident, "K": K.tolist(), "baseline_m": baseline, "size": [w, h], "source": "analytic synthetic fixture", "hashes": {n: sha256(dest / n) for n in ("left.png", "right.png", "K.txt")}})
    manifest.update(frames=frames, synthetic=True, revision=manifest["revision"] + 1, rectified_confirmed=True)
    write_json(scene / "manifest.json", manifest)
    write_json(scene / "ground_truth_poses.json", truth)
    return manifest
