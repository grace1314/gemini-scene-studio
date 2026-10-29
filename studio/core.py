from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("GEMINI_STUDIO_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = ROOT / "data" / "scenes"
STAGES = ("depth", "poses", "cloud", "tsdf", "train")
DEPENDENCIES = {"depth": [], "poses": ["depth"], "cloud": ["depth", "poses"], "tsdf": ["depth", "poses"], "train": ["depth", "poses", "cloud"]}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_path(scene_id):
    if not scene_id or any(c not in "0123456789abcdef" for c in scene_id) or len(scene_id) != 12:
        raise ValueError("无效场景ID")
    p = DATA / scene_id
    if not (p / "manifest.json").is_file():
        raise ValueError("场景不存在")
    return p


def runs(scene):
    return sorted([read_json(p) for p in (scene / "runs").glob("*/status.json")], key=lambda r: r["created_at"])


def active_runs(scene):
    revision = read_json(scene / "manifest.json")["revision"]
    current = {}
    for run in runs(scene):
        if run["state"] != "succeeded" or run["revision"] != revision:
            continue
        if all(current.get(k, {}).get("id") == v for k, v in run["dependencies"].items()):
            current[run["stage"]] = run
            for stage in list(current):
                if any(current.get(k, {}).get("id") != v for k, v in current[stage]["dependencies"].items()):
                    del current[stage]
    return current


def artifact_path(scene, relative):
    p = (scene / relative).resolve()
    if not p.is_relative_to(scene.resolve()) or not p.is_file() or p.is_symlink():
        raise ValueError("文件不存在或路径越界")
    return p


def verify_inputs(scene):
    manifest = read_json(scene / "manifest.json")
    for frame in manifest["frames"]:
        for name, digest in frame["hashes"].items():
            if sha256(scene / "frames" / frame["id"] / name) != digest:
                raise ValueError(f"输入已被外部修改: {frame['id']}/{name}，请重新导入")
    return manifest
