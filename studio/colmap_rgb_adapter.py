"""Launch the RGB-only COLMAP pipeline in the CUDA reconstruction environment."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .core import read_json


def run_colmap_rgb(scene: Path, out: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    python = Path(os.environ.get(
        "COLMAP_RGB_PYTHON", str(Path.home() / "miniconda3/envs/bridgedepth/bin/python")
    ))
    if not python.is_file():
        raise RuntimeError(f"COLMAP RGB Python environment is missing: {python}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(root), str(root / "3rdparty" / "LightGlue"), env.get("PYTHONPATH", "")
    ])
    env["TORCH_HOME"] = str(root / "models" / "lightglue")
    subprocess.run(
        [str(python), "-u", "-m", "studio.colmap_rgb_runner", str(out)],
        cwd=root, env=env, check=True,
    )
    return read_json(out / "result.json")
