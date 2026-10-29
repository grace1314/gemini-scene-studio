"""Launch OmniVGGT in the existing CUDA/PyTorch BridgeDepth environment."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .core import read_json


def run_omnivggt(scene: Path, out: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    python = Path(os.environ.get(
        "OMNIVGGT_PYTHON", str(Path.home() / "miniconda3/envs/bridgedepth/bin/python")
    ))
    if not python.is_file():
        raise RuntimeError(f"OmniVGGT Python 环境不存在: {python}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "3rdparty" / "OmniVGGT")
    env["OMNIVGGT_SKIP_DINOV2_INIT"] = "1"
    subprocess.run(
        [str(python), "-u", "-m", "studio.omnivggt_runner", str(scene), str(out)],
        cwd=root, env=env, check=True,
    )
    from .omnivggt_refine import refine_omnivggt
    return refine_omnivggt(scene, out)
