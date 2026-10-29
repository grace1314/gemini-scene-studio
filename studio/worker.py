"""One-shot model subprocess. Production HTTP and scheduling live in C++."""
import os
import sys
import traceback
import zipfile
from pathlib import Path

from .core import read_json, write_json, verify_inputs


def configure_cuda_build_environment():
    """Make the local CUDA compiler and extension builder visible to gsplat.

    The server launches this interpreter by absolute path, so activating neither the
    project venv nor the BridgeDepth conda environment is implied.  gsplat's first
    import may JIT-build its CUDA extension and therefore needs both nvcc and ninja
    on PATH even though PyTorch can already see the GPU runtime.
    """
    # sys.executable is a symlink in this venv; resolving it would point at the
    # BridgeDepth interpreter and hide the venv-local ninja executable.
    venv_bin = str(Path(sys.prefix) / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([venv_bin, *path_parts])

    configured = os.environ.get("GEMINI_CUDA_HOME") or os.environ.get("CUDA_HOME")
    candidates = [configured, str(Path.home() / "miniconda3/envs/bridgedepth"), "/usr/local/cuda"]
    cuda_home = next((Path(item) for item in candidates if item and (Path(item) / "bin/nvcc").is_file()), None)
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        cuda_bin = str(cuda_home / "bin")
        if cuda_bin not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = os.pathsep.join([cuda_bin, os.environ["PATH"]])

        # Conda's split CUDA packages keep development headers/libraries under
        # targets/x86_64-linux instead of CUDA_HOME/include and CUDA_HOME/lib64.
        target = cuda_home / "targets" / "x86_64-linux"
        for variable, directory in (
            ("CPATH", target / "include"),
            ("LIBRARY_PATH", target / "lib"),
            ("LD_LIBRARY_PATH", target / "lib"),
        ):
            if directory.is_dir():
                existing = os.environ.get(variable, "")
                entries = existing.split(os.pathsep) if existing else []
                if str(directory) not in entries:
                    os.environ[variable] = os.pathsep.join([str(directory), *entries])

    os.environ.setdefault("MAX_JOBS", "2")
    # Gemini Scene Studio is deployed on the RTX 4060 Laptop GPU (Ada, sm_89).
    # Pinning the architecture avoids rebuilding an extra PTX variant at startup.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def main():
    out = Path(sys.argv[1]).resolve()
    scene = out.parent.parent
    request = read_json(out / "request.json")
    stage = request["stage"]
    if stage == "demo":
        from .dataset import make_demo
        result = {"frames": len(make_demo(scene)["frames"]), "synthetic": True}
    elif stage == "archive":
        paths = [p for p in scene.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix not in (".zip", ".tmp") and out not in p.parents]
        with zipfile.ZipFile(out / "scene_archive.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in paths:
                archive.write(p, p.relative_to(scene))
        result = {"files": len(paths), "archive": "scene_archive.zip", "scope": "scene data and run history; excludes weights and environment"}
    elif stage == "depth":
        from .pipeline import run_depth
        result = run_depth(scene, out, request, verify_inputs(scene))
    elif stage == "poses" and request["config"].get("backend") == "colmap_ba":
        from .colmap_refine import refine_colmap
        result = refine_colmap(scene, out, request, verify_inputs(scene))
    elif stage == "poses" and request["config"].get("backend") == "omnivggt_rgbd":
        from .omnivggt_adapter import run_omnivggt
        result = run_omnivggt(scene, out)
    elif stage == "poses" and request["config"].get("backend") == "colmap_rgb":
        from .colmap_rgb_adapter import run_colmap_rgb
        result = run_colmap_rgb(scene, out)
    elif stage == "train":
        if request["config"].get("quality_mode", False):
            from .training_quality import train_quality
            result = train_quality(scene, out, request, verify_inputs(scene))
        else:
            from .training import train
            result = train(scene, out, request, verify_inputs(scene))
    else:
        raise ValueError("Stage belongs to the native C++ worker")
    write_json(out / "result.json", result)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    configure_cuda_build_environment()
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR {exc}", flush=True)
        sys.exit(1)
