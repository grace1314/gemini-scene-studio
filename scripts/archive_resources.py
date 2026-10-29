"""Copy local pretrained assets into the new project; never modify source projects."""
import argparse
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

from studio.core import ROOT, sha256, write_json

parser=argparse.ArgumentParser()
parser.add_argument("--source",type=Path,default=Path.home()/"Tars_Project/realsenseD405")
args=parser.parse_args()
source=args.source.resolve()
repo=source/"third_party/BridgeDepth"
dest=ROOT/"3rdparty/BridgeDepth"
if not dest.exists():
    shutil.copytree(repo,dest,ignore=shutil.ignore_patterns(".git","__pycache__","*.pyc"))
weight_source=source/"models/bridgedepth/bridge_rvc_pretrain.pth"
weight_dest=ROOT/"models/bridge_rvc_pretrain.pth"
weight_dest.parent.mkdir(exist_ok=True)
if not weight_dest.exists():
    subprocess.run(["cp","--reflink=auto",str(weight_source),str(weight_dest)],check=True)
digest=sha256(weight_dest)
if digest!=sha256(weight_source):raise RuntimeError("Weight copy checksum mismatch")
revision=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],capture_output=True,text=True)
write_json(ROOT/"models/manifest.json",{"bridgedepth":{"filename":weight_dest.name,"sha256":digest,"size_bytes":weight_dest.stat().st_size,"source_local":str(weight_source),"upstream":"https://github.com/aeolusguan/BridgeDepth","code_revision":revision.stdout.strip() if revision.returncode==0 else "local snapshot; revision unavailable","downloaded_again":False}})
# The official Linux C++ Open3D release depends on the LLVM runtime bundled in its wheel.
import open3d
runtime=Path(open3d.__file__).parent
for name in ("libc++.so.1","libc++abi.so.1"):
    shutil.copy2(runtime/name,ROOT/"3rdparty/open3d/lib"/name)
records=[]
for p in (ROOT/"3rdparty").rglob("*"):
    if p.is_file() and p.suffix in (".h",".hpp",".xz") and "BridgeDepth" not in p.parts:
        records.append({"path":str(p.relative_to(ROOT)),"sha256":sha256(p)})
write_json(ROOT/"3rdparty/checksums.json",records)
print("Archived BridgeDepth code and weight, verified SHA256; Open3D LLVM runtimes copied.")
