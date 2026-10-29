import importlib.metadata
import platform
import subprocess
import time
import torch
from studio.core import ROOT,write_json

report={"checked_at":time.time(),"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"packages":{}}
for name in ("gsplat","open3d","numpy","opencv-python","ninja"):
    try:report["packages"][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:report["packages"][name]="missing"
if report["cuda_available"]:
    report["gpu"]=torch.cuda.get_device_name(0)
    free,total=torch.cuda.mem_get_info()
    report.update(free_vram_mib=free/2**20,total_vram_mib=total/2**20)
    report["cuda_smoke_sum"]=float(torch.ones(16,device="cuda").sum())
usb=subprocess.run(["lsusb"],text=True,capture_output=True)
report["usb_inventory"]=usb.stdout.strip().splitlines()
write_json(ROOT/"data/environment.json",report)
print(report)
