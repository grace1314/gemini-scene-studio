"""Real HTTP workflow on explicitly synthetic data. Requires running server."""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8787)
parser.add_argument("--gpu",action="store_true")
parser.add_argument("--scene",help="Resume a specific synthetic test scene")
parser.add_argument("--real",help="Import a known rectified stereo capture and test BridgeDepth")
args=parser.parse_args()
base=f"http://127.0.0.1:{args.port}"

def api(path,body=None,headers=None):
    request=urllib.request.Request(base+"/api"+path,data=json.dumps(body).encode() if body is not None else None,headers=headers if headers is not None else {"Content-Type":"application/json","X-Studio-Request":"1"})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)

def artifact(scene,path):
    with urllib.request.urlopen(base+f"/api/scenes/{scene}/file?path="+urllib.parse.quote(path),timeout=30) as response:return response.read()

def run(scene,stage,config):
    job=api(f"/scenes/{scene}/jobs",{"stage":stage,"config":config})
    start=time.time();last_report=0
    while time.time()-start<900:
        state=api(f"/scenes/{scene}")
        j=next(j for j in state["runs"] if j["id"]==job["id"])
        if j["state"] not in ("queued","running"):
            if j["state"]!="succeeded":
                try:print(artifact(scene,f"runs/{j['id']}/log.txt").decode(),flush=True)
                except Exception:pass
                raise AssertionError(j)
            result=json.loads(artifact(scene,f"runs/{j['id']}/result.json"))
            print(stage,"PASS",json.dumps(result,ensure_ascii=False)[:700],flush=True)
            return job["id"],result
        if time.time()-last_report>15:
            print(stage,j["state"],round(time.time()-start),"s",flush=True);last_report=time.time()
        time.sleep(1)
    raise TimeoutError(stage)

assert api("/health")["server"]=="C++17"
try:api("/scenes",{"name":"should reject"},headers={"Content-Type":"application/json"})
except urllib.error.HTTPError as error:assert error.code==403
else:raise AssertionError("CSRF header not enforced")
if args.real:
    scene=api("/scenes",{"name":"真实双目 · BridgeDepth验证"})["id"]
    print("SCENE",scene,flush=True)
    run(scene,"import",{"source":args.real,"rectified_confirmed":True})
    _,result=run(scene,"depth",{"backend":"bridgedepth","width":320,"max_depth":3.0})
    assert result["frames"],result
    print("REAL DEPTH SMOKE PASSED",scene,flush=True)
    raise SystemExit(0)
elif args.scene:
    scene=args.scene
    assert api(f"/scenes/{scene}")["synthetic"]
else:
    scene=api("/scenes",{"name":"合成验证 · C++ 全流程"})["id"]
    print("SCENE",scene,flush=True)
    run(scene,"demo",{})
    try:api(f"/scenes/{scene}/jobs",{"stage":"cloud","config":{}})
    except urllib.error.HTTPError as error:assert error.code==400
    else:raise AssertionError("Missing dependencies accepted")
    run(scene,"depth",{"backend":"sgbm","width":320,"max_depth":1.0})
    pose_run,_=run(scene,"poses",{})
    poses=json.loads(artifact(scene,f"runs/{pose_run}/poses.json"))
    truth=json.loads(artifact(scene,"ground_truth_poses.json"))
    errors=[]
    for key,T in poses.items():
        error=sum((T[i][3]-truth[key][i][3])**2 for i in range(3))**0.5
        errors.append(error)
    assert max(errors)<0.015,errors
    print("Metric translation error max:",max(errors),"m",flush=True)
    run(scene,"cloud",{})
    run(scene,"tsdf",{})
    run(scene,"archive",{})
    try:artifact(scene,"../../../.venv/pyvenv.cfg")
    except urllib.error.HTTPError as error:assert error.code==400
    else:raise AssertionError("Path traversal accepted")
if args.gpu:
    run(scene,"train",{"iterations":10,"max_gaussians":1000,"width":320})
print("SMOKE PASSED",scene,flush=True)
