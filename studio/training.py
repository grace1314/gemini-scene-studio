"""Small-scene SH0 3DGS: RGB + depth + surface anchors + dense 3D correspondences.

No densification, no neural SDF or pose optimization; dense matches add a cross-view
world-space consistency loss after forward/backward flow and RGB-D geometry filtering.
"""
import time
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .core import read_json, write_json


def _progress(out, value, message):
    print(f"PROGRESS {value:.3f} {message}", flush=True)
    write_json(out/"progress.json", {"value": max(0.0, min(1.0, float(value))), "message": message, "phase": "train"})


def _polygon_mask(manifest, ident, width, height):
    points = manifest.get("processing", {}).get("frame_rois", {}).get(ident, [])
    if len(points) < 3:
        return np.ones((height, width), dtype=bool)
    polygon = np.rint(np.asarray(points, dtype=np.float32) * [width - 1, height - 1]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def train(scene, out, request, manifest):
    from gsplat import rasterization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. The web service can run on CPU; Gaussian training cannot.")
    cfg = request["config"]
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    _progress(out, 0, "加载 RGB-D、位姿与融合点云")
    deps = {k: scene / "runs" / v for k, v in request["dependencies"].items()}
    cloud = o3d.io.read_point_cloud(str(deps["cloud"] / "fused_cloud.ply"))
    xyz, rgb, normals = np.asarray(cloud.points), np.asarray(cloud.colors), np.asarray(cloud.normals)
    if len(xyz)<100:
        raise ValueError("Need at least 100 fused points")
    take = np.random.default_rng(42).choice(len(xyz), min(cfg["max_gaussians"],len(xyz)), replace=False)
    xyz, rgb = xyz[take], rgb[take]
    normals = normals[take] if len(normals) else np.tile([0.,0.,1.],(len(xyz),1))
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    # Construct tangent frames; shortest Gaussian axis is local z.
    reference=np.tile([1.,0.,0.],(len(xyz),1))
    reference[np.abs(normals[:,0])>0.9]=[0.,1.,0.]
    tangent=np.cross(normals,reference)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
    second=np.cross(normals,tangent)
    rot=Rotation.from_matrix(np.stack([tangent,second,normals],axis=2)).as_quat()
    quat=np.column_stack([rot[:,3],rot[:,:3]])
    spacing=np.clip(cKDTree(xyz).query(xyz,k=2)[0][:,1],0.0005,0.02)
    scales=np.column_stack([spacing,spacing,spacing*0.25])
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device="cuda").contiguous()
    anchor=tensor(xyz); anchor_normal=tensor(normals)
    params=torch.nn.ParameterDict({
        "means":torch.nn.Parameter(anchor.clone()),
        "quats":torch.nn.Parameter(tensor(quat)),
        "log_scales":torch.nn.Parameter(tensor(np.log(scales))),
        "color_logits":torch.nn.Parameter(tensor(np.log(np.clip(rgb,0.001,0.999)/(1-np.clip(rgb,0.001,0.999))))),
        "opacity_logits":torch.nn.Parameter(torch.zeros(len(xyz),device="cuda")),
    })
    optimizer=torch.optim.Adam([
        {"params":[params["means"]],"lr":0.00005},
        {"params":[params["quats"],params["log_scales"]],"lr":0.001},
        {"params":[params["color_logits"],params["opacity_logits"]],"lr":0.01},
    ])
    poses=read_json(deps["poses"]/"poses.json")
    meta=read_json(deps["depth"]/"result.json")["frames"]
    # CPU storage; upload only one image and camera at a time.
    views=[]
    for ident,pose in poses.items():
        source=deps["depth"]/ident
        bgr=cv2.imread(str(source/"rgb.png"))
        d=cv2.imread(str(source/"depth.tiff"),cv2.IMREAD_UNCHANGED)
        quality=cv2.imread(str(source/"quality.tiff"),cv2.IMREAD_UNCHANGED)
        h0,w0=d.shape;w=min(cfg["width"],w0);h=round(h0*w/w0)
        K=np.array(meta[ident]["K"]);K[0]*=w/w0;K[1]*=h/h0
        bgr=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_AREA)
        d=cv2.resize(d,(w,h),interpolation=cv2.INTER_NEAREST)
        quality=cv2.resize(quality,(w,h),interpolation=cv2.INTER_NEAREST)
        x0=max(0,min(w-1,int(np.floor(cfg["roi_x"]*w))));y0=max(0,min(h-1,int(np.floor(cfg["roi_y"]*h))))
        x1=max(x0+1,min(w,int(np.ceil((cfg["roi_x"]+cfg["roi_w"])*w))));y1=max(y0+1,min(h,int(np.ceil((cfg["roi_y"]+cfg["roi_h"])*h))))
        train_mask=np.zeros((h,w),dtype=bool);train_mask[y0:y1,x0:x1]=True
        train_mask&=_polygon_mask(manifest,ident,w,h)
        train_mask&=np.isfinite(d)&(d>=cfg["crop_min_depth"])&(d<=cfg["crop_max_depth"])
        d=np.where(train_mask,d,0).astype(np.float32)
        quality=np.where(train_mask,quality,0).astype(np.float32)
        views.append((ident,bgr[:,:,::-1].copy()/255,d,quality,K,np.linalg.inv(np.array(pose)),train_mask))

    # Dense optical correspondences are computed on CPU once and retained as compact
    # sampled pixel pairs. Depth + current camera poses reject matches that disagree
    # geometrically before they can supervise Gaussian cross-view consistency.
    geometry_pairs=[]
    for index in range(len(views)-1):
        a,b=views[index],views[index+1]
        h,w=a[2].shape
        gray_a=cv2.cvtColor((a[1]*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
        gray_b=cv2.cvtColor((b[1]*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
        forward=cv2.calcOpticalFlowFarneback(gray_a,gray_b,None,.5,3,21,3,5,1.2,0)
        backward=cv2.calcOpticalFlowFarneback(gray_b,gray_a,None,.5,3,21,3,5,1.2,0)
        yy,xx=np.mgrid[8:h-8:16,8:w-8:16]
        fx=forward[yy,xx]
        qx=np.rint(xx+fx[...,0]).astype(np.int32);qy=np.rint(yy+fx[...,1]).astype(np.int32)
        inside=(qx>=2)&(qx<w-2)&(qy>=2)&(qy<h-2)
        xx,yy,qx,qy,fx=xx[inside],yy[inside],qx[inside],qy[inside],fx[inside]
        if not len(xx):
            continue
        back=backward[qy,qx]
        fb_error=np.linalg.norm(fx+back,axis=1)
        za=a[2][yy,xx];zb=b[2][qy,qx]
        qa=a[3][yy,xx];qb=b[3][qy,qx]
        valid=(np.isfinite(fb_error)&(fb_error<1.5)&(za>cfg["min_depth"] if "min_depth" in cfg else za>0)&(zb>0)&(qa>.2)&(qb>.2))
        xx,yy,qx,qy,za,zb=xx[valid],yy[valid],qx[valid],qy[valid],za[valid],zb[valid]
        if not len(xx):
            continue
        def world_points(view,x,y,z):
            K=view[4]; camera=np.column_stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z,np.ones_like(z)))
            return (np.linalg.inv(view[5])@camera.T).T[:,:3]
        wa=world_points(a,xx,yy,za);wb=world_points(b,qx,qy,zb)
        consistent=np.linalg.norm(wa-wb,axis=1)<.025
        if consistent.sum()>=8:
            geometry_pairs.append({"a":index,"b":index+1,"pixels_a":np.column_stack((xx[consistent],yy[consistent])).astype(np.int16),"pixels_b":np.column_stack((qx[consistent],qy[consistent])).astype(np.int16)})
    pair_summary=[{"frame_a":views[p["a"]][0],"frame_b":views[p["b"]][0],"matches":len(p["pixels_a"])} for p in geometry_pairs]
    write_json(out/"dense_matches.json",{"method":"Farneback forward/backward + RGB-D world consistency","pairs":pair_summary,"max_world_error_m":0.025,"stride_px":16})

    def render(view):
        ident,rgb,d,q,K,W,*_=view;h,w=d.shape
        rendered,alpha,_=rasterization(means=params["means"],quats=params["quats"]/params["quats"].norm(dim=-1,keepdim=True).clamp_min(1e-8),scales=params["log_scales"].exp(),opacities=params["opacity_logits"].sigmoid(),colors=params["color_logits"].sigmoid(),viewmats=tensor(W)[None],Ks=tensor(K)[None],width=w,height=h,render_mode="RGB+ED",packed=True,near_plane=0.005)
        return rendered[0],alpha[0,...,0]

    history=[]
    for step in range(cfg["iterations"]):
        view_index=step%len(views)
        view=views[view_index]
        rendered,alpha=render(view)
        target=tensor(view[1]);depth=tensor(view[2]);quality=tensor(view[3])
        train_mask=tensor(view[6])>0
        rgb_loss=(rendered[...,:3][train_mask]-target[train_mask]).abs().mean() if train_mask.any() else rendered.sum()*0
        mask=(depth>0)&(quality>0.2)&(alpha.detach()>0.1)
        depth_loss=((rendered[...,3][mask].clamp_min(1e-5).log()-depth[mask].log()).abs()*quality[mask]).mean() if mask.any() else rendered.sum()*0
        offset=params["means"]-anchor
        # Metric scale normalization; constrain normal displacement and discourage tangential drift.
        surface_loss=(offset*anchor_normal).sum(-1).abs().mean()/0.01 + offset.square().sum(-1).mean()/0.01**2*0.05
        dense_loss=rendered.sum()*0
        match=next((p for p in geometry_pairs if p["a"]==view_index or p["b"]==view_index),None)
        if cfg.get("dense_match_weight",0)>0 and match is not None:
            ia,ib=match["a"],match["b"]
            other_index=ib if view_index==ia else ia
            other=views[other_index]
            uv0=match["pixels_a"] if view_index==ia else match["pixels_b"]
            uv1=match["pixels_b"] if view_index==ia else match["pixels_a"]
            count=len(uv0);sample=np.random.default_rng(step+812).choice(count,min(256,count),replace=False)
            uv0=uv0[sample];uv1=uv1[sample]
            rd0=rendered[...,3][uv0[:,1],uv0[:,0]]
            rendered_other,alpha_other=render(other)
            rd1=rendered_other[uv1[:,1],uv1[:,0],3]
            visible=(rd0>0)&(rd1>0)&(alpha.detach()[uv0[:,1],uv0[:,0]]>.1)&(alpha_other.detach()[uv1[:,1],uv1[:,0]]>.1)
            if visible.any():
                def render_world(view,uv,z):
                    K=view[4];pix=tensor(uv.astype(np.float32));depth=z
                    cam=torch.stack(((pix[:,0]-float(K[0,2]))*depth/float(K[0,0]),(pix[:,1]-float(K[1,2]))*depth/float(K[1,1]),depth,torch.ones_like(depth)),dim=-1)
                    c2w=tensor(np.linalg.inv(view[5]))
                    return (c2w@cam.T).T[:,:3]
                xyz0=render_world(view,uv0,rd0);xyz1=render_world(other,uv1,rd1)
                dense_loss=torch.nn.functional.smooth_l1_loss(xyz0[visible],xyz1[visible],beta=.01,reduction="none").sum(-1).mean()/.01
        loss=rgb_loss+cfg["depth_weight"]*depth_loss+cfg["surface_weight"]*surface_loss+cfg.get("dense_match_weight",0)*dense_loss
        if not torch.isfinite(loss):raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():
            params["log_scales"].clamp_(np.log(0.0001),np.log(0.05))
            params["opacity_logits"].clamp_(-8,8)
        if step%10==0 or step+1==cfg["iterations"]:
            entry={"step":step+1,"loss":float(loss.detach()),"rgb":float(rgb_loss.detach()),"depth":float(depth_loss.detach()),"surface":float(surface_loss.detach()),"dense_match":float(dense_loss.detach())}
            history.append(entry)
            _progress(out, (step+1)/cfg['iterations'], f"训练 {step+1}/{cfg['iterations']} loss={entry['loss']:.6f}")
            write_json(out/"loss.json",history)
    cpu={k:v.detach().cpu() for k,v in params.items()}
    torch.save({"parameters":cpu,"config":cfg,"dependencies":request["dependencies"],"format":"SH0 RGB+depth+anchor baseline"},out/"checkpoint.pt")
    means=cpu["means"].numpy();quats=cpu["quats"].numpy();quats/=np.linalg.norm(quats,axis=1,keepdims=True)
    colors=cpu["color_logits"].sigmoid().numpy();dc=(colors-0.5)/0.28209479177387814
    data=np.column_stack([means,np.zeros_like(means),dc,cpu["opacity_logits"].numpy(),cpu["log_scales"].numpy(),quats]).astype("<f4")
    properties=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    with (out/"gaussians.ply").open("wb") as f:
        header="ply\nformat binary_little_endian 1.0\ncomment SH0 Gaussian model, not ordinary surface vertices\nelement vertex "+str(len(data))+"\n"+"".join("property float "+p+"\n" for p in properties)+"end_header\n"
        f.write(header.encode());f.write(data.tobytes())
    with torch.no_grad():
        psnrs=[]
        for view in views:
            r,a=render(view)
            rgb_render=r[...,:3].clamp(0,1)
            mse=(rgb_render-tensor(view[1])).square().mean().clamp_min(1e-10)
            psnrs.append(float(-10*torch.log10(mse)))
            image=(rgb_render.cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(str(out/("render_"+view[0]+".png")),image[:,:,::-1])
        novel_meta=[];video=None;height,width=views[0][2].shape
        for view_index in range(len(views)-1):
            first,second=views[view_index],views[view_index+1]
            c2w_first,c2w_second=np.linalg.inv(first[5]),np.linalg.inv(second[5])
            rotations=Slerp([0,1],Rotation.from_matrix(np.stack([c2w_first[:3,:3],c2w_second[:3,:3]])))
            for sample in range(1,4):
                amount=sample/4
                c2w=np.eye(4,dtype=np.float32)
                c2w[:3,:3]=rotations([amount]).as_matrix()[0]
                c2w[:3,3]=(1-amount)*c2w_first[:3,3]+amount*c2w_second[:3,3]
                K=((1-amount)*first[4]+amount*second[4]).astype(np.float32)
                dummy=("novel",None,np.zeros((height,width),dtype=np.float32),None,K,np.linalg.inv(c2w),None)
                novel_render,_=render(dummy)
                bgr=(novel_render[...,:3].clamp(0,1).cpu().numpy()[:,:,::-1]*255).astype(np.uint8)
                novel_index=len(novel_meta);name=f"novel_{novel_index:04d}.png"
                cv2.imwrite(str(out/name),bgr)
                if video is None:video=cv2.VideoWriter(str(out/"novel_view.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),15,(width,height))
                video.write(bgr)
                novel_meta.append({"file":name,"source_a":first[0],"source_b":second[0],"t":amount})
        if video is not None:video.release()
        write_json(out/"novel_views.json",novel_meta)
    from gsplat.exporter import export_splats
    export_splats(torch.as_tensor(means),cpu["log_scales"],torch.as_tensor(quats),cpu["opacity_logits"],torch.as_tensor(dc)[:,None,:],torch.empty((len(means),0,3)),format="splat",save_to=str(out/"gaussians.splat"))
    idx=np.linspace(0,len(means)-1,min(25000,len(means))).astype(int)
    write_json(out/"preview.json",{"positions":means[idx].ravel().tolist(),"colors":colors[idx].ravel().tolist(),"total_points":len(means),"sampled_points":len(idx),"label":"Gaussian centres, not splat rendering"})
    return {"backend":"gsplat 1.5.3 SH0 high-density RGB-D geometry","gaussians":len(means),"iterations":cfg["iterations"],"seconds":time.time()-started,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"final_loss":history[-1],"mean_train_psnr":float(np.mean(psnrs)),"dense_match_pairs":pair_summary,"novel_views":novel_meta,"crop":{"min_depth_m":cfg["crop_min_depth"],"max_depth_m":cfg["crop_max_depth"],"roi":[cfg["roi_x"],cfg["roi_y"],cfg["roi_w"],cfg["roi_h"]]},"limitations":["interpolated novel views stay near the captured camera path","no adaptive densification","no neural SDF or rendered-normal supervision","fixed input camera poses","browser centre preview is diagnostic; PNG/MP4 outputs use true Gaussian rasterization"]}
