# 源码流程与代码含义

## 一次任务如何流动

```text
web/app.js
  POST /api/scenes/<id>/jobs
        │
        ▼
apps/server.cpp
  校验场景、revision、依赖和配置
  创建 runs/<run-id>/{request,status,log}
        │
        ├─ depth / poses(colmap_rgb) / train ─► studio/worker.py
        │                                          │
        │                                          ├─ pipeline.py
        │                                          ├─ colmap_rgb_adapter.py
        │                                          └─ training*.py
        │
        └─ cloud / tsdf ───────────────────────► C++ geometry worker
                                                   │
                                                   └─ libs/reconstruction.cpp
```

任务成功后写 `result.json` 和具体产物。网页再次请求场景时，服务根据当前 `manifest.revision` 和依赖 run ID 计算 `active`；不是当前依赖链的旧成功任务不会被误当成当前结果。

## 源码目录职责

| 路径 | 作用 |
|---|---|
| `apps/server.cpp` | HTTP 服务、静态网页、相机代理、场景 CRUD、任务排队/取消、文件安全下载 |
| `apps/geometry.cpp` | 独立 C++ 几何 worker/CLI 入口 |
| `include/studio/*.hpp` | C++ 数据结构、配置和公共接口 |
| `libs/core.cpp` | manifest/run JSON、哈希、依赖有效性、文件与进程辅助 |
| `libs/reconstruction.cpp` | RGB-D 点云生成、体素融合、TSDF、预览抽样 |
| `studio/core.py` | Python 侧相同的数据契约、哈希检查和 active run 规则 |
| `studio/worker.py` | Python 阶段分发；每个任务独立进程，异常直接写日志并失败退出 |
| `studio/pipeline.py` | BridgeDepth 推理、输出 RGB/depth/quality 与相机内参 |
| `studio/colmap_rgb_adapter.py` | 准备深度依赖并调用正式 COLMAP runner |
| `studio/colmap_rgb_runner.py` | ALIKED/LightGlue 匹配、写 COLMAP 数据库、增量 SfM/BA、米制尺度恢复与轨迹导出 |
| `studio/training.py` | 当前网页实际使用的快速 3DGS 基线 |
| `studio/training_quality.py` | 尚未默认启用的 SH、SSIM、自适应密度实验训练器 |
| `web/app.js` | 路由、页面模板、API 调用、任务串联、ROI 编辑、结果加载 |
| `web/viewer.js` | 点云/网格/WebGL 诊断查看 |
| `3rdparty/splat-viewer/` | 真正的 Gaussian Splat 椭球渲染和相机交互（上游MIT项目加本地交互补丁） |
| `scripts/` | 构建、启动、依赖归档、诊断和离线试验工具 |

## 各阶段输入与输出

### 1. 采集

`apps/server.cpp` 代理本机 Gemini 相机服务。网页只对实时预览使用压缩 JPEG；采集端保存同步、已校正的左右无损 PNG 到：

```text
frames/<frame-id>/left.png
frames/<frame-id>/right.png
```

每次新增、删除帧或保存 ROI 都会增加 `manifest.revision`，使旧下游运行自动失效。

### 2. BridgeDepth

网页由 `submit('depth', ...)` 发起任务，`studio/worker.py` 路由到 `pipeline.run_depth()`。正式推理读取原始 1280×800 双目，输出：

```text
runs/<depth-id>/<frame-id>/rgb.png
runs/<depth-id>/<frame-id>/depth.npy      # float32，米
runs/<depth-id>/<frame-id>/quality.npy
runs/<depth-id>/<frame-id>/depth.png      # 仅预览
runs/<depth-id>/result.json               # K、分辨率和统计
```

### 3. COLMAP SfM / BA

`colmap_rgb_runner.py` 只把左目 RGB 送入特征与 SfM。ALIKED 检测局部特征，LightGlue 建立图像对对应，随后 COLMAP 完成增量建图、稀疏三角化和全局 BA。BridgeDepth **不参与位姿初始化**；BA 后才以稀疏点的深度观测鲁棒估计一个全局米制尺度。

主要输出为 `poses.json`、COLMAP sparse model、匹配诊断和 `result.json`。下游统一使用 `poses.json` 的米制 `T_world_camera`。

### 4. 点云与 TSDF

`libs/reconstruction.cpp` 读取每帧 RGB、米制深度、K、位姿、多边形 ROI 和深度范围。点云阶段把有效像素反投影到相机坐标，再变换到世界坐标并体素降采样。TSDF 阶段按同一轨迹将 RGB-D 集成到 Open3D `ScalableTSDFVolume`，最终抽取网格。

点云和 TSDF 是两个并列运行：二者都依赖 depth + poses；train 依赖 depth + poses + cloud。TSDF 当前不是 train 的依赖，也没有持久化体素场。

### 5. 当前 3DGS

`training.py` 从融合点云抽样初始化 Gaussian：点坐标作为均值、点法线构造切平面旋转、局部邻距初始化尺度、点颜色初始化 SH0/颜色参数。每步从一个训练相机渲染 RGB 和期望深度，优化：

```text
L = L1_RGB
  + depth_weight × L_log_depth
  + surface_weight × L_anchor
  + dense_match_weight × L_cross_view
```

当前网页配置把数量固定为 80k、SH=0、SSIM=0、adaptive_density=false。`dense_match_pairs` 只说明预处理找到过对应，不保证训练中有有效梯度；本次最终该项为 0。

训练输出包括 `gaussians.ply`、`gaussians.splat`、`checkpoint.pt`、训练位姿渲染、新视角插值 PNG/MP4、`losses.json` 和浏览器诊断预览。

## 配置和状态的来源

- 场景长期配置：`manifest.json` 的 `processing`，包括深度范围、体素尺寸和每帧 ROI。
- 单次运行配置：`runs/<id>/request.json`；它是复现实验的主要依据。
- 生命周期：`status.json`；真正的算法摘要：`result.json`；控制台输出：`log.txt`。
- 网页当前默认训练参数：`web/app.js` 中的 `QUALITY_TRAIN_CONFIG`。这个名字目前有误导性，因为 `quality_mode` 实际为 false。

## 修改代码时的推荐边界

- 算法配置与验证放 C++ 服务端，网页只负责表单与提交，避免仅靠前端限制。
- Python 只保留深度网络、特征网络和可微训练；点云、TSDF、文件服务继续放 C++。
- 新 3DGS 后端使用新的 `backend`/`quality_mode` 和独立产物，不覆盖基线结果。
- 不直接修改 `data/scenes` 中已有产物；所有算法试验创建新 run。
- 退役后端先标记 legacy 并补迁移测试，再移动文件，避免破坏历史 `request.json` 的复现。
