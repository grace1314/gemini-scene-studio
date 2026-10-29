# 数据契约 v1

- 单位米，深度为相机Z；OpenCV相机坐标（右、下、前）。
- `T_world_camera`：相机到世界；投影/TSDF外参使用逆矩阵。
- 入口仅接受已校正的同步双目图；v1不猜测原始畸变与校正矩阵。
- 左右图尺寸一致，校正后相同K、水平基线、零视差主点约定。
- 若不满足以上条件，应先独立校正并保存新的K与基线。

导入目录包含1至20个子目录，每个子目录有：

```
left.png
right.png
K.txt          第一行9个浮点数组成3x3 K；第二行baseline_m
capture.json   可选，原始采集元数据
```

导入时必须明确确认已校正；图像/标定复制到项目数据目录，保留SHA256及来源。
不同目录可有不同K，但图像尺寸和K必须自洽。基线正数，深度为fx*B/disparity。

场景：`data/scenes/<id>/manifest.json`与`frames/<id>/...`。
运行：`data/scenes/<id>/runs/<uuid>/request.json, status.json, log.txt, result.json`。
每次运行记录输入数据revision及上游run ID；重跑上游后，下游历史仍可下载但不再视为当前有效结果。
参数严格验证；失败不发布result.json。任务串行，重启时遗留运行标记interrupted。

深度：float32米制TIFF（C++主契约，BridgeDepth另有NPY）、有效掩码、预览PNG、对应缩放K、质量分数（非校准概率）。
点云：含xyz、rgb、normal的PLY；预览为抽样JSON，原始完整点云可下载。
位姿：每帧T_world_camera、匹配数/内点数/重投影统计；第一帧固定。
训练：gsplat checkpoint、标准SH0 Gaussian PLY、训练视图渲染、loss history；不把训练视图指标当泛化指标。

场景 `processing` 保存训练/融合选择：`crop_min_depth`、`crop_max_depth`（米）、`voxel_size`（米）和 `frame_rois`。`frame_rois` 是以帧 ID 为键的归一化多边形顶点数组 `[[x,y],…]`；空数组/缺失表示全画面，非空多边形为 3–128 点且面积必须非零。旧版 `roi_x/y/w/h` 保留兼容但新页面固定为全画面。

位姿阶段读取完整深度结果；点云与 TSDF 才应用逐帧多边形和 Z 范围。3DGS 从筛选后的点云初始化，并只在同一逐帧掩膜内计算 RGB、深度及稠密匹配监督。原始 RGBD 从不被掩膜覆盖。每个运行的 `request.json` 固化数值参数；`manifest.json` 固化当前逐帧掩膜，历史产物不被改写。
