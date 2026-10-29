# 模块边界与扩展

## 当前执行路径

浏览器 → `studio_server`（回环HTTP、请求校验、依赖图、串行队列）→ 子进程。

| 阶段 | 执行实现 | 输出契约 |
|---|---|---|
| import | studio_geometry / reconstruction库 | manifest、原图副本、哈希 |
| depth / sgbm | studio_geometry / OpenCV | depth/quality TIFF、RGB、K |
| depth / bridgedepth | studio.worker / pipeline | 同上，另有NPY |
| poses | ALIKED+LightGlue / COLMAP incremental SfM+global BA | 米制 T_world_camera、匹配统计、完整 COLMAP 稀疏模型 |
| cloud | studio_geometry / Open3D | 逐帧多边形/深度过滤、带色彩与法线的PLY |
| tsdf | studio_geometry / Open3D | 同掩膜 Scalable TSDF、PLY/OBJ表面与预览 |
| train | studio.worker / gsplat | Gaussian PLY、checkpoint、loss、渲染 |
| archive | studio.worker | 场景ZIP，不含模型和环境 |

算法不能自行标记HTTP任务成功。执行器检查退出状态并维护status.json；result.json只在阶段成功结束时写入。原生重建和训练 worker 以原子 `progress.json` 发布阶段、百分比和消息，场景 API 合并后供页面轮询。

正式离线定位只使用左目 RGB：ALIKED 提取关键点，LightGlue 对全部图像对建立对应，COLMAP 执行几何验证、增量 SfM、稀疏三角化和固定内参全局 BA。BridgeDepth 不参与位姿初始化；BA 完成后以 COLMAP 稀疏点的相机深度和 BridgeDepth 观测鲁棒估计一个全局尺度。随后按米制 COLMAP 位姿重新融合掩膜点云和 TSDF。

## 增加新后端的顺序

1. 将上游源码、许可证与固定版本放入3rdparty；权重放models，记录SHA256、上游地址、代码版本及使用限制。
2. 在独立适配器中转换到统一数据契约；不要在上游源码里混入业务逻辑。优先通过文件/CLI边界隔离CUDA/Python依赖。
3. 在libs/core.cpp加入显式阶段/参数验证与依赖；apps/server.cpp增加受控分派，禁止网页传入任意命令行。
4. 确认图像缩放后的K、米制尺度、相机坐标、pose方向、无效深度以及置信度含义。
5. 加入小样本测试、GPU峰值记录、错误退出与取消测试；未通过前网页标记实验性或未接入。

OmniVGGT、SIFT+RGB-D PnP 与 ICP 不再进入默认依赖图。历史运行保留用于回溯；新的 cloud、TSDF 和 train 只消费当前成功的 COLMAP 米制位姿。SDF/法线约束仍需新增带版本的表面产物依赖和损失实现。

## 8GB预算原则

单GPU任务串行、图像CPU缓存、逐帧深度与逐视图训练。优先320/640宽验证，再逐步提高分辨率与高斯数量。首次CUDA编译缓存保存3rdparty/torch_extensions；它依赖本机GPU、编译器和PyTorch版本，不跨机器复制使用。

20帧是数据规模上限，不是显存保证。每个新增后端都须独立测量；本机其他GPU程序占用计入可用显存，不自动结束外部程序。
