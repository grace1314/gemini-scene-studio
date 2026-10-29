# Gemini Scene Studio — 实施计划

目标：Gemini 305、最多20组静态双目、RTX 4060 Laptop 8 GB、离线本地网页。
推荐部署位置：`$HOME/Tars_Project/gemini-scene-studio`。也可以通过 `GEMINI_STUDIO_DEPLOY_DIR` 指定其他本地目录。

## 顺序与验收

1. **计划与页面设计**：先完成本文件、UI_SPEC、数据契约。
2. **项目基础**：独立 Git 仓库、C++17/CMake 主工程与 HTTP 服务、Python 模型 worker、测试与启动脚本；不改已有项目/环境。
3. **真实任务系统**：场景创建、严格导入、串行子进程、日志、取消、运行历史、产物归档及下载。
4. **算法基线**：BridgeDepth、逐帧多边形掩膜、ALIKED+LightGlue、COLMAP SfM/全局BA、深度观测米制尺度恢复、掩膜点云融合、Open3D Scalable TSDF。
5. **训练基线**：gsplat RGB+depth+表面锚定训练；限制高斯数，不宣称等价PGSR或完整SDF联合优化。
6. **依赖及权重**：独立环境；复用 BridgeDepth，归档 ALIKED/LightGlue 权重并记录 SHA256；COLMAP 作为正式相机位姿后端。PGSR 作为后续 3DGS 几何增强模块。
7. **验证**：合成多视图夹具测位姿方向/单位、TSDF、任务依赖；真实已有双目样本测深度；CUDA训练烟雾测试；网页各页与下载检查。

## 边界

- 当前相机未连接：采集页提供导入和采集格式说明，硬件采集接入单独标记未验证，不伪造实时流。
- 初始位姿主线为深度PnP；COLMAP格式导出不等于运行COLMAP SfM或rig BA。
- 20帧仍不保证任意大模型一次性装入8 GB。阶段串行、CPU存储、逐视图训练。
- 全部演示数据标注合成；真实场景未经测量不能宣称毫米精度。
- 暂不发布GitHub、不自动终止已有GPU任务、不安装系统驱动。

## 后续迭代

- Orbbec实时采集与硬件同步验证。
- LightGlue/RoMaV2匹配，完整rig BA，左右视图联合训练。
- 可微 SDF/法线约束、PGSR 适配与 8GB 显存评测。
- PGSR/Neural SDF、法线渲染监督、增密策略及真实留出集评估。

## 测试定义

CPU测试不要求GPU或相机。集成测试使用明确标记的合成静态场景。
GPU验证与真实数据验证分别记录，不用合成结果替代真实采集验证。
