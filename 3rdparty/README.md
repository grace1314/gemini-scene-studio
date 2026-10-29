# 第三方组件

自研代码仅放 apps/、libs/、include/、studio/、web/。本目录保存外部组件，遵循各自许可证。

| 目录 | 版本 / 来源 | 用途 |
|---|---|---|
| include/httplib.h | cpp-httplib v0.20.0 | 本地C++ HTTP服务 |
| include/nlohmann/ | nlohmann/json v3.11.3 | JSON序列化 |
| open3d/ | Open3D v0.19.0 官方Linux cxx11 ABI CPU发行包 | 点云/TSDF/位姿图 |
| open3d/lib/libc++*.so.1 | Open3D 0.19.0官方Python wheel内运行库 | 官方C++包依赖 |
| BridgeDepth/ | aeolusguan/BridgeDepth `9a231601e698`，由脚本获取 | 双目深度模型 |
| LightGlue/ | cvg/LightGlue `5a650c444718`，由脚本获取 | ALIKED特征与LightGlue匹配 |
| three/ | Three.js 0.170.0 | 离线网页三维预览 |
| splat-viewer/ | antimatter15/splat `ba182b51b7c2` + 本地交互补丁，随仓库发布 | 离线 WebGL Gaussian Splat 查看器（MIT）；按透明度加权质心自动聚焦，支持质心轨道旋转、平移、缩放与复位 |
| licenses/ | 上游许可证 | 许可证归档 |
| downloads/ | 上游开发包 | 可重现下载缓存 |
| torch_extensions/ | 本机CUDA编译产物，不提交Git | gsplat JIT编译缓存 |

大型下载源码不提交主仓库；`fetch_deps.sh`可恢复固定版本依赖。`archive_resources.py`是从已有工程迁移资源的可选路径。不要把模型权重放入此目录，权重统一在models/并由`fetch_models.sh`下载。

校验清单为本地下载后的SHA256记录，不应解释为上游签名认证。保留各上游LICENSE；本项目未替第三方授予额外使用权。
