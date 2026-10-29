# 项目文档导航

首次接手项目建议按以下顺序阅读：

1. [当前状态与审计](CURRENT_STATUS.md)：已经完成什么、当前效果、已知缺点和优先优化项。
2. [源码流程讲解](CODE_WALKTHROUGH.md)：从网页按钮到 C++ 服务、Python/C++ worker 和产物的完整调用链。
3. [数据与文件归档](DATA_LAYOUT.md)：源码、第三方库、模型、场景、运行历史和缓存分别放在哪里。
4. [3DGS 研究与升级路线](3DGS_ROADMAP.md)：为什么当前结果偏糊，以及 8 GB 显存下的实验顺序。
5. [数据契约](DATA_CONTRACT.md)：输入图像、标定、坐标系和输出格式。
6. [当前真实场景清单](SCENE_DC5A903752A0.md)：由只读工具生成的磁盘占用和有效运行链。

设计阶段资料仍保留在 [PLAN.md](PLAN.md)、[ARCHITECTURE.md](ARCHITECTURE.md)、[UI_SPEC.md](UI_SPEC.md) 和 [VALIDATION.md](VALIDATION.md)。它们记录演进过程；实现现状以前四份文档和源码为准。
