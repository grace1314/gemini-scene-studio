# 代码、数据与依赖归档

## 两个根目录

开发源码可以位于任意工作目录，例如：

```text
$HOME/src/gemini-scene-studio
```

推荐的实际运行安装位置为：

```text
$HOME/Tars_Project/gemini-scene-studio
```

如果采用开发/部署双目录，修改源码后需要显式同步/构建，不能假定两个目录自动一致。运行数据以部署目录下的 `data/scenes` 为准；不要为了“整理”把场景运行历史复制进 Git。

## 推荐分层

```text
gemini-scene-studio/
├── apps/ include/ libs/ studio/ web/   # 自研源码
├── tests/                              # 单元与端到端测试
├── scripts/                            # 运维、诊断、一次性迁移工具
├── docs/                               # 设计、流程、审计与实验记录
├── 3rdparty/                           # 固定版本外部源码/网页库/ABI 库
├── models/                             # 权重、来源和哈希，不进 Git
├── build/                              # CMake 生成物，可重建，不进 Git
└── data/scenes/<scene-id>/             # 原始帧与不可变运行历史，不进 Git
```

`.venv`/Conda 环境不是 3rdparty 源码，必须保留包管理元数据；CUDA JIT 缓存也是可重建缓存，不应和模型权重混为一类。

## 单个场景

```text
data/scenes/<scene-id>/
├── manifest.json              # 帧索引、哈希、revision、处理设置、ROI
├── frames/<frame-id>/
│   ├── left.png
│   └── right.png
├── runs/<run-id>/
│   ├── request.json           # 阶段、配置、依赖
│   ├── status.json            # queued/running/succeeded/failed/cancelled
│   ├── log.txt
│   ├── result.json
│   └── ...阶段产物
└── trash/                     # 网页删除帧后的可恢复数据
```

`active` 不是一个单独目录，而是服务按 `revision + dependencies` 动态算出的有效链。历史 run 仍然有审计价值，但不参与当前页面结果。

## 当前磁盘状态

源码侧约 1 GiB，其中 `3rdparty/` 约 799 MiB、`build/` 约 124 MiB、`models/` 约 49 MiB。当前真实场景原始帧约 42 MiB，运行历史约 1.7 GiB。

使用只读清单工具：

```bash
cd "$HOME/src/gemini-scene-studio"
python3 scripts/scene_inventory.py \
  "$HOME/Tars_Project/gemini-scene-studio/data/scenes/<scene-id>"

# 机器可读格式
python3 scripts/scene_inventory.py <scene-dir> --json
```

工具会复用网页的 active-run 判定逻辑，分别统计当前有效链和历史候选，**不会删除或移动任何文件**。

## 归档与清理策略

1. `frames/`、`manifest.json` 和当前 active 依赖链始终保留。
2. 先用网页/脚本导出整个场景 ZIP，再处理历史 run。
3. 对有价值的算法对比运行保留 `request/status/result/log` 和主要产物，并在实验记录中注明 run ID。
4. 失败运行通常只需日志，但删除前仍要人工确认。
5. `build/`、CUDA extension cache、缩略图等可重建内容可以作为第一批清理对象。
6. 不使用通配符递归删除场景根目录；清理工具未来应只接受清单中明确的 run ID，并默认 dry-run。

当前场景的生成报告见 [SCENE_DC5A903752A0.md](SCENE_DC5A903752A0.md)。
