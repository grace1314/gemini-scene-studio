# 3DGS 质量与几何升级路线

目标不是单纯让训练图“看起来更锐”，而是在 20 帧、8 GB 显存下同时获得可验证的新视角质量和有明确尺度/表面的几何。

## 当前结果为什么糊

1. **表达能力不足**：网页实际使用 SH0，视角相关外观只能被平均成一个颜色。
2. **密度固定**：80k 点从融合点云一次性抽样，纹理/轮廓复杂处不能增密，错误区域也不能剪枝。
3. **相机与深度误差被椭球吸收**：固定 COLMAP 位姿和单尺度深度存在残差时，Gaussian 会变大、变扁或变透明以降低 RGB loss。
4. **几何损失未生效**：本次最终 dense-match loss 为 0；表面锚定也不等于渲染法线或 SDF 约束。
5. **前景定义不完整**：18/20 ROI 使两帧背景进入训练，硬边界也会污染轮廓。
6. **评价方式不足**：训练 PSNR 仅 8.17 dB，且没有真正留出视角，不能用插值视频代替验证。

## Phase 0：先建立可信评价

- 补齐所有帧 ROI，或明确把帧排除；引入前景 alpha 和 3–5 px 软边界。
- 从 20 帧固定留出 2–4 帧，只用于验证；其余帧训练。
- 记录 PSNR、SSIM、LPIPS；几何记录深度 MAE/RMSE、法线角误差和对 TSDF/测量面的 Chamfer。
- 每次实验保存 config、随机种子、显存峰值、耗时、Gaussian 数量和 run ID。

没有 Phase 0，不应该根据单张截图判断算法升级是否成功。

## Phase 1：用现有 gsplat 做好可靠基线

项目已有 `training_quality.py`，但需先在当前数据上验证。8 GB 显存建议从以下配置开始：

| 项 | 快速预览 | 质量基线 A | 质量基线 B |
|---|---:|---:|---:|
| steps | 3000 | 6000 | 10000 |
| 初始 Gaussian | 80k 固定 | 30k | 40k |
| 最大 Gaussian | 80k | 120k | 180k |
| SH degree | 0 | 2 | 3 |
| SSIM weight | 0 | 0.2 | 0.2 |
| adaptive density | 否 | 是 | 是 |
| 分辨率 | 1280 | 960→1280 | 960→1280 |

使用 gsplat `DefaultStrategy` 做 split/duplicate/prune/reset，并设置明确的增密停止步。不要一开始就把上限设到百万级；20 帧小场景更需要干净的监督而非无上限堆点。

同时加入每相机曝光/白平衡补偿，避免固定曝光差被错误解释成几何或 SH。相机位姿先固定，质量基线稳定后再允许很小的 SE(3) 修正，并以 COLMAP BA 位姿为强先验。

## Phase 2：让 RGB-D 几何真正进入训练

### 深度

- 使用 ROI、左右一致性 quality、图像梯度共同加权的 edge-aware log-L1 深度损失。
- 在遮挡边界和反光/低纹理区域降低权重，不把 BridgeDepth 当成绝对真值。
- 对每帧允许轻量 scale/shift 校正，但以双目标定的米制尺度为正则中心。

### 法线与表面

- 从去噪深度和 TSDF 网格生成相机空间法线。
- 约束渲染深度梯度法线与参考法线，并加入局部平面一致性。
- 持久化 TSDF 体素、权重与梯度；对 Gaussian 均值采样 `|SDF(x)|`，对最短尺度轴约束其与表面法线对齐。
- 区分前景表面、自由空间和未知空间，不能只对零水平面做无符号最近邻锚定。

### 跨视角对应

移除 Farneback 作为核心几何来源。优先复用已经产生的 LightGlue/COLMAP tracks：对具有可靠深度的对应像素反投影为两侧 3D 点，经过优化位姿后约束世界坐标一致；按 LightGlue 分数、深度 quality、重投影误差和遮挡联合筛选。这样匹配与 SfM 使用同一证据链，也更容易诊断。

## Phase 3：几何优先的独立后端

如果 Phase 1–2 后 Gaussian 仍用厚椭球解释薄表面，再增加独立研究后端：

- **DN-Splatter 风格**：最适合当前 RGB-D 输入，可渐进加入深度、法线和 2D Gaussian 先验；改动相对可控。
- **2D Gaussian Splatting**：用有方向的平面盘代替 3D 椭球，并用深度畸变/法线一致性获得更清晰表面，适合网格重建。
- **PGSR**：加入无偏平面深度/法线、单视图与多视图几何约束以及曝光补偿，目标最贴近“明确几何意义”。

这些后端应使用同一 COLMAP 轨迹、同一 ROI、同一验证集和指标，以独立 `backend` 保存结果；不要直接覆盖现有 gsplat 基线。

## 实施顺序与验收门槛

1. `quality-v1`：SH2 + SSIM + densification。要求留出视角指标明显优于当前基线且没有显存溢出。
2. `geometry-v2`：置信深度 + 法线。要求深度/法线误差下降，视觉轮廓不变差。
3. `tracks-v3`：LightGlue track 3D 一致性。要求损失非零、内点数可解释、关闭该项时有可重复退化。
4. `sdf-v4`：持久化 TSDF/SDF 与自由空间约束。要求 Gaussian 表面厚度和 Chamfer 改善。
5. `2dgs/pgsr`：仅在上述基线仍达不到几何目标时引入。

## 主要参考实现

- [gsplat 官方仓库](https://github.com/nerfstudio-project/gsplat) 与 [DefaultStrategy](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/default.py)
- [Nerfstudio Splatfacto 官方实现](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)
- [2D Gaussian Splatting 官方实现](https://github.com/hbb1/2d-gaussian-splatting) 与 [论文](https://arxiv.org/abs/2403.17888)
- [DN-Splatter 官方实现](https://github.com/maturk/dn-splatter) 与 [WACV 2025 论文](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf)
- [PGSR 官方实现](https://github.com/zju3dv/PGSR) 与 [论文](https://arxiv.org/abs/2406.06521)
