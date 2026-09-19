# C50 MASK / INJECT 第二阶段执行报告

本阶段不启动 100k 训练、不控制机械臂；保持八维无主臂信号数据和既有 train40 / val10 划分。

## INJECT：I1 残差注入

新增配置 `fcos_inject_mode="residual"`、`fcos_residual_alpha=0.05`，同时设置 `fcos_feature_inject=True`、`fcos_inject_levels=["p4"]`、`mask_feature_inject=False`、`use_mask_guidance=False`。

动作编码图像特征为 `F_DET + clamp(alpha, 0, 1) * Z_FCOS`。保留原 centerness gate，不额外加入 p4 tokens；alpha 是可学习标量，初始 0.05 只是待验证的候选值。旧 token 注入仍为默认模式，不新增旧模型所需的参数键。

alpha 使用直接参数加 clamp，支持精确零权重回退。注意原始参数越界后 clamp 梯度为零，因此短训必须同时记录原始值和有效值；若持续贴边，先处理优化/参数化，不宣称已经学到合适注入权重。

远程运行 `PYTHONPATH=src /root/miniconda3/bin/python tests/policies/act_det/test_residual_injection.py`：6 项通过，检查共享权重下 alpha=0 与 DET 的逐元素精确一致、正权重改变动作并向注入分支传播梯度、旧 token 模式、配置约束、权重范围、checkpoint 保存/加载及 encoder token 数。MASK 原有 11 项和时序指标 3 项也通过。

上述验证使用小分辨率模型，证明实现与回归约束，不证明真机流畅性或 I1 比 I0 更好。旧 checkpoint 若用命令行强行切换 residual，会缺少新参数；下一阶段应从共同初始化创建对照模型，而非当作已经训练好的 I1。

另在远程 CUDA 上做 640×480 双相机、八维状态、100 步动作块随机输入检查：输出 `[1,100,6]` 有限，alpha=0 与共同权重 DET 最大误差精确为 0。检查结果保存为 `residual_fullsize_check.json`；与 SAM 并行期间不测部署延迟。

## MASK：纠正提示对照

原始三 episode 标签保留在 `outputs/mask_inject_phase1_20260917/mask_preview/`。

新候选标签使用训练 episode 0 / 20 / 49，共 2,154 帧；在帧 0、120、240、340、480、529、717 加入检测框纠正提示，并在框外 25 像素处加入图像范围内的背景负点。未硬裁剪 mask，未改变正式数据集标注。候选目录为 `mask_preview_corrected/`，远程 screen 名称为 `c50_mask_corrected`。

复现生成命令（目录已有结果会拒绝覆盖，重新实验需新输出目录）：

```bash
cd /root/autodl-tmp/lerobot/lerobot-main
export PYTHONPATH=/root/autodl-tmp/c50-mask-env/sam2-minimal:src
/root/autodl-tmp/c50-mask-env/bin/python -u scripts/prepare_c50_mask_preview.py \
  --dataset-root "$PWD/数据集/formal1_C50_nomaster" \
  --output outputs/mask_inject_phase1_20260917/mask_preview_corrected \
  --episodes 0 20 49 --prompt-frames 0 120 240 340 480 529 717 \
  --negative-margin 25 --checkpoint checkpoints/sam2.1_hiera_large.pt
```

框外负点和面积跳变仅作质量控制；检测框不等于像素真值，改善这些代理指标仍需视觉核查遮挡和夹爪边界。生成完毕后比较三 episode 的框外面积比例和突跳，并复查 339–341 / 528–530 帧。

结果：三 episode 全部完成，共 2,154 帧，无空帧。比较如下：

| episode | 平均框外 mask 面积占比：原始 → 纠正 | 框外占比超过 10% 的帧数 | 面积突跳帧：原始 → 纠正 |
| --- | --- | --- | --- |
| 0 | 0.336% → 0.120% | 11 → 0 | 无 → 无 |
| 20 | 7.327% → 1.843% | 251 → 8 | 无 → 400 |
| 49 | 1.094% → 0.972% | 4 → 0 | 340、529 → 280、314、332 |

逐图复查 episode 0 / 20 的六个常规采样点，以及 episode 20 / 49 的异常附近帧。episode 20 在 399→400 帧，mask 面积从 5,383 变成 8,107，新增部分明显伸到框下方；episode 49 的 313→314→315、331→332→333 在近似图像下明显切换杯体覆盖范围。原 340 / 529 的 >50% 突跳被消除不代表时间稳定性已经解决。

结论：纠正提示能减少边缘溢出，但本候选未通过质量门槛，不晋升正式训练标签、不生成全量标签。下一次应针对实际遮挡边界增加经核查的杯体正点及夹爪/背景负点，或逐帧框提示与视频传播做小样本对照；在有像素级核查样本前，不盲目增加更多定时提示。不要对 mask 硬裁框并把代理指标变好当作准确率提高。

NPZ 中 `valid=True` 只表示成功生成该帧，尚不是人工质量认可。审查状态另存 `mask_preview_corrected/review.json`；量化结果为 `mask_preview_comparison.json`，异常邻帧图为 `episode_*_flagged_preview.jpg`。

## 下一道门槛

1. 标签质量审查通过后才生成完整标签；明确遮挡时只标可见杯体，不能把夹爪当杯体。
2. 从 train40 内固定划分 32 / 8 做短训选择，避免用已反复查看的固定 val10 充当盲测；预处理和统计保持无主臂信号一致。
3. 同预算 DET / I0 / I1 / MASK 短训，记录第 0 步机械臂与夹爪误差、跨块冲突、轨迹差分、检测/分割指标及动作与辅助梯度。MASK 使用 BCE + Dice，不用已证实容易全背景塌缩的 L1 候选。
4. 通过模型行为、显存和保存空间检查后，再准备最终三组 100k screen 并行训练。现有 ACT / DET / INJ 旧启动器不直接当作本次 DET / INJECT / MASK 的训练准备完成证明。
