# C50 MASK / INJECT 第一阶段执行报告

日期：2026-09-17。执行位置：远程 `/root/autodl-tmp/lerobot/lerobot-main`，代码基线 `det @ 5abe98ad`。源文件修复与工具同步保存在本地，实验数据与计算在远程执行。

本次完成八维数据审计、MASK 正确性修复、三个 episode 的分割预览、全尺寸模型梯度验证、两种分割损失的小样本对照，以及连续动作块评估工具。未启动三组正式 100k 训练；INJECT 结构尚未改动。

## 1. 新发现及其含义

MASK 存在两层问题：原实现先把损失 `.item()`，导致没有分割梯度；修复之后，单纯逐像素 L1 在本次小样本拟合中又退化为全背景。

episode 0 的 32 个均匀采样训练帧中，杯子前景只占约 2.0192% 像素。两种损失使用相同 seed、相同模型初始化、相同采样帧、相同 300 步及分支学习率，结果如下：

| 对照 | 初始 IoU / Dice | 300 步 IoU | 300 步 Dice | 预测前景比例 |
|---|---|---:|---:|---:|
| 修复后的 L1 | 0.0244 / 0.0447 | 约 0 | 约 0 | 0% |
| BCEWithLogits + soft Dice | 0.0244 / 0.0447 | 0.9666 | 0.9830 | 2.0224% |

L1 对照最终像素 L1 约 0.020192，恰好接近前景比例；低损失来自预测全背景。BCE + Dice 对照最终像素 L1 约 0.000735，且学到了杯子区域。

这是训练集上的分割拟合诊断，评价标签为 SAM 2 伪标签，不是人工真值；不代表验证集分割精度、机器人成功率或动作流畅度。拟合专用学习率为 backbone / FPN 各 1e-4、decoder 1e-3，仅用于确认可学习性，不直接作为正式联合训练设置。

建议正式短试优先验证 `mask_loss_type=bce_dice`，辅助权重 0.1 仍只是待验证起点。保留 L1 接口用于对照与旧配置兼容。正常动作推理仍不需要 masks 或 SAM。

## 2. 数据与标签准备

`数据集/formal1_C50_nomaster` 的 35,900 帧审计通过：

- state 字段严格为八个既定 follower 字段，与原数据前八维完全一致，没有 `master_gripper.pos`。
- action、timestamp、episode_index、frame_index 与原数据相同。
- train40 / val10 划分未改动；数值统计计数为训练 28,720 帧。
- 50 个 episode 的 35,900 帧 cup 检测框齐全，视频文件及各 episode 的局部时间映射存在。

生成器按 episode 的视频 chunk / file / from_timestamp 解码，逐帧核对时间，避免用全局 data index 直接寻址视频。采用 SAM 2.1 large，仅 frame 0 的 cup 框作为初始提示，输出原尺寸二值伪标签，未加高斯模糊。传播缺帧会报错，未自动补零。

训练 episode 0、20、49 各 718 帧，共 2,154 帧生成完成。标签只保存在预览输出目录，没有放入正式数据集的 `annotations/masks`。

抽查了三个 episode 各六个画面，并查看 episode 49 的异常附近六帧：

- episode 0：抽查中跟踪大致合理，夹爪遮挡时主要保留杯子可见部分。
- episode 20：第 120、717 帧附近边缘有疑似背景溢出，需要修正提示 / 轮廓复核。
- episode 49：第 340、529 帧附近面积跳变超过 50%；相邻图像变化不足以直接解释全部跳变，需排查局部漏分、恢复和溢出。

自动检查均未发现空掩码，但这不足以说明标签准确。下一阶段应先验证额外关键帧提示、背景负点和遮挡轮廓规则，再批量生成；本次不把全部预览标签标为已通过质量验收。

SAM 运行依赖安装在 `/root/autodl-tmp/c50-mask-env`，复用主环境现有 torch，SAM 源码以独立 PYTHONPATH 引入，未升级主训练环境。源码为官方仓库 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`；依赖清单在实验目录。最初大仓库下载缓慢，已停止该下载进程并改用最小源码检出，未删除旧数据。

## 3. 修复与回归结果

源文件改动：

- `modeling_act_det.py`：损失 Tensor 保留计算图，只有日志转为数值；按有效像素归一化一次；明确记录监督覆盖率。
- `mask_loader.py`：训练使用严格加载；缺失、损坏、非有限或越界标签报错。可显式声明无效帧，只忽略该帧，不放弃整批。
- `mask_decoder.py`：增加 logits 返回路径与 BCE + Dice 损失，保留默认概率输出和原有 checkpoint 参数名。
- `configuration_act_det.py`：新增 `mask_loss_type=l1|bce_dice`，默认 l1；非法值报错。
- `offline_eval_act_det.py`：将 MASK Tensor 转为日志数值，按有效像素累积 MASK 指标，避免无效帧扭曲均值。

原代码先通过真实模型回归复现失败；修复后 11 项 MASK 测试通过。内容包括真实损失公式、decoder / FPN / backbone 梯度、两种损失的 batch 重复不变性、缺标签与损坏文件、显式无效帧、全无效帧覆盖率、零权重、checkpoint 重载和无标签动作推理。

在 640×480 双相机、八维状态、六维动作、chunk_size=100 的实际模型上也验证通过：分割损失能独立训练 decoder、FPN、backbone；检测辅助损失同时存在；移除 mask loader 后动作推理一致。BCE + Dice 初始分割梯度范数分别约为 22.74 / 55.13 / 148.83，说明监督有效，但不能据此认定与动作损失已平衡。后续联合训练还需比较各损失梯度及动作表现。

## 4. 连续轨迹评估工具

新增 `src/lerobot/scripts/offline_eval_act_sequence.py`：顺序保存完整动作块、原单位与归一化预测、目标、padding 有效位和状态；统计 h0 / horizon 误差、相邻预测对同一绝对时刻的冲突、队列执行长度 1/5/10 与 coeff=0.01/0.03/0.1 的离线重放、目标差分及分段耗时。

3 项时间指标测试通过：在线集成与直接对角加权结果一致；一致的绝对时刻目标没有虚假冲突；padding 不进入预测冲突指标。

使用旧 DET 100k checkpoint、episode 7 前 32 帧，并在预处理前将 raw master 置零，真实数据冒烟通过。该片段的 h0 归一化 arm / gripper MAE 约为 0.097609 / 0.013718，仅用于工具验证，不能与完整验证集均值直接比较。

此时 SAM 同时运行，测得的网络延迟受到资源争用影响，不作为部署速度判断。工具使用演示状态与图像，是 teacher-forced 回放，不能报告闭环成功率或声称已消除真机抽搐。阶段事件与真实部署电脑的相机 / 串口测量属于后续工作。

## 5. 目录与可运行入口

远程记录统一放在 `outputs/mask_inject_phase1_20260917/`：

- `phase1_status.json`：阶段状态，`formal_training_ready=false`。
- `mask_tests_before.log` / `mask_tests_after.log` / `sequence_tests.log`：前后回归。
- `mask_preview/`：对齐审计、三个 NPZ、抽查图、异常图及 `review.json`。
- `mask_training_check/`：L1 300 步拟合结果。
- `mask_training_check_bce_dice/`：BCE + Dice 300 步拟合结果及预测图。
- `sequence_smoke_det/`：动作块与轨迹指标。
- `source_before/`：五个原源文件备份；`sam_environment.txt` / `sam_source_commit.txt`：分割环境记录。

重新运行回归测试：

```bash
cd /root/autodl-tmp/lerobot/lerobot-main
export PYTHONPATH="$PWD/src"
/root/miniconda3/bin/python tests/policies/act_det/test_mask_supervision.py
/root/miniconda3/bin/python tests/policies/act_det/test_sequence_metrics.py
```

仅重新审核数据、不生成标签：

```bash
/root/miniconda3/bin/python scripts/prepare_c50_mask_preview.py \
  --dataset-root 数据集/formal1_C50_nomaster \
  --output outputs/mask_inject_phase1_20260917/mask_preview --dry-run
```

连续评估一个新八维 checkpoint 的命令模板；先将 `CHECKPOINT` 换成真实路径，`OUT` 使用不存在的新目录：

```bash
CHECKPOINT=outputs/train/C50_NOMASTER_DET_smoke_s1000/checkpoints/000020/pretrained_model
OUT=outputs/mask_inject_phase1_20260917/sequence_nomaster_det_new
/root/miniconda3/bin/python -m lerobot.scripts.offline_eval_act_sequence \
  --checkpoint "$CHECKPOINT" --repo-id QYyyyyyyy/formal1_C50_nomaster \
  --dataset-root 数据集/formal1_C50_nomaster --episodes 7 \
  --max-frames 32 --output "$OUT"
```

以上八维冒烟 checkpoint 不是完成训练的模型。正式评估去掉 `--max-frames`，统一验证 episode，并记录配置。生成 / 拟合入口默认拒绝覆盖已经存在的实验，重复实验需显式使用新输出目录。

## 6. 下一阶段顺序

1. 修正并重新审核分割伪标签，验证关键帧纠正提示策略，再生成完整 C50 标签。
2. 完成原 train40 内的 32/8 开发划分及对应统计；固定标签 / 数据 / 公共权重摘要。
3. 实现并回归 I1 的 p4 残差注入，验证关闭注入等价于相同权重 DET，保留 I0 对照。
4. 在等预算短试中比较 DET / I0 / I1 / MASK-BCE-Dice，重点看动作准确与相邻块一致性，而非分割拟合成绩。
5. 更新本次 DET / INJECT / MASK 专用训练和 readiness 检查，重新测三组资源与存储预算后进入各 100k 训练。

旧 readiness 只针对 ACT / DET / INJECT 的既有冒烟，不适用于新的 MASK 组；本次没有复用它启动正式训练。所有有限预览和小样本检查均已结束，旧模型、原数据、stash 与用户无关修改保留。
