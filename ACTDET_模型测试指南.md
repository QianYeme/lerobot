# ACTDet 模型测试指南（E1–E9 已训练模型）

本指南覆盖两类测试：

| 测试 | 脚本 | 指标 | 作用 |
|------|------|------|------|
| **离线评估**（无真机） | `offline_eval_act_det.py` | Action L1、检测损失、Mask 损失 | 辅助指标（实验计划 §6）；sanity check，反映模型在训练数据上的拟合质量 |
| **真机测试**（4 阶段成功率） | `control_act_det.py` | 抓取/移动/放下/整体 成功率 | **论文主指标**：机械臂自主执行 + 人工判定 + 录像 |

> ⚠️ 重要：离线评估使用每个数据集尾部 27 集（ep_063–089），这 27 集**曾参与训练**（训练用了全部 90 集），因此离线指标只反映拟合质量，不代表泛化能力；泛化结论以真机测试为准。

---

## 1. 实验–模型–数据集对照表

9 个实验全部训练完成（120000 steps）。checkpoint 目录统一为 `<运行目录>/checkpointsE<X>/last/pretrained_model`。

> 模型已上传 Hugging Face（15 个仓库），新机器上下载命令见 **《模型下载指南.md》**。

| 实验 | 版本 | 类型 | state | 检测 | Mask引导 | FCOS注入 | Mask注入 | 数据集 | 真机次数 |
|------|------|------|-------|------|----------|----------|----------|--------|----------|
| E1 | V1 基线 ACT | act | 6 维 | – | – | – | – | formal_A | **30（核心）** |
| E2 | V2 +检测 | act_det | 6 维 | ✓ | – | – | – | formal_A | **30（核心）** |
| E3 | V3 +Mask引导 | act_det | 6 维 | ✓ | ✓ | – | – | formal_A | 10 |
| E4 | 基线 ACT（力感） | act | 9 维 | – | – | – | – | formal1_B | 10 |
| E5 | V2（力感） | act_det | 9 维 | ✓ | – | – | – | formal1_B | 10 |
| E6 | V3（力感） | act_det | 9 维 | ✓ | ✓ | – | – | formal1_B | 10 |
| E7 | V4 +FCOS注入 | act_det | 9 维 | ✓ | – | ✓ | – | formal1_B | 10 |
| E8 | V5 +Mask注入 | act_det | 9 维 | ✓ | ✓ | – | ✓ | formal1_B | 10 |
| E9 | V6 双注入 | act_det | 9 维 | ✓ | ✓ | ✓ | ✓ | formal1_B | **30（核心）** |

checkpoint 路径（Ubuntu 训练机，`/root/autodl-tmp/lerobot/lerobot-main/` 下）：

```
outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model      # E1
outputs/train/2026-08-07/02-56-09_act_det/checkpointsE2/last/pretrained_model  # E2
outputs/train/2026-08-07/16-25-39_act_det/checkpointsE3/last/pretrained_model  # E3
outputs/train/2026-08-07/16-28-36_act/checkpointsE4/last/pretrained_model      # E4
outputs/train/2026-08-08/01-17-03_act_det/checkpointsE5/last/pretrained_model  # E5
outputs/train/2026-08-08/01-17-21_act_det/checkpointsE6/last/pretrained_model  # E6
outputs/train/2026-08-08/12-32-41_act_det/checkpointsE7/last/pretrained_model  # E7
outputs/train/2026-08-08/12-33-40_act_det/checkpointsE8/last/pretrained_model  # E8
outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model  # E9
```

- 数据集 A（formal_A）：6 维 state（5 关节 + gripper.pos），50ep 固定水位 + 40ep 随机水位 = 90ep
- 数据集 B（formal1_B）：9 维 state（5 关节 + gripper.pos + gripper.load + gripper.curr + master_gripper.pos），同样 90ep

> 测试时一律使用 `pretrained_model/` 目录（模型权重 + 配置）。同级的 `training_state/` 只保存优化器/调度器/随机数状态，仅用于 `lerobot-train --resume` 断点续训，无法用于测试。

---

## 2. 离线评估（offline_eval_act_det.py）

### 2.1 原理

脚本完全复刻训练时的前向路径：数据集按 `chunk_size=100` 组装动作块 → ACT 预处理管线（按数据集统计量归一化）→ 策略 **train 模式**前向（检测/Mask 损失只在 train 模式计算），逐帧累计：

- `l1_loss`：预测动作块 vs 记录动作的 L1（归一化空间，已掩掉 chunk 越界填充）
- `det_cls_loss` / `det_reg_loss` / `det_ctr_loss`：FCOS 检测分支（仅含检测分支的模型）
- `mask_loss`：pred_mask vs SAM2 真值 NPZ 的 L1（仅含 Mask 引导的模型）

### 2.2 运行命令

评估尾部 27 集（ep_063–089）。`act_det` 类型的 checkpoint 会自动从 `annotation_dir` 推断数据集；E1/E4（纯 act，无标注）需显式指定：

```bash
cd /root/autodl-tmp/lerobot/lerobot-main

# act_det 模型（E2/E3/E5/E6/E7/E8/E9）：无需写 dataset 参数
python src/lerobot/scripts/offline_eval_act_det.py \
    --checkpoint outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model \
    --episodes 63-89

# act 基线（E1/E4）：显式指定数据集
python src/lerobot/scripts/offline_eval_act_det.py \
    --checkpoint outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model \
    --dataset.repo_id formal_A \
    --dataset.root /root/autodl-tmp/lerobot/lerobot-main/数据集 \
    --episodes 63-89
```

参数说明：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--checkpoint` | 必填 | `pretrained_model` 目录 |
| `--dataset.repo_id` / `--dataset.root` | 自动推断 | 自动推断失败（纯 act）时才需要 |
| `--episodes` | `63-89` | 评估的 episode 范围，支持 `63-89`、`0,5,10-20` |
| `--batch-size` | 8 | 与训练一致，16GB 显存边界 |
| `--output` | checkpoint 目录下 | 结果 JSON 保存位置 |

### 2.3 输出与耗时

- 终端打印各指标均值；JSON 保存到 `<checkpoint>/offline_eval_results.json`
- 每模型约 10–20 分钟（27 集 × ~750 帧 / batch 8）
- train 模式下 dropout 与在线增强有随机性，数字有轻微噪声；对关键对比可跑两次取均值

### 2.4 结果汇总表（模板）

| 实验 | Action L1 | Det cls | Det reg | Det ctr | Mask L1 |
|------|-----------|---------|---------|---------|---------|
| E1 | | – | – | – | – |
| E2 | | | | | – |
| E3 | | | | | |
| E4 | | – | – | – | – |
| E5 | | | | | – |
| E6 | | | | | |
| E7 | | | | | – |
| E8 | | | | | |
| E9 | | | | | |

---

## 3. 真机测试（control_act_det.py，4 阶段成功率）

测试协议沿用论文方法：**预置固定杯位（跨模型复用）+ 人工分阶段判定 + 全程录像**。

### 3.1 测试前准备（一次性）

1. **机械臂**：已标定则直接连接（标定流程见 ACTDET_使用说明.md / ENVIRONMENT_SETUP.md）；串口号可用 `lerobot-find-port` 确认
2. **摄像头编号**：`lerobot-find-cameras` 确认 `top`（全局视角）与 `gripper`（腕部）对应的 `/dev/video*` 序号，填入 `--robot.cameras`
3. **杯位预置表（30 个位姿）**：按 §3.4 模板，首次测试前在桌面上实测确认每个杯位坐标并**拍照留档**。之后 9 个模型全部复用同一张表（控制变量）
4. **录像**：手机/相机架在正前方，每段测试全程录制；命名 `{实验}_{杯位编号}_run{n}.mp4`，如 `E9_C03_run1.mp4`
5. **安全**：leader 臂移出工作空间；每段开始前把从动臂摆到接近演示起始位姿（首条动作幅度可能较大）

### 3.2 运行命令（每模型一个会话）

```bash
cd /root/autodl-tmp/lerobot/lerobot-main

# E4–E9（9 维，formal1_B）示例：E9 核心模型
python src/lerobot/scripts/control_act_det.py \
    --policy.path outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model \
    --robot.type=so_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
    --dataset.repo_id formal1_B \
    --dataset.root /root/autodl-tmp/lerobot/lerobot-main/数据集 \
    --dataset.single_task="Cup pick and place" \
    --dataset.num_episodes 10 \
    --dataset.episode_time_s 60 \
    --dataset.reset_time_s 60

# E1–E3（6 维）只需把 dataset.repo_id 换成 formal_A，policy.path 换成对应 checkpoint
```

要点：

- **`--dataset.repo_id` 必须与该模型的训练数据集一致**（E1–E3 → formal_A，E4–E9 → formal1_B）：脚本按该数据集的 feature 名从机器人观测中取 state 维度（6 维自动取 5 关节+gripper.pos，9 维再附加 load/curr/master），并按该数据集的统计量做归一化
- `--robot.port` 与 `--robot.cameras` 的摄像头编号以实际为准
- `master_gripper.pos` 无 leader 臂时恒为 0.0，属预期行为（与 lerobot-record 的 policy 模式一致）
- 按键：`→` 提前结束当前段，`Esc` 结束整个会话

### 3.3 执行流程与判定

1. 每段开始：按杯位表摆放杯子（含旋转角）→ 从动臂摆到演示起始位姿附近 → 运行
2. 机械臂自主执行（每段最长 60s，任务完成可提前按 `→` 结束）
3. **人工判定 4 阶段**（依赖关系与实验计划一致）：

| 阶段 | 判定标准 | 依赖 |
|------|----------|------|
| 1. 抓取杯子 | 夹爪稳定夹住杯身且杯底脱离台面 | 独立 |
| 2. 移动杯子 | 平稳移向目标位置，水无大量洒出 | → 阶段 1 成功 |
| 3. 放下杯子 | 放到目标位置，松开夹爪后杯子直立不倒 | → 阶段 2 成功 |
| 4. 整体 | 三个阶段全部成功 | → 1、2、3 全部成功 |

4. 在判定表（§3.5）逐段记录 ✓/✗，对照录像可事后复核
5. 复位时间内按表摆放下一个杯位，继续下一段

### 3.4 杯位预置表（30 位姿，模板）

桌面以机械臂基座正前方为原点（单位 cm，x 左右、y 前后），旋转角为杯把朝向（°）。下表为建议初值，**首次测试前请在桌面上实测修正并拍照留档**：

| 编号 | x (cm) | y (cm) | 旋转角 | 用途 | 现场实测 | 照片 |
|------|--------|--------|--------|------|----------|------|
| C01 | -15 | 10 | 0 | 全员 | | |
| C02 | -10 | 14 | +10 | 全员 | | |
| C03 | -5 | 10 | -10 | 全员 | | |
| C04 | 0 | 15 | 0 | 全员 | | |
| C05 | +5 | 10 | +10 | 全员 | | |
| C06 | +10 | 14 | -10 | 全员 | | |
| C07 | +15 | 10 | 0 | 全员 | | |
| C08 | -12 | 6 | -10 | 全员 | | |
| C09 | +12 | 6 | +10 | 全员 | | |
| C10 | 0 | 8 | 0 | 全员 | | |
| C11 | -15 | 14 | +10 | 核心补充 | | |
| C12 | -8 | 16 | -10 | 核心补充 | | |
| C13 | 0 | 18 | +10 | 核心补充 | | |
| C14 | +8 | 16 | -10 | 核心补充 | | |
| C15 | +15 | 14 | +10 | 核心补充 | | |
| C16 | -18 | 10 | -10 | 核心补充 | | |
| C17 | +18 | 10 | +10 | 核心补充 | | |
| C18 | -6 | 12 | 0 | 核心补充 | | |
| C19 | +6 | 12 | 0 | 核心补充 | | |
| C20 | 0 | 12 | +10 | 核心补充 | | |
| C21 | -14 | 8 | +10 | 核心补充 | | |
| C22 | +14 | 8 | -10 | 核心补充 | | |
| C23 | -10 | 18 | 0 | 核心补充 | | |
| C24 | +10 | 18 | 0 | 核心补充 | | |
| C25 | -4 | 15 | -10 | 核心补充 | | |
| C26 | +4 | 15 | +10 | 核心补充 | | |
| C27 | -16 | 12 | 0 | 核心补充 | | |
| C28 | +16 | 12 | 0 | 核心补充 | | |
| C29 | -2 | 10 | -10 | 核心补充 | | |
| C30 | +2 | 10 | +10 | 核心补充 | | |

使用规则：

- **核心模型 E1/E2/E9**：C01–C30 全部 30 个位姿，各测一次
- **其余 6 个模型**：C01–C10 共 10 个位姿，各测一次
- 若某位姿超出机械臂可达空间，在实测阶段平移回可达区域并更新表

### 3.5 判定记录表（模板）

每模型一张：

| 段次 | 杯位 | 抓取 | 移动 | 放下 | 整体 | 录像文件 | 备注 |
|------|------|------|------|------|------|----------|------|
| 1 | C01 | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | | |
| … | | | | | | | |

成功率 = 成功段数 / 总段数（分阶段统计 + 整体统计）。最终汇总表：

| 实验 | 抓取成功率 | 移动成功率 | 放下成功率 | 整体成功率 | 测试段数 |
|------|-----------|-----------|-----------|-----------|----------|
| E1 | | | | | 30 |
| E2 | | | | | 30 |
| E3 | | | | | 10 |
| E4 | | | | | 10 |
| E5 | | | | | 10 |
| E6 | | | | | 10 |
| E7 | | | | | 10 |
| E8 | | | | | 10 |
| E9 | | | | | 30 |

---

## 4. 新测试数据采集流程（未来需要全新测试集时）

现有 90ep 已全部用于训练；若后续要采集**不参与训练**的全新标注测试集：

1. **采集**：按训练时同样的流程用 `lerobot-record` 示教采集（每条 25s 左右），存为 `数据集/<新数据集名>/`
2. **检测标注**：CVAT 标注 cup 框，导出 XML 到 `<新数据集名>/annotations/{top,gripper}/episode_XXX.xml`（命名与目录结构同 formal1_B）
3. **Mask 标注**：SAM2 逐帧生成 NPZ，存到 `<新数据集名>/annotations/masks/{top,gripper}/episode_XXX.npz`
4. **离线评估**：`offline_eval_act_det.py --checkpoint <模型> --dataset.repo_id <新数据集名> --episodes 0-<N-1>`——此时离线指标即为真正的泛化指标
5. **真机测试**：如新数据集的采集环境与测试环境一致，`control_act_det.py` 可直接用其统计量（`--dataset.repo_id` 指向新数据集）

---

## 5. 常见问题

| 现象 | 说明 |
|------|------|
| 离线评估 E1/E4 报错"无法推断数据集" | 纯 act 模型无标注目录，显式传 `--dataset.repo_id` / `--dataset.root` |
| 离线结果没有 det/mask 指标 | 该模型没有对应分支（对照 §1 表格），属预期 |
| 真机测试 `master_gripper.pos` 恒为 0 | 无 leader 臂的预期行为，训练时 policy 模式录制同样如此 |
| 控制循环低于 30Hz 警告 | 相机帧率不足或 GPU 推理慢；先调低相机 fps，或确认 GPU 空闲 |
| 首条动作幅度大 | 每段开始把从动臂摆到演示起始位姿附近 |
| 离线指标有轻微波动 | train 模式 dropout+增强的随机性，可跑两次取均值 |
| 需要边测边录机器人数据 | 用 `lerobot-record --policy.path <checkpoint> --resume`（会追加进数据集，注意别污染训练集） |
