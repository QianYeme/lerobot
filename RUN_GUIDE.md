# M-VF-ACT 完整运行指南

> 所有命令可直接复制到终端执行。使用 screen 后台运行，关终端不断。

---

## screen 速查

```bash
screen -S <名字>     # 创建新 session
Ctrl+A D             # 脱离，任务继续跑
screen -ls           # 列出所有 session
screen -r <名字>     # 重新进入
screen -X -S <名字> quit  # 删除指定 session
exit                 # 在 screen 内部退出（等同于删除）
```

---

## 〇、重训前必读（重要）

> 之前所有 `act_det` 模型的**检测分支都从未被训练**：前置处理器把 `frame_index` 丢掉了，
> 导致 `_build_detection_targets` 永远返回空目标，FCOS 检测头只见过"背景"，`det_reg` 停在 ~40。
> 已在 `src/lerobot/processor/converters.py` 补上保留 `frame_index`，并修正 `det_cameras` 的 `wrist→gripper` 命名。

**重训前必须把改动同步到训练机（AutoDL）**，否则重训无效：

```bash
# 本地提交
git add src/lerobot/processor/converters.py \
        src/lerobot/policies/act_det/configuration_act_det.py \
        src/lerobot/policies/act_det/label_loader.py \
        src/lerobot/policies/act_det/modeling_act_det.py \
        src/lerobot/policies/act/configuration_act.py \
        src/lerobot/policies/act/modeling_act.py
git commit -m "fix: preserve frame_index; det_cameras wrist->gripper; add gripper_loss_weight"
git push

# 训练机（AutoDL）上
cd /root/autodl-tmp/lerobot/lerobot-main
git pull
```

**重训后验证检测分支真的在学**：看训练日志里 `det_cls_loss` / `det_reg_loss` 是否下降，
离线评测 `det_reg` 是否不再停在 ~40。

> ⚠️ **数据层面遗留问题（根因 B，重训无法自动解决）**：数据集只有一个 task 标签，
> kind1(固定 50) 与 kind2(随机 40) 对模型没有区分信号，行为克隆会坍缩到固定方向。
> 重训修好检测后，ACTDet 才有能力"看到杯子位置"做随机抓取；纯 ACT(E1/E4) 无法解决，
> 除非把 kind1/kind2 拆成两个 task 标签。

> ⚠️ **根因 C：夹爪坍缩到均值（"只张爪子不抓/不放"）**：夹爪"张开→闭合→放下"是低占空比
> 短事件（一个 episode 只占 ~20%），L1 损失会把它平均成均值（~15），模型夹爪就恒在 ~12~15
> 不动、从不张/闭。已新增 `gripper_loss_weight` 字段（默认 1.0，加在 `ACTConfig`，act 与
> act_det 都继承），下方 15 条训练命令已统一加 `--policy.gripper_loss_weight=3.0`。
> **E1/E4 同样受夹爪坍缩影响**：frame_index 不影响它们，但若真机上也"只张爪子不抓"，
> E1/E4 也需要带该权重重训。

---

## 一、SAM 2 Mask 生成

> GPU 推理，每 episode ~2 分钟，360 个 mask 总计 ~12 小时。
> `--resume` 保证中断后重跑不重复。

```bash
screen -S mask_gen
cd /root/autodl-tmp/lerobot/lerobot-main

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal/kind1" \
    --cameras top gripper --resume --gaussian_sigma 2.0 --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal/kind2" \
    --cameras top gripper --resume --gaussian_sigma 2.0 --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1/kind1" \
    --cameras top gripper --resume --gaussian_sigma 2.0 --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1/kind2" \
    --cameras top gripper --resume --gaussian_sigma 2.0 --temp_dir /dev/shm
```

跑完 `Ctrl+A D` 脱离。

### 验证

```bash
for ds in formal/kind1 formal/kind2 formal1/kind1 formal1/kind2; do
    top=$(ls "/root/autodl-tmp/lerobot/lerobot-main/数据集/$ds/annotations/masks/top/"*.npz 2>/dev/null | wc -l)
    grip=$(ls "/root/autodl-tmp/lerobot/lerobot-main/数据集/$ds/annotations/masks/gripper/"*.npz 2>/dev/null | wc -l)
    echo "  $ds: top=$top, gripper=$grip"
done
```

期望：formal/kind1 top=50 gripper=50, formal/kind2 top=40 gripper=40, formal1/kind1 top=50 gripper=50, formal1/kind2 top=40 gripper=40。

---

## 二、训练实验

> GPU: RTX 5090 32GB，可并行 2 组训练。每组约 8h，9 组串行 ~72h，并行 ~40h。

### 数据集（已合并）

| 实验用名 | 路径 | state | episodes | frames |
|----------|------|-------|----------|--------|
| **dataset_A** | `数据集/formal_A` | 6-dim（无力感）| 90 | 62,239 |
| **dataset_B** | `数据集/formal1_B` | 9-dim（有力感）| 90 | 64,549 |

> 原始 kind1/kind2 未被修改，仍可在 `数据集/formal/` 和 `数据集/formal1/` 下找到。

### 优先级图例

| 标记 | 含义 | 模型 |
|------|------|------|
| 🔵 **第一步验证** | 先跑这 3 个确认改动有效，通过后再继续 | E4、E5、E6 |
| 🔴 **P0 立即重训** | 论文核心结论直接依赖 | E2、E9 |
| 🟡 **P1 核心对比** | 完整对比矩阵必需 | E3、E7、E8 |
| ⚪ **P2 消融（按需）** | 辅助结论 | E3b、E6b、E7b、E_A0、E_A1、E_A2、E_R0 |
| ✅ **无需重训** | 纯 ACT，无检测分支，不受 frame_index bug 影响 | E1 |

> **结论**：15 个模型里，**13 个 act_det 模型都需要重训**（检测分支之前没被训练）；纯 ACT 中只有 E1 不用重训，E4 需带夹爪权重重训（第一步验证之一）。

### 第一步验证三件套（先跑这 3 个）

> 目的：先验证本次改动是否有效，再决定是否继续全量重训（省 GPU 时间）。
> 三个模型都在数据集 B（9-dim，与真机一致）上，一个改动对应一个模型：

| 模型 | 角色 | 验证的改动 | 通过判据 |
|------|------|-----------|---------|
| **E4** | 基准（纯 ACT） | 根因 C：`gripper_loss_weight=3.0` | 真机夹爪诊断出现 张→闭 / 闭→张 转换，目标不再恒 ~11 |
| **E5** | 检测头 | 根因 A：frame_index 保留 + label 查找修正 | 训练日志 `det_cls_loss`/`det_reg_loss` 明显下降（旧模型 det_reg≈40） |
| **E6** | Mask 头 | 根因 A + mask 路径（`_load_mask_batch` 恢复工作） | `mask_loss` 非零且下降（旧模型恒为 0/None） |

**流程**：E5 det 损失下降 + E6 mask_loss 非零下降 → 根因 A 修复有效；E4 真机夹爪出现张闭转换 → 根因 C 修复有效。通过后再跑 P0 其余（E2/E9）→ P1 → P2。

---

### E1 — 标准 ACT 基线 (数据集 A，V1) ✅ 无需重训（frame_index 不影响；夹爪坍缩需带权重重训）

```bash
screen -S E1_act_A
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.push_to_hub=false \
    --policy.type=act \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --wandb.mode=offline \
    --wandb.notes=E1_act_baseline_A
```

### 🔴 E2 — 论文版检测 (数据集 A，V2) — P0 必须重训

```bash
screen -S E2_det_A
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E2_det_A
```

### 🟡 E3 — 检测+Mask (数据集 A，V3) — P1 必须重训

```bash
screen -S E3_mask_A
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E3_det_mask_A
```

### ⚪ E3b — 检测+Mask (数据集 A，V3b，gripper mask 消融) — P2 按需

> 与 E3 唯一区别：mask 引导同时作用于 top + gripper 相机。

```bash
screen -S E3b_mask_2cam_A
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.mask_cameras '{"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":true}}' \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E3b_det_mask_2cam_A
```

### 🔵 E4 — 标准 ACT 基线 (数据集 B，V1) — 第一步验证（夹爪权重重训）

```bash
screen -S E4_act_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.push_to_hub=false \
    --policy.type=act \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --wandb.mode=offline \
    --wandb.notes=E4_act_baseline_B
```

### 🔵 E5 — 论文版检测 (数据集 B，V2) — 第一步验证 + P0（E7/E8/E9 的对照基准）

```bash
screen -S E5_det_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E5_det_B
```

### 🔵 E6 — 检测+Mask (数据集 B，V3) — 第一步验证 + P1

```bash
screen -S E6_mask_B
screen -r E6_mask_B
screen -X -S <名字> quit

cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E6_det_mask_B
```

### ⚪ E6b — 检测+Mask (数据集 B，V3b，gripper mask 消融) — P2 按需

> 与 E6 唯一区别：mask 引导同时作用于 top + gripper。

```bash
screen -S E6b_mask_2cam_B

cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.mask_cameras '{"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":true}}' \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E6b_det_mask_2cam_B
```

### 🟡 E7 — +FCOS注入 (数据集 B，V4) — P1 必须重训

```bash
screen -S E7_fcos_inj_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=true \
    --policy.fcos_inject_levels '["p4"]' \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E7_fcos_inject_B
```

### ⚪ E7b — 检测+Mask+FCOS注入 (数据集 B，V4b) — P2 按需

> E7 没有 mask，E8 没有 FCOS 注入。此实验填补空白：同时有 mask 引导和 FCOS 注入。

```bash
screen -S E7b_fcos_inj_mask_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.fcos_feature_inject=true \
    --policy.fcos_inject_levels '["p4"]' \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E7b_fcos_inj_mask_B
```

### 🟡 E8 — +Mask注入 (数据集 B，V5) — P1 必须重训

```bash
screen -S E8_mask_inj_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.mask_feature_inject=true \
    --policy.mask_inject_pool_size=[15,20] \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E8_mask_inject_B
```

### 🔴 E9 — 全注入 (数据集 B，V6) — P0 必须重训（论文旗舰）

```bash
screen -S E9_full_inj_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.fcos_feature_inject=true \
    --policy.fcos_inject_levels '["p4"]' \
    --policy.mask_feature_inject=true \
    --policy.mask_inject_pool_size=[15,20] \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E9_full_inject_B
```

---

## 三、数据增强消融（⚪ P2 按需重训）

> **训练方案：论文版检测（有检测分支，无 mask 分支）**，与 E5 完全相同。
> 唯一变量是增强配置，对照基准为 E5（默认全增强）。
> 这三个模型的检测分支同样受 frame_index bug 影响，需要重训。

### ⚪ E_A0 — 无数据增强

```bash
screen -S EA0_noaug_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.aug_enable=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A0_noaug_B
```

### ⚪ E_A1 — 仅遮挡增强

> 关颜色抖动+噪声，只保留随机遮挡。

```bash
screen -S EA1_occ_only_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.aug_color_jitter_enable=false \
    --policy.aug_noise_enable=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A1_occ_only_B
```

### ⚪ E_A2 — 无遮挡增强

> 保留颜色抖动+噪声，关随机遮挡。

```bash
screen -S EA2_no_occ_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.aug_occlusion_enable=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A2_no_occ_B
```

---

## 四、数据组成消融（⚪ P2 按需）

> **问题**：kind1(固定 50) + kind2(随机 40) 混合训练是否导致"固定方向"坍缩？
> **实验**：E5 同配置，只用 kind2（随机杯位）训练——通过 `--dataset.episodes` 过滤，
> 不需要复制数据集。若该模型真机上不再"固定方向"，说明混合数据确实有害。

> ⚠️ **先确认 kind2 的 episode 区间**：默认假设 kind1=0–49、kind2=50–89（合并顺序决定）。
> 如区间不同，把下面命令里的列表换成实际区间。

### ⚪ E_R0 — 纯随机子集消融（E5 配置，仅 kind2 40 集）

```bash
screen -S ER0_random_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.episodes='[50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89]' \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_R0_random_only_B
```

---

## 五、启动顺序

```bash
# 1. 先跑第一步验证三件套（E4/E5/E6），确认改动有效（判据见上方三件套表）
# 2. 通过后跑 P0 其余（E2/E9）—— 论文最核心结论
# 3. 再跑 P1（E3/E7/E8）补全对比矩阵
# 4. 最后 P2 消融（E3b/E6b/E7b/E_A0-A2、E_R0），按需

# 查看当前所有 screen
screen -ls

# 进入指定 screen 看进度
screen -r E5_det_B

# 查看 GPU
nvidia-smi
```

### 实验总览（按优先级排序）

| 优先级 | ID | screen 名 | 数据集 | 版本 | 是否需重训 |
|--------|----|-----------|--------|------|-----------|
| ✅ 无需 | E1 | `E1_act_A` | A (6-dim) | V1 标准ACT | 否 |
| 🔵 第一步 | E4 | `E4_act_B` | B (9-dim) | V1 标准ACT | 是（夹爪权重） |
| 🔴 P0 | E2 | `E2_det_A` | A (6-dim) | V2 论文版 | **是** |
| 🔵 第一步 | E5 | `E5_det_B` | B (9-dim) | V2 论文版 | **是** |
| 🔴 P0 | E9 | `E9_full_inj_B` | B (9-dim) | V6 全注入 | **是** |
| 🟡 P1 | E3 | `E3_mask_A` | A (6-dim) | V3 检测+Mask(top) | **是** |
| 🔵 第一步 | E6 | `E6_mask_B` | B (9-dim) | V3 检测+Mask(top) | **是** |
| 🟡 P1 | E7 | `E7_fcos_inj_B` | B (9-dim) | V4 +FCOS注入(无mask) | **是** |
| 🟡 P1 | E8 | `E8_mask_inj_B` | B (9-dim) | V5 +Mask注入 | **是** |
| ⚪ P2 | E3b | `E3b_mask_2cam_A` | A (6-dim) | V3b 检测+Mask(top+gripper) | **是** |
| ⚪ P2 | E6b | `E6b_mask_2cam_B` | B (9-dim) | V3b 检测+Mask(top+gripper) | **是** |
| ⚪ P2 | E7b | `E7b_fcos_inj_mask_B` | B (9-dim) | V4b +FCOS注入(有mask) | **是** |
| ⚪ P2 | E_A0 | `EA0_noaug_B` | B (9-dim) | 检测 无增强 | **是** |
| ⚪ P2 | E_A1 | `EA1_occ_only_B` | B (9-dim) | 检测 仅遮挡 | **是** |
| ⚪ P2 | E_A2 | `EA2_no_occ_B` | B (9-dim) | 检测 无遮挡 | **是** |
| ⚪ P2 | E_R0 | `ER0_random_B` | B 仅kind2(40集) | V2 论文版 | **是** |

---

> **一句话总结**：先跑**第一步验证三件套 E4/E5/E6**（基准/检测头/Mask头，各验证一处改动），判据通过后再继续：**P0=E2/E9 → P1=E3/E7/E8 → P2=其余+E_R0**。重训前务必 `git push` 同步 `frame_index` 补丁到训练机。

---

## 六、强化训练（2026-09-05 追加：相机已确认正确，主因是训练强度不足）

> **背景**：真机「固定姿态、不追杯子」排查后，相机序号已目视确认正确（gripper=2、top=4，
> 早先「index 4 绿色」是相机 `connect()` 后暖机瞬态误判）。离线评测：动作 `l1=0.508`
> （仅优于常量均值基线 0.763 约 34% = **半坍缩**）、检测 `det_cls=1.92`（仍不自信）。
> A/C/D/E 四个结构性 bug 均已修，现在的问题是**训练还没训够**，不是 bug、不是相机。

### 决策：resume 续训（不要从零重训）

1. **ACT/ACTDet 无学习率调度器**（`ACTConfig.get_scheduler_preset()` 返回 `None`，
   LR 恒 `1e-5`，`validate()` 里 scheduler 为 `None`）→ **不存在余弦退火到 0 导致 resume
   学不动的问题**，继续训练 = 干净续训。
2. 现有 E6 是 09-04 重训、A/C/D/E 修复**已全部 baked-in**（`det_reg=1.48` 证明 E 生效），
   不是需要重来的脏模型。
3. 省钱省时：120000 步（5090 上约 8h）不浪费。

> 唯一需要从零重训：resume 后前几千步 `l1`/`det_cls` 不再下降（= 已收敛），再走「超参强化」。

### E6 续训（检测+Mask，数据集 B）——主推

```bash
# 0) 在训练机（AutoDL）找到 E6 上次输出目录（选 use_mask_guidance=true 的那个 = E6）
ls /root/autodl-tmp/lerobot/lerobot-main/outputs/train/

# 1) 续训：120000 → 240000 步
screen -S E6_resume_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --config_path=/root/autodl-tmp/lerobot/lerobot-main/outputs/train/<日期>/<时间>_act_det/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --steps=240000 \
    --wandb.mode=offline \
    --wandb.notes=E6_mask_B_resume_240k
```

> ⚠️ `<日期>/<时间>_act_det` 换成 E6 实际目录名（`ls` 确认，选 `use_mask_guidance=true`）。
> ⚠️ `--config_path=` 必须用 `=`（脚本只匹配 `--config_path=` 前缀）。
> ⚠️ 续训前确认 AutoDL 代码仍是训 E6 时的版本（`fcos.py` stride 归一化 + `det_weight=1.0`），
>   可 `git -C /root/autodl-tmp/lerobot/lerobot-main status` 核对。
> ⚠️ 续训的新 checkpoint 存到**同一输出目录**（checkpoints/140000、…、240000），原 120000 不删，
>   `last` 链接自动指向最新。

### 续训后验证

- 前 5000 步盯日志：`l1_loss` 是否继续降（0.508 → 目标 <0.4）、`det_cls_loss` 是否继续降
  （1.92 → 目标 <1.0）、`det_reg_loss` 保持个位数。
- 两者持续降 → 训到 240000，甚至 300000。
- 前 5000 步基本不降 → 已收敛，停掉，改走「超参强化」路线（见下）。

### 备选：从零重训（仅当 resume 证明已收敛时用）

> 超参已 baked 进 config.json，改超参必须从零（resume 不会干净生效）。

```bash
screen -S E6_strong_B
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.annotation_dir=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B/annotations \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 240000 \
    --batch_size 8 \
    --policy.gripper_loss_weight=3.0 \
    --policy.focal_gamma=1.0 \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E6_strong_B_240k
```

> `focal_gamma` 是 `ACTDetConfig` 已定义字段（默认 2.0）。检测头对「小目标+少正样本」迟迟
> 不自信时，gamma 2→1 降低对已分对样本的抑制、让头更敢学。这是备选微调，非主推。
