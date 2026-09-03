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
| 🔴 **P0 立即重训** | 论文核心结论直接依赖 | E2、E5、E9 |
| 🟡 **P1 核心对比** | 完整对比矩阵必需 | E3、E6、E7、E8 |
| ⚪ **P2 消融（按需）** | 辅助结论 | E3b、E6b、E7b、E_A0、E_A1、E_A2 |
| ✅ **无需重训** | 纯 ACT，无检测分支，不受 frame_index bug 影响 | E1、E4 |

> **结论**：15 个模型里，**13 个 act_det 模型都需要重训**（检测分支之前没被训练），只有 E1、E4 两个纯 ACT 不用。

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

### E4 — 标准 ACT 基线 (数据集 B，V1) ✅ 无需重训（frame_index 不影响；夹爪坍缩需带权重重训）

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

### 🔴 E5 — 论文版检测 (数据集 B，V2) — P0 必须重训（E7/E8/E9 的对照基准）

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

### 🟡 E6 — 检测+Mask (数据集 B，V3) — P1 必须重训

```bash
screen -S E6_mask_B
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

## 四、启动顺序

```bash
# 1. 先跑 P0（E2/E5/E9），检测修好后最核心的结论
# 2. 再跑 P1（E3/E6/E7/E8）补全对比矩阵
# 3. 最后 P2 消融（E3b/E6b/E7b/E_A0-A2），按需

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
| ✅ 无需 | E4 | `E4_act_B` | B (9-dim) | V1 标准ACT | 否 |
| 🔴 P0 | E2 | `E2_det_A` | A (6-dim) | V2 论文版 | **是** |
| 🔴 P0 | E5 | `E5_det_B` | B (9-dim) | V2 论文版 | **是** |
| 🔴 P0 | E9 | `E9_full_inj_B` | B (9-dim) | V6 全注入 | **是** |
| 🟡 P1 | E3 | `E3_mask_A` | A (6-dim) | V3 检测+Mask(top) | **是** |
| 🟡 P1 | E6 | `E6_mask_B` | B (9-dim) | V3 检测+Mask(top) | **是** |
| 🟡 P1 | E7 | `E7_fcos_inj_B` | B (9-dim) | V4 +FCOS注入(无mask) | **是** |
| 🟡 P1 | E8 | `E8_mask_inj_B` | B (9-dim) | V5 +Mask注入 | **是** |
| ⚪ P2 | E3b | `E3b_mask_2cam_A` | A (6-dim) | V3b 检测+Mask(top+gripper) | **是** |
| ⚪ P2 | E6b | `E6b_mask_2cam_B` | B (9-dim) | V3b 检测+Mask(top+gripper) | **是** |
| ⚪ P2 | E7b | `E7b_fcos_inj_mask_B` | B (9-dim) | V4b +FCOS注入(有mask) | **是** |
| ⚪ P2 | E_A0 | `EA0_noaug_B` | B (9-dim) | 检测 无增强 | **是** |
| ⚪ P2 | E_A1 | `EA1_occ_only_B` | B (9-dim) | 检测 仅遮挡 | **是** |
| ⚪ P2 | E_A2 | `EA2_no_occ_B` | B (9-dim) | 检测 无遮挡 | **是** |

---

> **一句话总结**：E1/E4 不用重训；其余 13 个 act_det 模型全部需要重训，优先级 **P0=E2/E5/E9 → P1=E3/E6/E7/E8 → P2=其余**。重训前务必 `git push` 同步 `frame_index` 补丁到训练机，并确认检测 loss 开始下降。
