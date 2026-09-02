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

## 一、SAM 2 Mask 生成

> GPU 推理，每 episode ~2 分钟，360 个 mask 总计 ~12 小时。
> `--resume` 保证中断后重跑不重复。

### 创建 screen 并运行

```bash
screen -S mask_gen

screen -r mask_gen  # 重新进入

screen -X -S mask_gen quit
```

进入 screen 后执行：

```bash
cd /root/autodl-tmp/lerobot/lerobot-main

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal/kind1" \
    --cameras top gripper \
    --resume \
    --gaussian_sigma 2.0 \
    --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal/kind2" \
    --cameras top gripper \
    --resume \
    --gaussian_sigma 2.0 \
    --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1/kind1" \
    --cameras top gripper \
    --resume \
    --gaussian_sigma 2.0 \
    --temp_dir /dev/shm

python scripts/generate_sam2_masks.py \
    --dataset_root "/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1/kind2" \
    --cameras top gripper \
    --resume \
    --gaussian_sigma 2.0 \
    --temp_dir /dev/shm
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
> 建议先跑 E1-E6（基线），分析结果后再跑 E7-E9（注入消融）。

### 数据集（已合并）

| 实验用名 | 路径 | state | episodes | frames |
|----------|------|-------|----------|--------|
| **dataset_A** | `数据集/formal_A` | 6-dim（无力感）| 90 | 62,239 |
| **dataset_B** | `数据集/formal1_B` | 9-dim（有力感）| 90 | 64,549 |

> 原始 kind1/kind2 未被修改，仍可在 `数据集/formal/` 和 `数据集/formal1/` 下找到。

### E1 — 标准 ACT 基线 (数据集 A，V1)

```bash
screen -S E1_act_A
screen -r E1_act_A
cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.push_to_hub=false \
    --policy.type=act \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal_A \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --wandb.mode=offline \
    --wandb.notes=E1_act_baseline_A
```

### E2 — 论文版检测 (数据集 A，V2)

```bash
screen -S E2_det_A
screen -r E2_det_A
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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E2_det_A
```

### E3 — 检测+Mask (数据集 A，V3)

```bash
screen -S E3_mask_A
screen -r

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E3_det_mask_A
```

### E3b — 检测+Mask (数据集 A，V3b，gripper mask 消融)

> 与 E3 唯一区别：mask 引导同时作用于 top + gripper 相机，验证 gripper 近景 mask 的增益。

```bash
screen -S E3b_mask_2cam_A
screen -r E3b_mask_2cam_A

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E3b_det_mask_2cam_A
```

### E4 — 标准 ACT 基线 (数据集 B，V1)

```bash
screen -S E4_act_B
screen -r E4_act_B

cd /root/autodl-tmp/lerobot/lerobot-main

lerobot-train \
    --policy.push_to_hub=false \
    --policy.type=act \
    --dataset.repo_id=/root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.video_backend=pyav \
    --steps 120000 \
    --batch_size 8 \
    --wandb.mode=offline \
    --wandb.notes=E4_act_baseline_B
```

### E5 — 论文版检测 (数据集 B，V2)

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E5_det_B
```

### E6 — 检测+Mask (数据集 B，V3)

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E6_det_mask_B
```

### E6b — 检测+Mask (数据集 B，V3b，gripper mask 消融)

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E6b_det_mask_2cam_B
```

### E7 — +FCOS注入 (数据集 B，V4)

```bash
screen -S E7_fcos_inj_B
screen -r
screen -X -S E7_fcos_inj_B quit
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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E7_fcos_inject_B
```

### E7b — 检测+Mask+FCOS注入 (数据集 B，V4b)

> E7 没有 mask，E8 没有 FCOS 注入。此实验填补空白：同时有 mask 引导和 FCOS 注入，便于对照两种注入的独立/叠加效果。

```bash
screen -S E7b_fcos_inj_mask_B
screen -r
screen -X -S <名字> quit
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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E7b_fcos_inj_mask_B
```

### E8 — +Mask注入 (数据集 B，V5)
screen -r
```bash
screen -S E8_mask_inj_B

screen -X -S E8_mask_inj_B quit
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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E8_mask_inject_B
```

### E9 — 全注入 (数据集 B，V6)

```bash
screen -S E9_full_inj_B
screen -X -S E9_full_inj_B quit
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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E9_full_inject_B
```

---

## 三、数据增强消融

> **训练方案：论文版检测（有检测分支，无 mask 分支）**，与 E5 完全相同。
> 当前所有实验默认 `aug_enable=true`（颜色抖动 + 高斯噪声 + 随机遮挡，每种 90% 概率）。
> 唯一变量是增强配置，对照基准为 E5（默认全增强）。

### E_A0 — 无数据增强

> 方案：检测分支 ✅ | mask分支 ❌
> 验证增强是否对透明杯子任务有帮助。

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A0_noaug_B
```

### E_A1 — 仅遮挡增强

> 方案：检测分支 ✅ | mask分支 ❌
> 关颜色抖动+噪声，只保留随机遮挡。验证空间遮挡的独立贡献。

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A1_occ_only_B
```

### E_A2 — 无遮挡增强

> 方案：检测分支 ✅ | mask分支 ❌
> 保留颜色抖动+噪声，关随机遮挡。验证遮挡是否不可或缺。

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
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    --wandb.notes=E_A2_no_occ_B
```

### 启动顺序

```bash
# 1. 先跑基线 E1-E3b (A) → E4-E6 (B)，GPU 可并行 2 组
# 2. E1-E6 结果出来后，按需跑 E7-E9

# 查看当前所有 screen
screen -ls

# 进入指定 screen 看进度
screen -r E1_act_A

# 查看 GPU
nvidia-smi
```

### 实验总览

| ID | screen 名 | 数据集 | 版本 | 预计耗时 |
|----|-----------|--------|------|----------|
| E1 | `E1_act_A` | A (6-dim) | V1 标准ACT | ~8h |
| E2 | `E2_det_A` | A (6-dim) | V2 论文版 | ~8h |
| E3 | `E3_mask_A` | A (6-dim) | V3 检测+Mask(top) | ~8h |
| E3b | `E3b_mask_2cam_A` | A (6-dim) | V3b 检测+Mask(top+gripper) | ~8h |
| E4 | `E4_act_B` | B (9-dim) | V1 标准ACT | ~8h |
| E5 | `E5_det_B` | B (9-dim) | V2 论文版 | ~8h |
| E6 | `E6_mask_B` | B (9-dim) | V3 检测+Mask(top) | ~8h |
| E6b | `E6b_mask_2cam_B` | B (9-dim) | V3b 检测+Mask(top+gripper) | ~8h |
| E7 | `E7_fcos_inj_B` | B (9-dim) | V4 +FCOS注入(无mask) | ~8h |
| E7b | `E7b_fcos_inj_mask_B` | B (9-dim) | V4b +FCOS注入(有mask) | ~8h |
| E8 | `E8_mask_inj_B` | B (9-dim) | V5 +Mask注入 | ~8h |
| E9 | `E9_full_inj_B` | B (9-dim) | V6 全注入 | ~8h |
| E_A0 | `EA0_noaug_B` | B (9-dim) | 检测 无增强 | ~8h |
| E_A1 | `EA1_occ_only_B` | B (9-dim) | 检测 仅遮挡 | ~8h |
| E_A2 | `EA2_no_occ_B` | B (9-dim) | 检测 无遮挡 | ~8h |
