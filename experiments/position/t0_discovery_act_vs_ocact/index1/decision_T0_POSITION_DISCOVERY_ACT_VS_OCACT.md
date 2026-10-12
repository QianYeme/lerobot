# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T0 — Position Discovery Training: ACT vs OC-ACT

## 1. Identity

- **Experiment ID**: `T0_POSITION_DISCOVERY_ACT_VS_OCACT`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_TRAINING`
- **Mode**: `DISCOVERY`
- **Upstream Requirements**:
  - `D1_FORMAL4_TRAINING_READINESS = PASS`
  - `M0_OC_ACT_ARCHITECTURE_BUILD = PASS`
  - `P0_1_CUP_SPATIAL_POOLING_FIX = PASS`
- **Real Robot**: 禁止
- **Final Test**: 禁止
- **Seeds**: `1` for this Discovery round

本轮目标：

> 在完全一致的数据、训练预算和 ACT 主干设置下，对比标准 ACT 与第一版 OC-ACT，判断 object-conditioned representation 是否值得进入后续多 seed Confirmation。

---

## 2. Frozen Dataset

正式训练数据使用：

```text
数据集/formal4/kind_merged_nomaster
```

正式 split 使用 D1 冻结的：

```text
split_manifest_v1.json
```

固定：

```text
Train = 36 episodes
Dev   = 12 episodes
Final Test = 12 episodes
```

本轮：

```text
Train: allowed
Dev: allowed for validation/model selection
Final Test: FORBIDDEN
```

---

## 3. Normalization

必须使用 D1 已冻结的：

```text
train_normalization_v1.json
```

禁止重新用：

```text
Train + Dev
或
全部60条
```

计算 normalization。

---

## 4. Compared Models

只比较两个模型。

### B0 — ACT Baseline

```text
Top RGB
Wrist RGB
8D NOMASTER state
↓
Original ACT
↓
6D action chunk
```

不得加入：

- DINO；
- object token；
- prototype；
-额外 detector；
- force module。

---

### P1 — OC-ACT

固定使用 M0 + P0.1 已冻结结构：

```text
Top RGB
├─ Original ACT Top branch
└─ Frozen DINOv2-S
      ↓
   P0 frozen prototype bank
      ↓
max cosine similarity
      ↓
temperature-scaled soft pooling
      tau = 0.20
      ↓
1 Global + 4 Local Cup Tokens
      ↓
semantic + x/y + confidence
      ↓
Adapter
      ↓
Pre-Encoder Cross-Attention
      ↓
Gated Residual
      ↓
Original ACT Encoder/Decoder
```

Wrist RGB：

```text
Original ACT Wrist branch
```

Robot state：

```text
8D NOMASTER
```

---

## 5. Frozen Object Artifacts

OC-ACT 必须使用：

```text
cup_prototype_bank_v1.pt
object_pooling_config_v1.json
```

其中：

```text
prototype_count = 4
temperature = 0.20
DINO = dinov2_vits14
DINO frozen = true
object tokens = 5
```

本轮禁止重新：

- 构建 prototype；
- 更换 reference；
- 调 temperature；
- 修改 token 数；
- fine-tune DINO。

---

## 6. Fair-Comparison Rule

B0 和 P1 必须尽可能保持一致：

- 相同 Train split；
- 相同 Dev split；
- 相同 normalization；
- 相同 image preprocessing；
- 相同 action chunk length；
- 相同 optimizer family；
- 相同 learning-rate schedule；
- 相同 batch size；
- 相同训练步数；
- 相同 seed；
- 相同 checkpoint cadence；
- 相同 evaluation cadence；
- 相同 ACT hidden dimensions；
- 相同 original ACT backbone initialization policy。

唯一主要科研差异：

> OC-ACT 增加冻结 DINO object-conditioning path。

---

## 7. Training Budget

本轮建议沿用项目当前成熟 ACT 配置的正式训练预算：

```text
100k optimizer steps
```

如果现有项目配置的标准正式 ACT 预算不是 100k，Codex 必须：

1. 读取当前正式 ACT config；
2. 在训练前报告真实值；
3. B0 / P1 使用完全一致预算；
4. 不得根据中途结果改变任一模型预算。

本轮禁止：

```text
B0 100k
P1 150k
```

之类不公平比较。

---

## 8. Seed

Discovery 本轮固定：

```text
1 seed
```

seed 值必须在启动前写入：

```text
training_manifest.json
```

B0/P1 使用相同 seed。

本轮不是最终 paper confirmation。

如果结果支持 hypothesis，下一阶段才进入：

```text
>= 3 seeds Confirmation
```

---

## 9. Checkpoint Policy

训练时定期保存 checkpoint。

最终 Discovery checkpoint 的选择规则必须在看 Dev 结果前冻结。

推荐：

> 使用 Dev primary metric 最优 checkpoint；若项目现有 ACT 标准流程固定使用 final checkpoint，则两模型统一使用 final checkpoint。

Codex 必须在训练启动前记录：

```text
checkpoint_selection_rule
```

不得 B0 和 P1 使用不同规则。

---

## 10. Training Health Gates

必须记录：

- train loss；
- dev loss；
- gradient norm；
- NaN/Inf；
- learning rate；
- GPU memory；
- throughput；
- wall-clock；
- gate alpha trajectory（P1）；
- object branch gradient norms（P1）。

### T0-G1 — Numerical Health

两模型均：

```text
no NaN
no Inf
no catastrophic divergence
```

### T0-G2 — Baseline Health

ACT baseline 必须达到与当前项目历史合理范围一致的训练行为。

若 baseline 自身异常：

> 本轮 BLOCKED，不允许用 P1 对比得出科研结论。

---

## 11. OC-ACT Mechanism Logging

P1 必须额外记录：

### Gate

```text
alpha over training
```

### Gradients

至少：

- Adapter grad norm；
- Cross-Attention grad norm；
- gate grad；
- DINO grad = none/zero。

### Object Response

在固定 Train/Dev diagnostic batch 上保存：

```text
cross-attention response norm
object residual norm
```

用于判断 object branch 是否完全死亡。

注意：

> alpha 非零不等于机制成功。

---

## 12. Primary Discovery Question

本轮核心不是：

> 谁的 teacher-forced loss 最低。

而是：

> OC-ACT 是否显示出值得进一步验证的目标位置敏感性和行为改善信号。

因此评估分成：

```text
Training Health
Mechanism
Offline Behavior
```

---

## 13. Offline Evaluation — Common Metrics

B0/P1 都至少报告：

- Dev action loss；
- per-action-dimension error；
- trajectory/action-chunk error；
- h0 / early-step action error；
- gripper action error；
- arm joint action error。

但这些只属于 teacher-forced offline metrics。

---

## 14. Position-Specific Offline Evaluation

在 Dev 中设计不改变模型的诊断分析：

### P2-A — Target Position Groups

根据 Top image 中 cup position，把 Dev 样本按真实/可信位置分组：

```text
left / center / right
near / center / far
```

分组只能用于分析，不修改 split。

### P2-B — Conditional Action Response

在相似 robot state 下比较不同 cup position：

> 模型早期 reach action 是否随目标位置变化。

重点查看与横向/纵向 Reach 最相关的 arm action dimensions。

### P2-C — Direction Consistency

例如杯子从左→右变化时：

> 对应关键 action dimension 是否呈合理单调/方向变化。

不要只看绝对误差。

---

## 15. Counterfactual Mechanism Tests

P1 必须在固定 Dev diagnostic subset 上做：

### Normal

正确 object representation。

### Zero Object

```text
object tokens = 0
```

### Shuffle Object

将 object tokens 在 batch / frames 间打乱。

### Spatial Swap

至少做：

```text
x → -x
```

或等价左右空间翻转诊断。

目的：

> 判断 policy 是否真的使用 object representation。

这些只用于机制诊断，不用于重新训练。

---

## 16. Baseline Comparison Logic

理想证据模式：

```text
B0:
target position changes
→ action response weak / averaged

P1:
target position changes
→ action response stronger and directionally correct
```

同时 P1：

```text
zero/shuffle/spatial swap
→ response measurably degrades or changes
```

这才支持“object conditioning 被 policy 使用”。

---

## 17. Discovery PASS Criteria

本轮不设置“必须胜多少百分比”的伪精确阈值。

### PASS

要求同时满足：

1. B0/P1 training health 均正常；
2. P1 object branch 非死亡；
3. P1 在 Position-specific diagnostic 中比 B0 显示更清晰、方向正确的目标位置响应；
4. P1 的 zero/shuffle/spatial-swap 至少一种产生符合机制预期的明显变化；
5. P1 没有明显牺牲整体基本 action quality；
6. 没有 Final Test contamination。

### PARTIAL

例如：

- teacher-forced loss 无明显改善，但位置响应明显增强；
- x 方向有效，y 方向弱；
- object ablation 有效，但总体 offline action error略差。

PARTIAL 进入人工 review，不自动否决。

### FAIL

例如：

- P1 object branch 几乎不被使用；
- P1 与 B0 对目标位置变化同样不敏感；
- spatial swap 几乎不影响输出；
- P1 明显训练不稳定；
- 性能改善只能由更大模型容量解释而没有机制证据。

---

## 18. Final Test Protection

本轮严格禁止：

```text
Final Test inference
```

包括：

- 不跑 loss；
- 不画 prediction；
- 不抽图看结果；
- 不用 Final Test 选择 checkpoint；
- 不用 Final Test 调 threshold；
- 不用 Final Test 调 architecture。

Final Test 保留到 Confirmation。

---

## 19. No Real Robot Yet

本轮禁止真机。

原因：

> Discovery 先确认离线机制是否存在，再决定是否值得支付真机成本。

---

## 20. Resource Logging

因为最终必须部署到：

```text
RTX 4060 Laptop
8 GB VRAM
16 GB RAM
```

训练时记录：

- total params；
- trainable params；
- frozen params；
- checkpoint size；
- peak VRAM；
- inference latency smoke；
- DINO forward cost。

训练服务器资源不是部署结论。

本轮只保存数据，正式 G6 Deployment Equivalence 后再在目标 laptop 测。

---

## 21. Required Outputs

建议目录：

```text
experiments/position/t0_discovery_act_vs_ocact/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── training_manifest.json
    ├── checkpoints.json
    ├── mechanism.json
    ├── position_diagnostics.json
    ├── resources.json
    ├── figures/
    │   ├── train_loss.*
    │   ├── dev_loss.*
    │   ├── gate_alpha.*
    │   ├── position_response.*
    │   ├── normal_vs_zero.*
    │   ├── normal_vs_shuffle.*
    │   └── normal_vs_spatial_swap.*
    └── proposal.md
```

Raw checkpoints：

```text
results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/
```

---

## 22. Codex MAY

- 启动 B0/P1 训练；
- 恢复中断训练；
- 保存 checkpoints；
- 运行 Train/Dev evaluation；
- 运行固定 counterfactual diagnostics；
- 生成图表；
- 做工程 bugfix；
- 创建 proposal.md。

---

## 23. Codex MUST

- B0/P1 训练条件一致；
- 使用 frozen 36/12 split；
- 使用 train-only normalization；
- 使用 frozen prototype + tau=0.20；
- DINO frozen；
- 保存 seed / config / git commit；
- 报告 mechanism diagnostics；
- 不读取 Final Test；
- 完成本轮后停止。

---

## 24. Codex MUST NOT

- 修改本 decision.md；
- 使用 Final Test；
- 启动 real robot；
- 因 B0/P1 中途结果不同而修改训练预算；
- 调 tau；
- 重建 prototype；
- fine-tune DINO；
- 加 Force module；
- 更换 injection point；
- 增加额外 detector；
- 根据 Dev 结果开始 architecture search；
- 自动进入多 seed Confirmation。

---

## 25. Completion Rule

训练与全部 Discovery diagnostics 完成后停止。

由用户 + ChatGPT 判定：

```text
PASS
→ C0 Position Offline Confirmation (>=3 seeds)

PARTIAL
→ targeted diagnostic / one new decision

FAIL
→ reject or redesign object-conditioning mechanism
```

不得自动进入下一阶段。
