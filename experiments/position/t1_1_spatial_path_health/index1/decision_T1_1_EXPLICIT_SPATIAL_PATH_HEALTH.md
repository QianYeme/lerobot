# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T1.1 — Explicit Spatial Path Health Diagnostic

## 1. Identity

- **Experiment ID**: `T1_1_EXPLICIT_SPATIAL_PATH_HEALTH`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_DIAGNOSTIC`
- **Mode**: `DIAGNOSIS`
- **Upstream**: `T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING = FAIL`
- **Retraining**: 禁止
- **Architecture Change**: 禁止
- **Dev Use**: 允许
- **Final Test Use**: 禁止
- **Real Robot**: 禁止

本轮只回答：

> 为什么 P2 的显式 `(x,y,confidence) → SpatialMLP → beta → state token` 路径没有被模型使用？

---

## 2. Frozen Evidence

T1 已确认：

```text
B1 Dev L1 = 0.132089
P2 Dev L1 = 0.145435
P2 worse than B1
```

P2：

```text
beta_init = 0.01
beta_final ≈ 0.00318068
```

Spatial ablation：

```text
max Dev L1 change ≈ 9.47e-08
```

说明：

> 显式 spatial path 对最终输出几乎无可测影响。

同时：

```text
DINO frozen = true
SpatialMLP parameters finite = true
beta finite/nonzero = true
```

因此当前问题不是 NaN / checkpoint 损坏，而是：

> path effectively unused.

---

## 3. Diagnostic Questions

必须回答：

### Q1
SpatialMLP 在训练期间是否获得有效梯度？

### Q2
`beta` 为什么从 0.01 向更小值收缩？

### Q3
Spatial embedding 的数值尺度相对于 state token / visual token / object residual 是否过小？

### Q4
即使强制放大该路径，动作是否对 spatial embedding 敏感？

### Q5
问题主要来自：
- optimization；
- scale；
- fusion location；
- representation；
中的哪一类？

---

## 4. Frozen Checkpoint

固定使用：

```text
P2 / 100000
```

如训练日志中存在中间 checkpoint：

```text
10k, 20k, ..., 100k
```

允许读取全部已有 checkpoint 做诊断。

禁止重新训练。

---

## 5. Gradient History

若训练日志已记录：

- SpatialMLP grad norm；
- beta grad；
- state projection grad；
- object adapter grad；
- cross-attention grad；

必须整理成曲线。

如果日志没有完整记录：

> 不允许通过重新训练补齐。

只能报告 `NOT_RECORDED`，并继续做 checkpoint-based diagnostics。

---

## 6. Parameter Trajectory

对已有 checkpoints 提取：

```text
beta(step)
||SpatialMLP weights||
||state projection weights||
```

至少报告：

```text
10k
20k
...
100k
```

如果中间 checkpoint 缺失则使用现有可得点。

目标：

> 确认 beta 是持续被压低、早期就关闭，还是后期才衰减。

---

## 7. Representation Scale Diagnostic

在固定 Dev diagnostic batch 上测：

### 7.1 State Token Norm

```text
||state_token||
```

### 7.2 Raw Spatial Embedding Norm

```text
||spatial_embedding||
```

### 7.3 Effective Spatial Residual Norm

```text
||beta * spatial_embedding||
```

### 7.4 Object Residual Norm

```text
||alpha * object_response||
```

### 7.5 Visual Token Norm

记录代表性：

```text
||visual_token||
```

必须报告比例：

```text
effective spatial / state
effective spatial / object residual
effective spatial / visual
```

---

## 8. Sensitivity Sweep — No Retraining

固定 P2 checkpoint，不修改权重。

仅推理时临时覆盖：

```text
beta_effective
```

测试：

```text
0
0.00318   # learned
0.01      # init
0.05
0.10
0.50
1.00
```

这只是诊断，不是候选模型。

固定：

```text
Top/Wrist RGB
state
object tokens
```

比较：

```text
Δaction(beta_override)
```

目的：

> 判断 path 是“表达本身无效”，还是“只是 learned beta 太小”。

---

## 9. Coordinate Counterfactual Under Beta Override

对每个 beta override 至少做：

### X flip

```text
x -> -x
```

### Y flip

```text
y -> -y
```

### XY shuffle

跨 sample 打乱：

```text
x,y
```

### Zero

```text
x=y=confidence=0
```

记录：

- all-action delta；
- arm-only delta；
- per-dim delta；
- horizontal primary dim0；
- longitudinal primary dim2。

---

## 10. Fusion-Linearity Check

检查：

```text
state_token' = state_token + beta * spatial_embedding
```

在实际 forward 中是否真的进入 encoder。

必须通过 hook / tensor capture 验证：

```text
state_token' - state_token
```

与：

```text
beta * spatial_embedding
```

数值一致。

排除实现 bug：

- spatial embedding 被后续覆盖；
- wrong token index；
- detach；
- normalization 后被完全消除；
- checkpoint load 漏字段。

---

## 11. LayerNorm / Normalization Check

若融合后紧接 LayerNorm，必须检查：

> LayerNorm 是否几乎消除了 spatial residual。

报告：

```text
pre-LN difference
post-LN difference
```

以及：

```text
cosine(state_before, state_after)
```

如果 post-LN 差异极小：

> 记录为潜在机制原因。

不得本轮修改 LayerNorm。

---

## 12. Input Variance Check

在 Train/Dev 上报告冻结 spatial inputs：

```text
x_global
y_global
confidence
```

的：

- mean；
- std；
- min/max；
- quantiles。

确认不是：

```text
x/y dynamic range too small
confidence nearly constant
```

导致 MLP 没有有用变化。

---

## 13. Action-Relevance Check

对 Dev expert action：

```text
x_global / y_global
vs
action dims
```

重新报告预注册相关性。

目的：

> 确认当前 frozen soft center 本身确实与动作有可学习关系。

不得看结果后换 primary dims。

---

## 14. Diagnostic Classification

本轮最终必须把失败归到至少一个主类别。

### A — SCALE_STARVATION

如果：

```text
||beta * spatial_embedding||
<<
||state_token||
```

且 beta override 放大后 action sensitivity 明显出现。

### B — FUSION_SUPPRESSION

如果 pre-fusion 有明显残差，但 LayerNorm / encoder entry 后几乎消失。

### C — REPRESENTATION_WEAKNESS

如果即使 beta=0.5/1.0，x/y counterfactual 仍几乎不影响 action。

### D — OPTIMIZATION_REJECTION

如果 path 有合理 scale 和 sensitivity，但训练把 beta 稳定压低，说明 action loss 主动拒绝该路径。

### E — IMPLEMENTATION_BUG

如果实际 state token 没有按合同融合。

允许多标签，但必须给出主因排序。

---

## 15. PASS Definition

本轮没有“模型性能 PASS”。

本轮 PASS 指：

> 已经有足够证据定位显式 spatial path 失败的主要原因，并能支撑下一份单变量 architecture decision。

---

## 16. Next-Step Mapping

### If SCALE_STARVATION

下一轮可考虑：

```text
remove / fix tiny beta gate
or
normalize residual scale
```

但只能选一个单变量。

### If FUSION_SUPPRESSION

下一轮可考虑：

```text
fusion after normalization
or
concat+projection
```

单独实验。

### If REPRESENTATION_WEAKNESS

不应继续单纯放大 beta。

应重新设计 spatial encoding，例如：

```text
relative target-state representation
```

### If OPTIMIZATION_REJECTION

需要考虑：

```text
auxiliary objective
or
architectural coupling
```

但不得在本轮实施。

### If IMPLEMENTATION_BUG

先工程修复，再重新执行原 T1，不算新科研 intervention。

---

## 17. Required Outputs

```text
experiments/position/t1_1_spatial_path_health/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── diagnosis.json
    ├── parameter_trajectory.json
    ├── scale_analysis.json
    ├── beta_override_sensitivity.json
    ├── implementation_check.json
    ├── input_variance.json
    ├── figures/
    │   ├── beta_trajectory.*
    │   ├── representation_norms.*
    │   ├── beta_override_action_delta.*
    │   ├── pre_post_norm_effect.*
    │   └── xy_action_correlation.*
    └── proposal.md
```

---

## 18. Codex MAY

- 读取已有 checkpoints；
- 对固定 P2 做 forward hooks；
- 临时 override beta 做推理诊断；
- 做 x/y counterfactual；
- 读取训练日志；
- 生成图表；
- 创建 proposal.md。

---

## 19. Codex MUST

- 不重训；
- 不改模型；
- 不读 Final Test；
- 不修改 prototype/tau；
- 不修改 checkpoint；
- 明确区分工程 bug 与科研失败；
- 给出主诊断类别；
- 完成后停止。

---

## 20. Codex MUST NOT

- 修改本 decision.md；
- 启动新训练；
- 手工永久放大 beta；
- 改 SpatialMLP；
- 改 fusion point；
- 加 auxiliary loss；
- 使用 Final Test；
- 启动 real robot；
- 自动进入下一 architecture intervention。

---

## 21. Completion Rule

本轮结束后：

```text
DIAGNOSIS_READY = TRUE
```

并停止。

由用户 + ChatGPT 根据诊断类别创建下一份单变量实验 decision。
