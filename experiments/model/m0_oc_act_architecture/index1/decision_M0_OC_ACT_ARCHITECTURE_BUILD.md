# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# M0 — OC-ACT Architecture Build

## 1. Identity

- **Experiment ID**: `M0_OC_ACT_ARCHITECTURE_BUILD`
- **Track**: `MODEL`
- **Stage**: `ARCHITECTURE_IMPLEMENTATION`
- **Mode**: `ENGINEERING`
- **Dataset State**: `TRAINING_READY = TRUE`
- **Formal Training**: 禁止
- **Formal Dev/Test Evaluation**: 禁止
- **Real-Robot Execution**: 禁止

本轮唯一目标：

> 在现有 LeRobot ACT 基础上实现第一版 OC-ACT（Object-Conditioned ACT），完成结构、shape、梯度、checkpoint 与 baseline-regression 验证，为下一轮正式训练做准备。

本轮不允许根据 smoke loss、短跑训练或单样本输出得出科研结论。

---

# 2. Frozen Scientific Intent

第一版 OC-ACT 只解决：

> **Position / target-aware representation**

不在本轮实现 Force 模块。

核心研究思想：

```text
原始 ACT 视觉与策略主干
+
Top-camera object-aware DINO branch
+
Object tokens
+
Cross-Attention
+
Gated Residual Injection
```

目的：

> 在不替换 ACT 主干的前提下，引入显式目标物体与空间信息，验证其是否能成为后续 Position research 的可控 intervention。

---

# 3. Input Contract

## 3.1 Cameras

正式输入：

```text
Top RGB
Wrist / Gripper RGB
```

要求：

- Top RGB 同时进入原 ACT Top branch 与 DINO branch；
- Wrist RGB 只进入原 ACT Wrist branch；
- 第一版不允许 Wrist DINO。

---

## 3.2 Robot State

正式输入固定为：

```text
8D NOMASTER
```

禁止任何模型路径读取：

```text
master_gripper.pos
```

具体字段顺序必须来自 D1 已冻结 schema/config，不得凭记忆重新定义。

---

## 3.3 Action

保持现有：

```text
6D action
```

不改变：

- action semantics；
- ACT chunking semantics；
- action normalization semantics。

---

# 4. Baseline Preservation

第一版必须完整保留：

```text
Original ACT Top visual backbone
Original ACT Wrist visual backbone
Original ACT encoder
Original ACT decoder
Original ACT action head
```

不得为了 OC-ACT 方便而重写 baseline 科研语义。

必须做到：

```text
ACT_BASELINE
OC_ACT
```

可以通过 config 独立选择。

---

# 5. OC-ACT Frozen Architecture

总体结构：

```text
                         Top RGB
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
    Original ACT Top Backbone       Frozen DINOv2-S
             │                           │
             │                     Patch Features
             │                           │
             │                   Prototype Matching
             │                           │
             │                   Cup Similarity Map
             │                           │
             │          1 Global + 4 Local Cup Tokens
             │                           │
             │         semantic + (x,y) + confidence
             │                           │
             │                        Adapter
             │                           │
             └─────────────┐             │
                           │ Q         K,V
                     Object Cross-Attn
                           │
                           ▼
                     Gated Residual
                           │
Wrist RGB → Original ACT Wrist Backbone ─┤
Robot State (8D NOMASTER) ───────────────┤
                                         ▼
                               Original ACT Encoder
                                         ↓
                               Original ACT Decoder
                                         ↓
                                   Action Chunk
```

---

# 6. DINOv2 Branch

## 6.1 Encoder

第一版固定：

```text
DINOv2-S
```

要求：

- pretrained；
- frozen；
- Top camera only；
- 不接受 action loss gradient；
- 不参与 end-to-end fine-tuning。

如果工程环境暂时无法下载正式权重：

> 允许先完成接口 mock，但本轮完成前必须接回真实 pretrained DINOv2-S，除非报告 BLOCKED。

---

## 6.2 Freeze Verification

必须确认：

```text
requires_grad = False
```

并在一次真实 backward 后验证：

```text
DINO parameter grad = None / zero
```

---

# 7. Cup Prototype Bank

## 7.1 Purpose

Prototype Bank 用于定义：

> 当前 Top image 中哪些 DINO patches 更可能属于目标透明杯。

第一版不训练 detector。

---

## 7.2 Formal Prototype Source

正式 Prototype Bank 只能来自：

```text
Train split
```

或：

```text
独立 Reference Calibration Set
```

禁止使用：

```text
Dev
Final Test
```

---

## 7.3 Prototype Construction

第一版推荐支持：

```text
N reference images
↓
人工粗标 cup region / selected cup patches
↓
Frozen DINOv2-S features
↓
Prototype Bank
```

Prototype Bank 为：

```text
FROZEN
```

不接受 action loss gradient。

第一版不允许：

- online prototype adaptation；
- action-loss learnable prototype；
- Test-image prototype；
- 自动用 Final Test 优化 prototype。

---

## 7.4 Number of Prototypes

实现时应支持：

```text
num_prototypes = configurable
```

第一版建议默认：

```text
4–8
```

具体正式数量在下一轮 prototype calibration 中冻结。

M0 只要求接口支持，不要求本轮通过科研结果选择最优数量。

---

# 8. Patch Similarity

对每个 DINO patch feature：

```text
z_i
```

与 Prototype Bank：

```text
p_j
```

计算 cosine similarity。

第一版建议：

```text
s_i = max_j cosine(z_i, p_j)
```

或等价 soft aggregation。

要求：

- similarity map 保留 patch spatial layout；
- 不使用 hard binary mask 作为唯一表示；
- 支持 soft weighting。

---

# 9. Object Token Extraction

第一版冻结：

```text
K = 5
```

组成：

```text
T0 = Global Cup Token
T1 = Upper-Left Local Token
T2 = Upper-Right Local Token
T3 = Lower-Left Local Token
T4 = Lower-Right Local Token
```

---

## 9.1 Global Token

使用全部 patch 的 similarity-weighted soft pooling：

```text
T0_semantic
=
Σ w_i z_i
```

其中：

```text
w_i = soft weight from similarity
```

---

## 9.2 Local Tokens

使用 2×2 spatial partition：

```text
upper-left
upper-right
lower-left
lower-right
```

每个区域内部做 similarity-weighted soft pooling。

要求：

- 不将 2×2 网格画到真实图像；
- 它只是 feature-space pooling；
- local token 必须保留空间差异。

---

# 10. Explicit Spatial Information

每个 token 必须包含：

```text
semantic feature
+
x
+
y
+
confidence
```

其中：

- `(x, y)` 为该 token 对应 soft spatial center；
- 坐标需归一化；
- 推荐范围：

```text
[-1, 1]
```

- `confidence` 为该 token 的 similarity mass / normalized confidence。

第一版不得只用 DINO semantics 而丢掉显式位置。

---

# 11. Adapter

Adapter 作用：

```text
DINO token representation
→ ACT d_model
```

允许：

```text
Linear
```

或：

```text
Linear
→ LayerNorm
→ GELU
→ Linear
```

要求：

- trainable；
- 参数量保持小；
- 不设计大网络；
- 输出维度严格等于 ACT hidden dimension。

---

# 12. Cross-Attention Injection

第一版固定：

> **Pre-Encoder Object Cross-Attention**

形式：

```text
Q = ACT observation / visual tokens
K = Cup Tokens
V = Cup Tokens

object_response = CrossAttention(Q, K, V)
```

然后：

```text
h' = h + alpha * object_response
```

再进入：

```text
Original ACT Encoder
```

---

## 12.1 Injection Position

第一版禁止：

- mid-encoder injection；
- decoder-only injection；
- multi-stage repeated injection；
- 同时比较多个 insertion points。

M0 只实现一个：

```text
pre-encoder
```

---

# 13. Gate

必须存在显式 gate：

```text
alpha
```

要求：

- trainable；
- 初始化接近 baseline；
- 推荐：

```text
alpha_init = 0
```

或非常小的非零值。

允许：

```text
scalar gate
```

优先。

第一版不要做复杂 dynamic gate。

---

# 14. Config Requirements

必须可以从 config 控制至少：

```text
model_type = act / oc_act

object_conditioning.enabled
object_conditioning.dino_model
object_conditioning.dino_frozen
object_conditioning.prototype_path
object_conditioning.num_prototypes
object_conditioning.num_object_tokens
object_conditioning.adapter_dim
object_conditioning.gate_init
```

字段名字可按项目代码风格调整。

要求：

- checkpoint save/load 后 config 能恢复；
- baseline config 不需要 object prototype 才能启动。

---

# 15. Recommended Code Organization

Codex 应优先：

> 最小侵入式实现。

推荐新增类似：

```text
lerobot/common/policies/act/
├── modeling_act.py
├── configuration_act.py
└── object_conditioning/
    ├── dino_encoder.py
    ├── prototype_bank.py
    ├── object_tokenizer.py
    └── cross_attention_adapter.py
```

具体路径可根据当前仓库真实结构调整。

原则：

- object-aware 代码尽量模块化；
- 不把 DINO / prototype / tokenization 全塞进单个巨大 forward；
- baseline path 清晰可见。

---

# 16. Mandatory Engineering Gates

## M0-G0 — Baseline Import / Construction

验证：

```text
ACT baseline
```

仍可正常：

- import；
- construct；
- load config；
- instantiate。

---

## M0-G1 — OC-ACT Construction

验证：

```text
OC_ACT
```

可正常构建，且：

- 无 prototype path 时给出清晰错误或 smoke mode；
- 有 prototype path 时正常加载。

---

## M0-G2 — Shape Contract

至少记录：

```text
Top RGB shape
Wrist RGB shape
8D state shape
DINO patch feature shape
similarity map shape
5 object token shape
adapter output shape
cross-attention output shape
ACT encoder input shape
action chunk output shape
```

---

## M0-G3 — DINO Freeze

真实 backward 后：

```text
DINO grad = none/zero
```

必须 PASS。

---

## M0-G4 — Trainable Gradient

至少以下模块必须获得非零梯度：

```text
Adapter
Cross-Attention
Gate
ACT trainable path
```

如果 gate 初始化为 0 导致某些下游梯度初始为 0：

> 必须解释并额外用非零 gate smoke 验证路径。

---

## M0-G5 — Gate Baseline Equivalence

使用相同：

- ACT weights；
- input；
- normalization；
- random state；

在：

```text
alpha = 0
```

时：

> OC-ACT 输出应在合理数值误差内接近原 ACT baseline。

如果不等价：

```text
FAIL
```

除非存在明确、可证明的工程原因。

---

## M0-G6 — Object Path Sensitivity

固定：

```text
Top RGB
Wrist RGB
state
```

改变 object token：

```text
normal
zero
shuffle
spatially shifted
```

在：

```text
alpha != 0
```

时，至少 downstream hidden 或 action output 应发生可测变化。

目的：

> 证明 object path 真的接通。

这不是科学机制 PASS。

---

## M0-G7 — Checkpoint Roundtrip

验证：

```text
save
→ reload
```

后：

- config；
- ACT weights；
- adapter；
- cross-attention；
- gate；
- prototype metadata；

均可恢复。

DINO pretrained path / identifier 必须可复现。

---

## M0-G8 — Baseline Regression

原 ACT 至少完成：

```text
forward
loss
backward
checkpoint save/load
```

不得因新增 OC-ACT 代码破坏。

---

## M0-G9 — Resource Smoke

记录：

```text
total params
trainable params
frozen params
checkpoint size
peak VRAM
forward latency smoke
```

分别报告：

```text
ACT
OC-ACT
```

当前只做工程 profiling。

正式 deployment threshold 留到 G6 Deployment Equivalence 阶段冻结。

---

# 17. Prototype Smoke Policy

M0 允许：

```text
SMOKE_ONLY prototype bank
```

但必须明确标记：

```text
NOT_FOR_FORMAL_TRAINING
```

例如：

- 从 Train 中临时选少量 frame；
- 人工构造 random prototype；
- synthetic prototype。

这些只允许用于：

```text
shape / path / gradient test
```

不得作为下一轮正式 OC-ACT 训练 prototype。

---

# 18. Formal Prototype Bank Must Be Separate

M0 结束后，在正式训练前必须单独完成：

```text
P0 — Prototype Calibration
```

该阶段负责冻结：

- reference image 数；
- reference selection；
- cup region marking protocol；
- prototype 数；
- prototype bank file；
- train-only provenance；
- object similarity visualization。

M0 不越权完成这些科研选择。

---

# 19. Explicit Non-Goals

本轮禁止：

- 100k 正式训练；
- Dev model selection；
- Final Test；
- 多 seed；
- DINO fine-tuning；
- SAM；
- FCOS；
- CutLER；
- second detector；
- Wrist DINO；
- Force module；
- gripper controller；
- real robot；
- 多 injection variant；
- 自动 architecture search。

---

# 20. Result Deliverables

建议：

```text
experiments/model/m0_oc_act_architecture/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── architecture.md
    ├── changed_files.txt
    ├── config_example.yaml
    ├── figures/
    │   ├── object_similarity_smoke.*
    │   ├── object_token_layout.*
    │   └── output_sensitivity.*
    └── proposal.md        # optional
```

---

# 21. Validation Report Must Include

至少回答：

1. baseline 是否仍可运行？
2. OC-ACT 是否可正常构建？
3. DINOv2-S 是否冻结？
4. 5 个 object tokens shape 是否正确？
5. x/y/confidence 是否实际进入 token？
6. adapter 是否有梯度？
7. cross-attention 是否有梯度？
8. gate 是否工作？
9. alpha=0 是否近似 baseline？
10. object perturbation 是否改变 downstream output？
11. save/load 是否通过？
12. 资源增量是多少？

---

# 22. Codex MAY

- 修改 ACT config；
- 修改 ACT model construction；
- 新增 object-conditioning modules；
- 下载/接入 DINOv2-S；
- 编写 smoke prototype bank；
- 编写 tests；
- 编写 profiling；
- 编写 visualization；
- 创建 proposal.md。

---

# 23. Codex MUST

- 保持 baseline path；
- 使用 8D NOMASTER；
- 不读取 Final Test；
- 标记所有 smoke-only artifact；
- 保留 DINO frozen；
- 证明 alpha=0 baseline equivalence；
- 证明 object path sensitivity；
- 报告全部改动文件；
- 完成后停止。

---

# 24. Codex MUST NOT

- 修改本 decision.md；
- 启动正式训练；
- 根据 smoke loss 改科研结构；
- 用 Dev/Test 选 prototype；
- fine-tune DINO；
- 删除 original ACT branch；
- 把 Top backbone 完全替换成 DINO；
- 自行增加 Force branch；
- 自行增加其他 detector；
- 自行启动真机；
- 自行进入下一阶段。

---

# 25. Completion Rule

只有以下全部 PASS：

```text
M0-G0 PASS
M0-G1 PASS
M0-G2 PASS
M0-G3 PASS
M0-G4 PASS
M0-G5 PASS
M0-G6 PASS
M0-G7 PASS
M0-G8 PASS
M0-G9 PASS
```

才可标记：

```text
ARCHITECTURE_READY = TRUE
```

然后停止。

下一步由用户 + ChatGPT 创建：

```text
P0 Prototype Calibration
```

通过后再进入：

```text
Formal ACT Baseline Training
Formal OC-ACT Training
```
