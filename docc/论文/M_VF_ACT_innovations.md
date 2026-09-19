# M-VF-ACT 六大创新说明

> 任务场景：力控抓取半满透明塑料杯（SO-ARM101 机械臂，LeRobot v3.0）
>
> 基线模型：ACT (Action Chunking Transformer)

---

## 总览

```
输入 (Top/Wrist RGB + Robot State)
        │
        ▼
  ┌─ 创新1: Mask-Guided Perception ─── 解决"看不见" (透明杯 ≈ 隐形)
  │        SAM2 逐像素 mask 监督 → Backbone 学会识别透明物体 (梯度雕刻)
  │
  ├─ 创新2: FCOS Feature Injection ──── 解决"检测白学了" (检测知识锁在 loss 里)
  │        FCOS tower 中间特征 → 投影为额外 Encoder token (显式注入)
  │
  ├─ 创新3: Mask Feature Injection ──── 解决"分割白学了" (分割知识锁在 loss 里)
  │        Mask Decoder 融合特征 f432 → 投影+降采样为额外 Encoder token
  │
  ├─ 创新4: Visual-Force Fusion ────── 解决"摸不到" (夹爪接触杯壁, 模型不知道)
  │        力感信号作为独立 token → Transformer 中与视觉特征交叉注意
  │
  ├─ 创新5: Hybrid Action Head ─────── 解决"握不稳" (不同水量需要不同力度)
  │        位置头 + 力控头双路输出 → 自动调节夹持力度
  │
  └─ 创新6: Temporal Frame Stacking ─ 解决"看不准" (水晃动, 单帧无法预判)
           t/t-1/t-2 三帧堆叠 → 模型的时序上下文
```

---

## 创新 1: Mask-Guided Perception (✅ 已实现)

### 解决什么问题

透明塑料杯在 RGB 图像中几乎不可见——杯壁和灰色桌面融为一体，仅有微弱折射亮线。标准 ResNet18 无法从 RGB 中有效提取透明物体的视觉特征。

### 怎么做

| 阶段 | 做法 |
|------|------|
| **离线生成** | SAM 2 利用 CVAT 标注的第一帧 bbox 作为 prompt，对整段视频逐帧生成 mask（杯子区域=1，背景=0，高斯模糊软边界） |
| **训练时** | Mask Decoder 挂在 FPN 三路输出上，通过逐级上采样 + 通道拼接串行融合，输出 480×640 的预测 mask。与 SAM 2 GT mask 计算逐像素 L1 Loss |
| **训练时** | Mask Decoder 和 FCOS（检测头）在 FPN 上是并行兄弟节点。FCOS 提供稀疏监督（"杯子在这个框里"），Mask Decoder 提供密集监督（"每个像素是不是杯子"），两种梯度联合塑造 Backbone |
| **推理时** | Mask Decoder 不运行。但 Backbone 已被像素级梯度"雕刻"——学会从 RGB 中区分透明物体和背景 |

### 作用

使模型从 RGB 中"看见"透明杯子。通过像素级监督，Backbone 学到：
- 杯壁边缘的微弱折射亮线 = 杯子边界（P2 浅层特征）
- 灰色透明区域 = 杯子内部（P4 深层语义）
- 灰色桌面 = 背景（被 mask loss 抑制）

---

## 创新 2: FCOS Feature Injection (✅ 已实现)

### 解决什么问题

创新 1 的 Mask Decoder 和 FCOS 检测头对策略网络的贡献只有「梯度反传」一条路径——它们的输出（分类分数、bbox、mask）仅用于计算 loss，从未进入 Transformer Encoder。这意味着 Transformer 从未「看到」检测或分割的结果，只能间接依赖被梯度雕刻过的 backbone 特征。

**检测知识被锁在 loss 里，策略网络用不到。**

### 怎么做

| 阶段 | 做法 |
|------|------|
| **训练+推理** | 取 FCOS 的 `cls_feature`（128 维分类中间特征）和 `reg_feature`（128 维回归中间特征）在 tower 输出后、最终预测层前截取，通道维拼接为 256 维 |
| **门控** | 用 centerness 预测值做空间置信度门控：`combined × (1.0 + sigmoid(ctr_pred))`——高置信度区域放大到 2 倍，低置信度保持原样 |
| **投影** | 1×1 Conv 256 → dim_model (512)，得到与图像 token 同维度的 `(B, 512, H, W)` |
| **注入** | 添加与同视角图像 token 相同的 2D 正弦位置编码后 flatten，紧跟在图像 token 之后拼入 Encoder。默认只用 P4（15×20=300 tokens） |

### 与创新 1 的对比

| 机制 | 创新 1（纯梯度雕刻） | 创新 2（显式注入） |
|------|--------------------|-------------------|
| 检测信息如何到达策略 | 梯度 → FPN → Backbone → 间接影响 F4 | 中间特征 → 投影 → 直接作为 token |
| 推理时是否运行 | 不运行 | 运行（tower 前向，~3ms） |
| Transformer 能否显式关注检测区域 | 否 | 是（self-attention 跨 token 关联） |

---

## 创新 3: Mask Feature Injection (✅ 已实现)

### 解决什么问题

与创新 2 相同的问题——Mask Decoder 的中间特征包含丰富的「透明物体边缘/轮廓」信息，但从未显式进入 Transformer。创新 1 赌的是梯度雕刻足够强，创新 3 用显式注入补上这一环。

### 怎么做

| 阶段 | 做法 |
|------|------|
| **特征提取** | 取 Mask Decoder 的三层融合中间特征 `f432`（P2/P3/P4 reduce + upsample + fuse 两轮融合后的 32 维 × 60×80 浓缩语义），在 ×8 上采样之前截取 |
| **投影** | 1×1 Conv 32 → dim_model (512)，得到 `(B, 512, 60, 80)` |
| **降采样** | `adaptive_avg_pool2d` → (15, 20)，得到 300 tokens（避免 60×80=4800 tokens 爆炸） |
| **注入** | 添加 2D 正弦位置编码后 flatten，紧跟图像 token 和 FCOS 注入 token 之后拼入 Encoder |

### 为什么注入 f432 而不注入最终 pred_mask

- 最终 `pred_mask` 是 1 通道 × 480×640 = **307,200 像素**——全 flatten 会炸掉 token 数
- `f432` 是 32 维 × 60×80 = **4800 个 32-dim 向量**——降采样到 15×20 后仅 300 tokens
- 32 个通道各自编码了不同的语义模式（边缘方向、纹理类型、区域归属），比单通道 mask 信息量更大

---

## 创新 4: Visual-Force Fusion (📋 规划中)

### 解决什么问题

视觉无法判断夹爪是否接触杯壁。透明杯的视觉特征弱，接触瞬间视觉几乎无变化。当前方案把力感信号（gripper.load + gripper.curr）嵌入在 9 维 state 向量的同 1 个 token 中，被 600 个视觉 token 稀释——接触事件的关键信号被淹没。

### 怎么做

| 阶段 | 做法 |
|------|------|
| **数据** | 已采集完成。数据集 B 的 observation.state 中自动包含 gripper.load 和 gripper.curr（舵机 Sync Read 同步读取，无需额外标注） |
| **训练时** | 力感信号提取为独立 Force Token。可选两种融合方式：(a) 作为额外 1D token 加入 Transformer Encoder，与视觉 token 一起做 self-attention；(b) 通过轻量 Force Encoder（MLP）编码后，作为特殊 token 用 cross-attention 注入 Transformer |
| **推理时** | Force Token 正常传入（力感传感器在真实部署中实时可用，不需要 SAM2 那样的额外推理开销） |

### 作用

使模型"感知"到接触事件。夹爪触碰到杯壁的瞬间，load/curr 上升 → Force Token 激活 → Transformer 中自注意力机制将该信号与视觉 token 关联 → 模型学会"这个力感模式 + 这个视觉画面 = 该停止闭合了"。

**为什么独立 token 比嵌入 state token 好：** 600 个视觉 token 中，self-attention 权重自然偏向视觉。当力感在独立 token 中时，query 可以从数百个 key 中选择"哪些 key 和力感最相关"——本质上是在 "所有视觉位置中寻找接触点"。

---

## 创新 5: Hybrid Action Head (📋 规划中)

### 解决什么问题

不同水量的杯子需要不同的夹持力度。满杯重 → 需要大力捏紧防滑落。空杯轻 → 捏太紧会碎、捏太轻会掉。当前 action head 只输出 6 个关节位置（包括 1 个固定的夹爪位置），无法自适应。

### 怎么做

| 阶段 | 做法 |
|------|------|
| **数据** | 已采集完成。数据集 B 中自动记录了 master_gripper.pos（人类操作者捏遥控器的程度）。ΔP = master_gripper.pos - gripper.pos = 人类期望的力控刚度 → 天然训练标签 |
| **训练时** | Action Head 分两路：**位置头**（Linear→6，关节目标位置）+ **力控头**（Linear→1，gripper_effort_limit）。力控头用 human demo 的 ΔP 作为监督标签（L1 Loss） |
| **推理时** | 位置头输出关节角度序列，力控头输出夹爪最大扭矩限制 → 实时写入舵机寄存器 |

### 作用

模型根据视觉输入（水量、杯子位置）自动调节夹持力度。半满重杯→ΔP 大→捏紧。空杯→ΔP 小→轻握。**本质上是把人类操作者的经验——"看到这个杯子该使多大劲"——作为可学习策略编码进模型。**

---

## 创新 6: Light Temporal Modeling (📋 规划中)

### 解决什么问题

杯子移动时水面晃动。单帧图像只能看到"水面在这一刻的位置"，无法判断水面正在往哪个方向运动、速度多大。模型缺少时序上下文——无法预判水的运动趋势来做补偿动作。

### 怎么做

| 阶段 | 做法 |
|------|------|
| **数据** | 无需额外标注。Dataloader 自动从 MP4 中提取连续帧 (t, t-1, t-2) |
| **训练时** | 三帧在 channel 维度拼接（3ch×3帧=9ch），作为"多通道时间切片"输入 ResNet18。或三帧各自独立过 Backbone，三组视觉 token 都进入 Transformer，让自注意力学习跨帧关系 |
| **推理时** | 和训练一致——维护最近 3 帧的缓冲区，每次输入 t, t-1, t-2 |

### 作用

使模型看到水的运动趋势。t-2 水面在左边，t-1 在中间，t 在右边 → 模型推断水正在向右晃 → 输出轻微左倾补偿动作（预测性控制）。**本质上是把"单帧快照"变成"3帧短视频"，让模型学会运动动力学。**

---

## 六个创新的架构层级关系

```
                      输入层
                   Top / Wrist / State
                        │
  ┌─────────────────────┤
  │                     │
  ▼ 创新1: Mask (梯度)  ▼
  SAM2 Mask Decoder     共享 ResNet18 Backbone
  (像素级语义监督)      (层1→层2→层3→层4)
  │                     │
  │              ┌──────┴──────┐
  │              │             │
  │              ▼             ▼
  │            FPN        创新6: Temporal
  │         (特征金字塔)   (t/t-1/t-2 帧堆叠)
  │              │
  │     ┌────────┼────────┬──────────┐
  │     ▼        ▼        ▼          ▼
  │  FCOS    Mask Dec   Fusion  创新2: FCOS 注入
  │ (检测)   (分割)    (注意力)  cls+reg tower → tokens
  │     │        │        │          │
  │     │        ▼        │          │
  │     │  创新3: Mask注入│          │
  │     │  f432 → tokens  │          │
  │     │        │        │          │
  │     └────────┼────────┼──────────┘
  │              ▼        ▼
  │        Transformer Encoder  ←── 创新4: Force Token
  │              │
  │              ▼
  │        Transformer Decoder
  │              │
  │              ▼
  │         Action Head  ←── 创新5: 位置头 + 力控头
  │              │
  │              ▼
  │         动作序列 (chunk_size, 6)
  │
  └──── 创新1: 训练时梯度监督 (推理丢弃) ────┘
  　　  创新2/3: 训练+推理时显式注入 (推理保留) ────┘
```

---

## 实施优先级与决策依据

| 优先级 | 创新 | 触发条件 |
|--------|------|----------|
| **Phase 1** (已实现) | Mask-Guided Perception (创新1) | 透明杯视觉感知难（明确已知问题） |
| **Phase 1** (已实现) | FCOS Feature Injection (创新2) | 检测知识不进入策略 → 浪费（架构缺陷） |
| **Phase 1** (已实现) | Mask Feature Injection (创新3) | 分割知识不进入策略 → 浪费（架构缺陷） |
| **Phase 2** (待实验后定) | Visual-Force Fusion (创新4) | 实验失败主因 = 接触判断延迟 / 滑落 / 捏碎 |
| **Phase 3** (待实验后定) | Hybrid Action Head (创新5) | 实验失败主因 = 不同水量下抓取不稳定 |
| **Phase 4** (待实验后定) | Temporal Modeling (创新6) | 实验失败主因 = 移动时水晃动导致丢失 |

---

## 各创新对训练/推理的影响

| 创新 | 训练额外开销 | 推理额外开销 | 额外标注需求 |
|------|-------------|-------------|-------------|
| Mask-Guided Perception (创新1) | SAM2 离线生成 (一次性) + Mask L1 Loss | **零** (Mask Decoder 丢弃) | 需要 (CVAT bbox 作为 SAM2 prompt) |
| FCOS Feature Injection (创新2) | FCOS tower 前向 (极小) | FCOS tower 前向 (~3ms) | **零** (复用检测标注) |
| Mask Feature Injection (创新3) | Mask Decoder 前向 f432 (极小) | Mask Decoder f432 + pool (~2ms) | **零** (复用 SAM2 mask) |
| Visual-Force Fusion (创新4) | Force Token 编码 (极小) | Force Token (传感器实时可用) | **零** (力感自动采集) |
| Hybrid Action Head (创新5) | 力控头 L1 Loss | 输出额外 1 维 + 写舵机寄存器 | **零** (ΔP 自动采集) |
| Temporal Modeling (创新6) | 3帧堆叠后 forward | 维护 3 帧缓冲区 | **零** (自动帧组装) |

**关键变化**：创新 2 和创新 3 是第一批「推理时有额外计算」的创新——FCOS tower 和 Mask Decoder 中间层在推理时运行。额外开销约 5ms（vs 创新 1 的零开销），换来的是检测/分割知识的**显式注入**。是否值得，由消融实验 E7-E9 回答。

**创新 4/5/6 的数据需求已在现有数据集中自动满足——不需要额外标注。**
