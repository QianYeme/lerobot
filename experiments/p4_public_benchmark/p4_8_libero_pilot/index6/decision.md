# decision.md — P4.8-r6 LIBERO Object Path 机制诊断（只读优先）

- 决策状态：**FROZEN / AUTHORIZED FOR DIAGNOSTICS ONLY**
- 实验身份：`P4_8_r6_OBJECT_PATH_MECHANISM_DIAGNOSTIC`
- 建议仓库路径：`experiments/p4_public_benchmark/p4_8_libero_pilot/index6/decision.md`
- 前驱：P4.8-r5 `index5` 三模型 10k 配对 Pilot；G0–G4 已完成。
- 研究责任：用户 + ChatGPT 冻结科研语义；唯一的**本地 Codex** 通过 SSH 连接 RTX 5090 执行；不允许 Codex 自行启动 r7 或修改本 decision 的核心假设。

## 0. 本次 decision 做什么 / 不做什么

**做：**在不改模型、不重训、不打开 Final 数据的前提下，解释为什么 LIBERO 单任务 10k 时 Legacy/Fixed P1 的 Object Path ON/OFF 差异约为数值噪声级别。复用 5k/10k checkpoint、Dev9/Train35、固定观测、固定 LIBERO 初始化；输出可检验的因果链与有边界的下轮建议。

**不做：**不得修改 Gate 初值、模型结构、DINO 预处理、Prototypes、训练超参数；不得对诊断过的 Dev9 调参后宣称泛化；不得自动进行长训练、多任务、真机、上传 HF 或发布；不得将诊断性的强行开启 Gate 结果作为正式任务成功率；不得更改 r5 资产。

## 1. 冻结上下文和关键证据

- 任务：LIBERO-Object task0，alphabet soup 罐头放入篮子；Train35/Dev9；四个 Codex 辅助 Train-only 目标中心构建的 4×384 DINOv2-S 原型，**Legacy/Fixed 共用相同 bank**，来源中包含少量执行方目标位置先验，不称为完全无监督。
- r5 训练：ACT、Legacy P1、Fixed P1 各 seed1、10k；paired initial common ACT 权重和 Legacy/Fixed 初始权重一致；只有在线 DINO 标准化不同。
- r5 闭环 init10–19 均为 4/10，相同 init10–13 成功、14–19 失败；pilot 不支持性能排名。
- r5 10k Gate：Legacy `-6.7515434e-6`；Fixed `-4.15e-7`（精确值从 checkpoint 再读）。ON/OFF 臂部动作差约 1e-8–1e-7；两个 gate 接近零。
- r5 训练时序：原始数据 FPS 元数据 10，LIBERO 控制 20Hz；保留 native one-action-to-one-env-step 协议，不在 r6 改动。
- r5 产物是服务器上的真实权重；上传 ZIP **只有 manifest/日志及 SHA 指针，不包含 checkpoint 权重**。必须通过 SSH 核对真实路径、哈希和配置。

## 2. 研究问题 / 假设（不得提前预设答案）

- H1：**标量 Gate 抑制**。alpha 极小，导致即使 object residual 本身有效，乘积在 Encoder 注入尺度上也近零。
- H2：**梯度动力学**。alpha 初值 0 导致 object-path 内部参数在初始阶段梯度受限；其后仍未获得有效梯度/稳定上升。
- H3：**视觉定位失效**。DINO 与 Prototypes 的相似度/空间热图没有持续指向 alphabet soup 目标，或 Global/Local tokens 弱/趋同。
- H4：**残差与主干尺度**。Cross-Attn 输出或归一化、主干输入尺度不匹配，导致门控前信号本身弱，或正常前向有覆盖/旁路。
- H5：**策略替代路径**。主干足够拟合训练任务，从而无需 object 信号；或者 Object Path 对长尾状态有效但目前抽样不足。
- H6：**测量误差**。原 ON/OFF 实现、评估 precision、采样范围导致近零差异的诊断假象。

## 3. 执行依赖与数据锁

1. 读取并核验 r5 `result/checkpoint_manifest.csv` 里 act/legacy/fixed 的 5k/10k `resume.pt`，记录路径、sha256、config SHA、源码 commit、prototype SHA、数据 split。找不到完整权重时 `BLOCKED_MISSING_CHECKPOINT`，不得凭 ZIP 推断梯度。
2. 从冻结的 Dev9 按 episode ID 固定挑选 `start / ~50% / end` 三帧，形成统一观察集合（最多 27 帧）；**再用 Train35 少量同规则帧复核诊断，不能选择最有利帧**。保留具体 ID、frame、图像哈希和筛选规则。
3. 设置 deterministic eval、禁用随机增强/Dropout，保持与 r5 同样的 Camera/State/Action normalization 和模型映射。检查 Fixed/Legacy 预处理确实只差在线 DINO 标准化，禁止自动修复 Legacy。
4. 一切读数与模拟探针使用新的 `index6/result/`，禁止写入或覆盖 `index5` 文件。

## 4. 分层诊断任务（按顺序）

### D0. 测量与加载正确性

- 对 10k checkpoint 复现 r5 相同输入的 `ON/OFF` 输出（完整 100-step chunk + h0）；使用完全一致的 backbone 与 eval 模式。
- 将 Gate 在 hook 中设为 0，并与物理绕过 Object Residual 的输出比较；预期数值容差内一致。分别测 fp32 与当前训练数值模式，以排除精度阈值误判。
- 对输入、原型、CNN feature、DINO feature、object tokens、Encoder query、raw residual、gated residual、预测动作做 shape/finite/dtype/NaN 检查。
- **禁止用纯 `gate != 0` 判定 object-path 真正生效。**

### D1. 从输入到动作的信号尺度解剖

对 Legacy/Fixed 两者（5k 和 10k）在同一固定观测集合上，记录分布：

- `alpha` 原始值和绝对值；`||Q||`、`||Object tokens||`、`||Raw CrossAttn residual||`、`||alpha * residual||`、`||Encoder input||`；报告 `||alpha*residual|| / max(||Q||,eps)` 的 p50/p90/max。
- ON/OFF 的 `h0`、全部 chunk 前若干离散索引及全 chunk 均值/最大绝对误差，按 arm / gripper 拆分，既给归一化又给原始 LIBERO OSC 单位。严禁把 ON/OFF 值直接理解为成功率提升。
- 同一输入下做**只读强制 Gate 剂量响应**：alpha = 原 checkpoint 值、0、+0.001、+0.01、+0.1 和 -0.01（具体是否数值溢出必须先断言，超出正常输出需标记 `OOD_PROBE`）；每次仅临时 hook，不写 checkpoint、不执行仿真物理动作。
- 记录 `action delta` 随 alpha 的趋势并解释：若强制门控可改变动作，则网络通路存在但正常 Gate 处于关态；改变动作不等于动作变好。

### D2. 梯度与训练动力学

- 读取 5k、10k 的 Gate 和 Object adapter/attention 参数、optimizer state；如仅有 5k/10k 两点，不得伪称具有完整 Gate 曲线。
- 使用当前 checkpoint 和**冻结 Train35 诊断 batch** 进行不更新参数的 backward（独立图、`optimizer.step()` 严禁调用）测 `dL/dalpha`、`dL/d(object adapter)`、`dL/d(cross-attention)`、`dL/d(CNN trunk)`；报告 L2 norm、零比例、finite 比例与 gate 对 upstream gradient 的代数影响。
- 若必须验证 alpha=0 的初始化梯度，则重建**等价初始模型**并校验 r5 initial hash；记录 `alpha=0` 时 upstream path 的梯度能否传播。无法验证原始初始化则明确 `INIT_GRAD_UNVERIFIED`，禁止构造为历史事实。
- 比较 legacy/fixed 和 train/dev 的梯度，禁止用 Dev9 做参数更新。

### D3. Prototype 匹配与 Object Token 语义

- 固定来自 Train35 的四张已存参考图，用 r5 真正的 DINO preprocessor 和 bank，输出每帧 16×16 patch 相似度热图、四个固定 2×2 local 区域权重、token pooling/坐标和 top-k patch 位置。
- 对一组不同阶段的 Train/Dev 帧作相同可视化；Codex 查看并记录“疑似在物体/机械臂/背景/遮挡”的**定性描述**，但不把这种描述冒充像素级真值或盲测准确率。
- 如需要比较已有四个 Codex 指定 ROI，仅将它们用于参考帧的有限 sanity check；不可用 Dev9 手动挑 ROI 重建 bank，也不可利用仿真隐藏 segmentation 或物体状态泄露目标位置。
- 明确分开：prototype 建库**单次**标准化；Legacy 在线**双次**，Fixed 在线**单次**。

### D4. 低成本有效性反事实探针（只读）

在同一固定输入上比较：真实 bank、同样尺度的**固定随机打乱 bank 槽位**、随机无关 Train-only patch bank（不重新构建训练用 bank）、同一 Top 帧变化与 Wrist/State 不变的有限探针（如技术允许）。这些只是网络依赖性诊断，**不是正式架构消融或可发布性能数字**。

不允许改 global/local token 数、温度 0.20、adapter 结构、attention Q/K/V 位置或修改时间窗口。

### D5. 训练/仿真结果溯源

- 对 r5 `libero_rollouts.csv` 中成功 init10–13 与失败 init14–19 的现有轨迹视频与观测，检查是否均有可观测的关键事件；不能根据单个 success flag 推断抓取失败阶段。
- 如需要单独采集新的不改策略的诊断 rollout，必须不影响 r5 正式结果，记录 init、seed、次数；优先复用现有视频。

## 5. 解释判据与输出结论

最终只能选择以下一个主状态（可附多个并存子机制）：

- `GATE_SUPPRESSION_SUPPORTED`：raw residual 有尺度、强制 gate 能可靠改变输出，而实际 gate 极小；仅说明机制受抑制，不保证增强有益。
- `UPSTREAM_SIGNAL_WEAK`：即使按规范临时打开 gate，残差信号仍弱或无明显输出影响。
- `PROTOTYPE_SEMANTICS_UNRELIABLE`：多帧可视化一致显示注意力主要落非目标、原型匹配不稳定；须把没有真值标注的证据边界写清。
- `GRADIENT_PATH_CONSTRAINED`：经过梯度检查确认路径存在有效限制，并记录确切作用层。
- `MEASUREMENT_ARTIFACT`：复现发现旧诊断有实现、精度或聚合错误。
- `MULTIFACTOR_OR_INCONCLUSIVE`：多原因并存或现有样本不足判别。

**禁止**根据 `alpha≈0` 单独断言“Gate 是根因”；禁止根据相关性就宣称已证明真实操作效果。

## 6. 资源和执行调度

- 这是**短时诊断**，不申请新的 10 小时训练租用；优先在现有 RTX 5090 空闲时跑 GPU probe，CPU 端汇总图表。不能影响正在执行的 P4.7 真机或其他正式训练。
- 一个本地 Codex 通过 SSH 调度；可对 Legacy/Fixed **独立诊断并行**，但先做进程/显存与合计吞吐检查；如果影响其他任务则顺序执行。
- 完整模型权重存服务器；仅输出小型结果包（数值 CSV/JSON、关键热图、归因报告、路径+SHA manifest）。不得上传私有模型权重或 HF token。
- 禁止新训练、`optimizer.step`、任何权重保存/覆盖及新的正式性能比较；仅可创建临时 gradient diagnostic graph。

## 7. 必须交付的结果目录

```
index6/
  decision.md
  result/
    status.json
    upstream_manifest.json
    input_lock.json
    load_and_measurement_audit.md
    object_signal_scales.csv
    gate_dose_response.csv
    gradients.csv
    checkpoint_5k_10k_comparison.csv
    prototype_heatmap_index.csv
    prototype_heatmaps/...
    object_dependency_probes.csv
    baseline_rollout_observations.md
    mechanism_findings.md
    next_experiment_proposal.md
    builder_review.md
    redteam_review.md
    summary.md
```

每个 CSV 要有 model、checkpoint step、episode/frame 或 probe ID、数值单位、测量位置和 eval dtype；每张图片记录来源及 SHA。`next_experiment_proposal.md` 只能提出可供用户+ChatGPT 审批的**单变量 r7 候选**，不得自动执行。

## 8. Builder / Red-Team Gate

**Builder PASS 条件：** r5 checkpoint SHA/输入/prototype 锁定；D0–D4 证据完整；Legacy 与 Fixed 测量同协议；未触碰 r5 checkpoint、未执行训练；材料能在服务器重复生成。否则 `PARTIAL_PASS` 或 `BLOCKED`。

**Red-Team 必查：** 诊断 alpha 强制值属于 OOD 探针、没有把它们当作正式策略；没有用 Dev9 标注优化 prototype；没有混淆 DINO 双标准化与 Gate；没有把 10 次相同成功/失败夸大为模型等价；没有把 ON/OFF 当作重新训练的严格消融；每个归因均列竞争性解释。

独立 Builder / Red-Team 复核若未实际执行，只能标记 `PENDING`，不能虚构其审查通过。

## 9. 下一轮决策边界

- 若 `GATE_SUPPRESSION_SUPPORTED`：r7 才考虑**单变量** Gate 初始化/学习率/门控优化等对照，控制 seed、训练时长、数据顺序并至少保留 Vanilla ACT/Legacy/Fixed；不一次改多项。
- 若 `PROTOTYPE_SEMANTICS_UNRELIABLE`：先研究 prototype 质量控制与其额外监督来源，不能悄悄重建训练 bank 伪装原网络。
- 若 `UPSTREAM_SIGNAL_WEAK`：核对实现/数值尺度，任何融合改造应明确命名新网络，原 P1 保留为冻结基准。
- 若不确定：保留 r5 结果，进一步只增加**诊断证据**，不按 4/10 成功率盲目扩展到全套 LIBERO。

**本 decision 授权完成后停止；由用户与 ChatGPT 验收结果，再冻结 r7。**
