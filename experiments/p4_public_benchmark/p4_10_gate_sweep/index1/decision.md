# P4.10-r0 — Gate 小规模参数扫描与 Object Path 有效性受控实验

- 状态：**待执行**。决策制定：用户 + ChatGPT；执行：唯一的本地 Codex 通过 SSH 调度服务器；不得擅改本文件。
- 前置：P4.8-r5/r6、P4.9-r1 已完成；P4.7 真机流程不受影响。
- 本轮定位：**探索性小预算机制实验**，不直接用于论文性能宣称、不覆盖旧权重、不安排无人值守真机。

## 1. 已有证据与问题

1. LIBERO r5：ACT、Legacy P1、Fixed P1 在同一单任务 10k、10 个配对初始状态的成功率均为 4/10，Object ON/OFF 动作差约 1e-7 到 1e-8，Gate 接近零。
2. LIBERO r6：Gate 前 Residual 非零，改变诊断 Gate 可引起剂量依赖的动作变化；Gate=0 时 Object 分支内部训练参数的初始梯度受阻。但强制 Gate 并未证明成功率提升。
3. P4.9-r1：formal6 L1/F3 的 Gate+0.001 在 80k Dev15 误差均退化；LIBERO Legacy Gate+ 10/20、Fixed Gate+ 6/20，相对原版 8/20，结果不一致。formal6 历史初始化并非逐字节配对，不能将差异完全归因于 Gate。
4. 用户希望小训练预算多试 Gate，并判断 Object Path 是否**真正有效**；不能用 Gate α、ON/OFF 差异单独充当有效性证据。

## 2. 冻结主要问题与假设

- Q1：Gate=0 初始梯度抑制是否是 LIBERO 10k 失活的重要因素？
- Q2：非零 Gate 初值的改进是否稳定，是否存在过强初始化损害训练？
- Q3：Object Path 比没有 Object Path 的**重新训练**基线是否提高闭环成功率？
- Q4：Object Path 是否关注任务目标，还是通过背景/机械臂/数据偏差干扰预测？
- Gate 数值和动作变化只是机制指标；主判断依赖配对闭环成功率，辅以离线误差、目标匹配诊断及训练稳定性。

## 3. 冻结比较对象和唯一变量

### 3.1 LIBERO：优先 Fixed P1 以降低试验规模

先冻结 LIBERO-Object 已跑通的 task index 24、Train35 / Dev9、相同 4×384 Prototype Bank、seed、batch、数据顺序、完整 ACT 主干、DINO 单次预处理、数据时基与 rollout 控制器。**不修改原型规则、归一化或参数结构**。

探索性 Gate 初始化扫描（**所有 Gate 均为可学习标量**）：

| 实验 ID | Model | Gate 初始值 | 训练步数 | 备注 |
|---|---|---:|---:|---|
| LIB-F-G0 | Fixed P1 | 0 | 已有 10k | 复用 r5 |
| LIB-F-G1e-3 | Fixed P1 | +0.001 | 已有 10k | 复用 r7/r9 |
| LIB-F-G1e-2 | Fixed P1 | +0.01 | 5k 探索 | 新训练 |
| LIB-F-G5e-2 | Fixed P1 | +0.05 | 5k 探索 | 新训练 |
| LIB-F-G1e-1 | Fixed P1 | +0.1 | 5k 探索 | 新训练 |
| LIB-F-GNEG | Fixed P1 | -0.01 | 5k 探索 | 新训练；测试符号敏感性 |
| LIB-F-NOOBJ | ACT 主体、无 Object 分支 | 不适用 | 5k 探索；若已有同配方模型可复用 | **从头训练**的模块消融 |

注：Gate=0 和 +0.001 已有 10k 结果不得与新 5k 结果直接排名。为了公平的 **5k 横向扫描**，从已有训练结果加载当时 5k checkpoint（必须验证状态）；如果不可得，从初始权重/同配方数据流重训到 5k。所有新 Gate 在相同训练步数、同一个 seed、同样训练流下比较。保留 0.0、+0.001 既有 10k 结果作为历史参照，明确将 5k 与 10k 分表展示。

**Gate 固定与可学习 Gate 的比较属于不同机制试验**：仅在上述扫描后如仍有时间，再增加一个固定 α=+0.01 的 5k 对照；此项必须保持 alpha 不在优化器内，明示它并不是“仅改变初始值”，不与学习门控扫描混成单变量比较。

### 3.2 Legacy P1 验证，不全面网格搜索

仅当 Fixed 的 5k 探索出现可复现的机制差异，使用 **同样 Gate 初值** 在 Legacy 上做一次 5k 配对；必须保留 Legacy 在线双重标准化，仅改变 Gate 初值。历史 Legacy Gate=0 与 +0.001 结果保留，不为了看起来更好而重新挑 seed。

### 3.3 formal6 真机数据小预算验证

**不能直接将 LIBERO 最优 Gate 定为真机最优值**。在 LIBERO 提供可信机制候选后，最多选 1–2 个 Gate 值，分别以 L1 seed1、F3 seed3 做 formal6 **20k pilot**，与相同 seed/相同训练步数的 Gate=0 基线匹配；若找不到原历史 20k 的可比数据与冻结配置，先补齐 baseline，而不是错误地同 80k 比。Train90、Dev15、Final15 规则不变；不新增真机部署授权。

## 4. 附加关键证据：Object Path 是否真正有效

每个新模型在同一固定输入上输出：

- train/dev loss、Arm/Gripper normalized MAE（相同 chunk_valid 定义）、Gate α 曲线与梯度曲线；
- Gate 前 Residual norm、Query norm、Gate 后相对比率、Cross-Attention/Adapter 梯度及参数变化；
- Object ON/OFF 输出差（仅用于诊断，不当作独立性能收益）；
- 目标罐头参考帧与独立 Train/Dev 帧的 DINO 匹配热图，明确参考图自一致性与跨帧跟踪差别；
- **单独训练的 NOOBJ 基线**对照（不是临时把同一个 P1 的 Gate 置零）。

闭环评估：冻结与模型选择相互隔离的初始状态，至少 20 组配对 rollout 进行探索，保存每次 success、执行步数、失败阶段、视频与环境版本；**不要继续拿 10–29 的旧状态调参后又当盲测**。最终报告同样初始状态下的配对成功/失败表及不确定性，不夸大 20 次的小样本差异。可在探索集筛选 1–2 个 Gate 值后，使用未看过的 held-out init 作一次确认性对照；held-out 在筛选完成前不可见。

## 5. 资源、并行与无人值守调度

- 不抢占现有 P4.7 真机作业/服务器作业；先列出进程、GPU、磁盘与持久化会话。
- 采用单 Codex SSH 下的持久化队列；先各取短窗口实测单/双/多进程总有效吞吐和显存/CPU IO，再选择吞吐最高且安全的并行数；不要凭 GPU 利用率定并发。
- 执行优先级：配置校验 → Fixed 5k 扫描 + NOOBJ → 统一机制诊断/闭环 → 条件满足再 Legacy 5k → 条件满足再 formal6 20k → 保存/汇总。
- 每个模型必须有独立输出目录、配置哈希、初始化权重指纹、prototype SHA、训练状态、数据顺序证据与 checkpoint；5k 完整可恢复，不混用 optimizer。
- 容许服务器长时运行，但停止条件为**计划内结果完成或资源/兼容性阻断**；不可自动扩大训练预算、修改任务、跑 Final15 或发布真机模型。

## 6. Gate 判定标准与停机规则

- 工程 PASS：所有训练配方、seed、输入/输出、DINO 标准化与 Prototype 共享经过审计，checkpoint 可完整恢复。
- 机制 PASS：非零 Gate 组出现可重复梯度与 Residual 增益，且无数值异常；不代表方法有效。
- 方法有效性候选：相同预算的 P1 vs 从头训练 NOOBJ/ACT，在独立初始状态上有一致的闭环成功收益，且热图/干预诊断不支持单纯背景捷径。仅 5k/20 rollouts 时只能称 *exploratory signal*。
- 若 Gate 扫描只使 ON/OFF 增大而闭环无收益、或数值明显不稳定，**停止扩大 Gate 网格**，转入 Prototype 目标跟踪与特征融合研究，不自行再试新的初始化参数。
- 如果 5k 训练整体尚未学到任务而全部零成功，不能得出 Object 无用结论：先报告预算不足，后续须单独决定统一延长预算。

## 7. 交付目录与报告

建议：`experiments/p4_public_benchmark/p4_10_gate_sweep/index1/decision.md`，结果放 `result/`。包括：

- `execution_manifest.json`（源码提交、环境、训练协议及 dataset revision）
- `gate_sweep_config_matrix.csv`、`checkpoint_manifest.csv`
- `parallel_throughput_benchmark.csv`、`train_metrics.csv`
- `dev_comparison.csv`、`object_path_diagnostics.csv`、`gradients.csv`
- `rollout_per_init.csv`、`paired_summary.md`、`prototype_heatmaps/`
- `builder_review.md`、`red_team_review.md`、`blocking_proposal.md`（如需）
- `summary.md`：报告是否存在 Object Path 的**闭环收益**、是否仅数值激活、是否需进一步确认。

## 8. 不可变约束

1. 本 decision 不替代历史 P1 真机主模型，不修改已上传 Hugging Face 模型。
2. 不得使用 Dev9/held-out 测试来选择/手工调整原型位置；原型来自已冻结 Train 图像的 Codex 辅助目标坐标。
3. LIBERO 中只验证 Position/Object-Centric；不注入不存在的 Load/Current，也不改成 New-GL。
4. 发现新模型或配方兼容性问题，必须停止相关分支并提交 PROPOSAL，不能擅自变更网络定义。
5. 不得将本轮探索性最佳 Gate 直接解释为论文主结论；正式结果需要相同预算、多 seed、独立初始状态的确认性实验。
