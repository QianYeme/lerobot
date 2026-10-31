# P4.10-r1 — Gate Selection → Final L1 / F3 80k Candidates

- **状态**：待执行。科研决策由用户 + ChatGPT 冻结；唯一的本地 Codex 负责通过 SSH 远程实施，不可自行改研究假设或阈值。
- **承接**：P4.10-r0 5k Fixed P1 LIBERO Gate Sweep 结果；P4.9-r1 formal6 +0.001 80k 结果；P4.8-r6 诊断。原训练模型及真机 P4.7 工作不受影响。
- **最终目标**：经过低成本、可追溯筛选后，确定 Gate 配置，使用 formal6_merged Train90、历史配置和历史最佳 seed 从头训练 **L1 Gate-Selected Seed1 80k**、**F3 Gate-Selected Seed3 80k**；与历史 L1/F3 对照并进行受监督真机验收。**当前文件仅授权门控选择阶段，不自动授权最终 80k 和真机运动**；通过门槛后形成冻结的后续训练 decision / approval。

## 1. 问题和现有证据

- LIBERO Fixed 5k 统一配对的探索结果：No-Object ACT 12/20；Gate=0 12/20；+0.001 8/20；+0.01 12/20；**+0.05 14/20**；+0.1 10/20；-0.01 9/20。+0.05 相比 ACT 配对净增两次成功，样本量不足，且存在多配置筛选偏倚；不能声称最优。
- Gate+0.05 / +0.1 使 Object ON/OFF 动作差及梯度显著增加，能证明**分支参与动作**，不能证明其视觉语义正确或闭环增益稳定。
- formal6 上 +0.001 Gate+ 的两组 80k Dev15 相对于原版 L1/F3 均退化，且有 Object Path 实际贡献减弱的现象；**不能假定 LIBERO 最佳 Gate 直接迁移真机**。
- Legacy(L1) 与 Fixed(F3) 在线 DINO 预处理不同，历史最佳训练 seed 分别为 1 和 3；跨 L1/F3 的非同 seed 差异不能解释为 DINO 单变量因果效应。
- 本轮核心假设：一个适中非零可学习 Gate 初值，在有可信目标追踪的前提下，可改变 Object Path 的梯度和实际利用程度，并在独立条件中改善闭环行为；该假设允许被证伪。

## 2. 阶段 A：审计已有 5k 扫描与语义证据（优先，低成本）

A0. 只读核查 r0 的 train configs、Gate 为 **可学习 alpha 的初始值**、优化器、模型输入、原型库哈希、数据顺序、seed、5k checkpoint 实际存在及完整恢复；缺失证据记录 BLOCKED，不编造。

A1. 从相同 init 30–49 中选定、锁定关键实例：+0.05 相比 ACT 的独有成功/独有失败，以及共同失败案例；从服务器取得实际 video、帧/状态/动作轨迹、DINO 相似度热图及 crop/overlay（如果已有），打包到 result。不要只输出服务器视频路径。

A2. 审查阶段：approach → contact → lift → transport → place，分别报告目标位置响应（必要时仅作执行者评议，明确非 GT）、机械臂/背景响应、接触后移动目标是否被追踪。Codex 可看图作初步判读，但不能将自己的视觉判断称为独立人工 ground truth。未取到视频/热图标记 `SEMANTIC_REVIEW_INCOMPLETE`。

A3. 补充**固定 alpha=0.05** 对照，仅在 r0 相同 5k 训练协议下从头训练；固定参数不参与优化器，明确这不是单纯修改初值的比较。与可学习 alpha_init=+0.05 使用相同初始非 Gate 权重、同 seed/数据流、相同测试环境。此阶段不探索更多 Gate 数值。

**A Gate**：若语义证据明显显示原型主要锁定背景/机器人，或者有关键 lineage 不合规，不直接进入最终模型；提出改进原型/表征的独立 decision。固定 Gate/可学习 Gate 结论限于机制。

## 3. 阶段 B：LIBERO 低成本候选复验

- Gate 集合冻结为 `[0, +0.01, +0.05, +0.1]`，主指标是 **可学习 Gate 初始化**，禁止随评估添加新数值。先用 Fixed P1，Legacy 仅对阶段 B 的前 1–2 个候选作补充并说明其预处理差异。
- 首先比较每个候选 **相同训练步数**：统一 5k；若 5k 尚不足以稳定学会任务，则事先报告并统一升到 10k，而不是只延长获胜配置。至少两个独立训练 seeds 用于候选稳定性检查，保持一个固定 seed 的逐 Gate 配对以隔离初始化影响。按 GPU 资源并行。
- **区分选择集与确认集**：r0 init 30–49 已被使用，只作为探索/诊断。先冻结未使用的 confirmation init IDs（建议 >=50 个配对 rollout/模型，若环境数量不支持则提交阻断及替代设计），将名单哈希封存，禁止基于确认集挑 Gate 或调原型。筛选时使用探索集与 Dev9（Dev9 不作原型选图）。先锁定前 1–2 个 Gate，才对这些 Gate 及 No-Object ACT 使用确认集做一次评估。
- 记录逐 init paired differences、成功率不确定性、不同 seed 方向一致性、训练稳定性、Dev9 Arm/Gripper MAE、Gate 曲线/梯度、Residual/Query、Object ON/OFF、推理延迟。对多次候选选择的偏倚明确说明；不以单次 2/20 优势定胜负。
- Stage B 结论可以是 `LIBERO_CANDIDATE`, `NO_RELIABLE_GAIN`, `INCONCLUSIVE`。即使 NO_RELIABLE_GAIN，也可继续 formal6 小预算**验证是否存在任务差异**，但不得宣传 Gate 已有收益。

## 4. 阶段 C：formal6 20k 初值筛选（最终 Gate 的关键依据）

**必须与真机历史架构、数据、原型对齐**：`QYyyyyyyy/formal6_merged` 120 episodes → Train90 / Dev15 / sealed Final15；Top/Wrist 摄像头、6D action、历史 8D state，冻结 4×384D 杯子原型及其精确哈希；L1 继承 Legacy 双重在线标准化，F3 继承 Fixed 单次在线标准化。禁止把 LIBERO 罐头原型用于真机杯子。

- 对 Legacy Seed1 与 Fixed Seed3 分别从头训练：`alpha_init=0`、`alpha_init=+0.05`，若 B 阶段的另一个 Gate 显示稳定优势则额外测试它（最多两种非零 Gate，且须在 C 启动前明确锁定）。可复用已有 20k 对照 checkpoint，**前提**其数据流、优化器、模型种子与原始 checkpoint 来源可审计，否则统一重训，禁止 20k 新模型直接和历史 80k 比。
- 每个架构内除 Gate 初值外不改权重初始化、学习率、模型、训练批次、预处理及数据顺序。每个新进程初始非 Gate 权重指纹应一致；若框架存在初始化随机顺序差异，显式恢复共同非 Gate 初值并留哈希证据。
- 在同一冻结 Dev15 上用 `chunk_valid` 定义 Arm/Gripper normalized MAE，另测不同操作阶段误差、Gate/Residual/Query、梯度与 ON/OFF；关注靠近、夹取、抬升、运输、放下等真实任务环节。Dev15 仅选择候选、Final15 保持封存。
- 不把 Dev MAE 改善等价于真机成功，亦不把更大的 gate 或 ON/OFF 当作优胜。若两个 Gate 20k 差异不明显，则优先保守地维持原版训练而非盲选。
- Legacy 与 Fixed 可以有不同的最终 Gate，**优先尝试共享值但不强制共享**，分别报告架构内 Gate 因果比较。对不同 seed 的 L1 vs F3 不作单变量声称。

## 5. 最终 Gate 选择标准和停止规则

选择分三道门：

1. **工程/机制合法性**：数据与 protos 仅来自 Train，训练真实达标、checkpoint 真实可复现；无数值爆炸；Gate 对梯度和实际融合有可测、稳定的作用。
2. **方法价值**：在与从头训练 No-Object 的公平对照下，LIBERO 有独立确认集上的正向证据；或 formal6 的稳定 Dev / 阶段指标及潜在真机受监督先导表明有价值。单靠较高 Object ON/OFF 不足以过门。
3. **稳健性**：多个 seed 或不同难度/阶段的收益方向基本一致，不以选中单个胜出种子的结果作为可靠收益。若成本有限，把不确定性明确标记为 `PROVISIONAL`，不能声称理论最优。

最终输出每个架构各自的候选：`L1_selected_gate`、`F3_selected_gate`，允许两者相同或不同，甚至都选择 `0`；如证据不足选 `NO_GATE_CHANGE`。并给出选择使用的指标和未使用的确认集说明。

## 6. 后续目标：最终 80k 训练与真机验证（本 decision 不授权自动启动）

- Gate 冻结后，另起经用户批准的 final training decision：L1 Seed1 80k 和 F3 Seed3 80k，**按选定的 Gate 初值从头训练**，原始 L1/F3 80k 保留并复用。每组单变量配对、checkpoint/优化器完整保存、Dev15 按冻结规则评估、Final15 仅最终一次性报告（防止泄漏）。
- 经本地 RTX4060 8GB 部署 Smoke、policy 加载/维度检查、图像标准化、动作边界和夹爪校验后，进行**有人现场监督**的真机 Shadow→短动作 Smoke→正式对照；无杯场景不先执行机械臂动作。需要明确人工 E-stop 及异常停止标准。禁止无人值守真机运行。
- 如 Gate-Selected 未胜出原版，最终论文主线应保留原版 L1/F3，不强行命名新模型为最优。

## 7. 资源调度与 Codex 分工

- 只有一个本地 Codex，远程 SSH 管理 RTX5090；不得覆盖运行中的任务或 repo 历史成果，使用 `tmux`/持久化作业、独立日志目录与停止条件。
- 首先复核 A，再由符合门槛的任务进入 B、C；大规模并行度用有效训练步/秒与推理资源实测决定，不硬上四并发。允许时间足够时并行分析与训练，不能混淆阶段门槛。
- Codex 不可自行修改 Gate 搜索集合、选择标准、任务范围、训练终点或确认集。如阶段 A/B 出现阻断或证据相互矛盾，生成 `blocking_proposal.md`，保留已完成结果停止推进。
- Builder 检查训练/恢复/数值/图像 hash，Red Team 查数据泄漏、配对、seed、指标和不确定性；`PENDING` 不得写为 PASS。

## 8. 交付物

建议位置：`experiments/p4_public_benchmark/p4_10_gate_sweep/index2/decision.md`，旁边 `result/`。至少提供：

- `status.json`, `execution_manifest.json`, `lineage_audit.md`
- `semantic_review/`（关键 rollouts 视频、热图、带标记关键帧、审查表）
- `fixed_vs_learnable_gate.csv`, `libero_seed_gate_matrix.csv`, `libero_confirmation_pairs.csv`
- `formal6_20k_comparison.csv`, `gate_dynamics.csv`, `object_path_diagnostics.csv`
- `gate_selection_report.md`（L1/F3 各自推荐 Gate、证据、信心、不确定性、是否建议 80k）
- `builder_review.md`, `red_team_review.md`, `blocking_proposal.md`（若需要）
- 汇总 `summary.md`：要明确 **机制激活≠目标追踪正确≠闭环收益**。

## 9. 禁止事项

不启用真实机器人、不自动训练最终 80k、不重新标记原型、不修改 P4.7 历史模型、不重复使用确认集做调参、不使用 Final15 做 Gate 选择、不因 Gate 更大而宣布更优、不以 5k/20 rollout 结果宣传 SOTA。
