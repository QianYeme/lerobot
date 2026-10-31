# decision.md — P4.9-r1 统一 Gate 初始化实验（LIBERO + formal6，10 小时无人值守）

- Decision ID: `P4_9_r1_UNIFIED_LIBERO_FORMAL6_GATE_INIT_FOUR_MODELS_10H`
- 状态：`APPROVED_FOR_CODEX_EXECUTION`（待交本地 Codex 执行；不是已执行）
- 日期：2026-10-31
- 取代：**尚未执行的 P4.8-r7** 与 **尚未执行的 P4.9-r0** 的训练/评估任务，不更改原文件和历史证据。
- 执行：**唯一的本地 Codex**，通过 SSH 管理 RTX 5090 服务器，使用持久化作业。没有第二个 Codex。
- 建议位置：`experiments/p4_final_p1_training/p4_9_unified_gate_init/index1/decision.md`

## 0. 科研问题与前提

冻结一个唯一因变量：P1 Object Cross-Attention 的**可学习标量 Gate α 的初始值由 0.0 改为 +0.001**。并行检验该改变在两种环境（LIBERO 单任务、SO-101 formal6 数据）与两种 DINO 在线预处理（Legacy 双重标准化、Fixed 单次标准化）中的训练与表现影响。**不修复 Gate 的结构、不强行固定 Gate、不调整残差比例、不改其他超参数**。

LIBERO r6：10k 时 Gate 接近零且 Object ON/OFF 极小，这是 LIBERO 的机制证据。formal6 F3 80k 的 Gate ≈0.02476、ON/OFF≈0.1197，**没有证明 formal6 同样塌缩**；新实验是初始化敏感性研究，禁止称为“已确认 bug 修复”。

`formal6_merged` **120 个 episode**（冻结 Train90 / Dev15 / Final15），不是 120k 训练步数。formal6 新模型各 80,000 effective optimizer steps；LIBERO 新模型各 10,000 steps。

## 1. 四个新训练作业与对应历史对照

| 作业 ID | 任务 | 预处理 | Seed | Gate 初值 | 训练步数 | 完全匹配的历史对照 |
|---|---|---|---:|---:|---:|---|
| `LB_LEGACY_GATEPLUS` | LIBERO-Object 已验证单任务 | Legacy double-norm | 继承 r5 Legacy seed | +0.001 | 10k | r5 Legacy P1 Gate=0 同 seed、同配置 10k |
| `LB_FIXED_GATEPLUS` | 同一 LIBERO 任务 | Fixed single-norm | 继承 r5 Fixed seed | +0.001 | 10k | r5 Fixed P1 Gate=0 同 seed、同配置 10k |
| `F6_LEGACY_GATEPLUS_S1` | formal6 左抓右放 | Legacy double-norm | **1** | +0.001 | **80k** | **L1**：Legacy P1 Seed1 80k Gate=0 |
| `F6_FIXED_GATEPLUS_S3` | formal6 左抓右放 | Fixed single-norm | **3** | +0.001 | **80k** | **F3**：Fixed P1 Seed3 80k Gate=0 |

**为何 formal6 不使用同一个 seed：** L1（Legacy seed1）和 F3（Fixed seed3）分别是其历史版本中有真机依据的较强候选。这样可以**分别**严谨评估 L1 vs L1 Gate+、F3 vs F3 Gate+；但绝不能把新 Legacy seed1 和新 Fixed seed3 的差别归因于预处理，它们 seed 不同。若未来要求 Legacy/Fixed 的严格单变量对照，须另设同 seed 实验。

LIBERO 三模型原对照：Vanilla ACT 10k、Legacy P1 Gate0 10k、Fixed P1 Gate0 10k，全部复用 r5，不重新训练。formal6 原 F3/L1 同样保留。SO-101 的 A1/Vanilla ACT 只作为背景参照，不纳入 Gate 单变量胜负。

## 2. 模型、数据和身份硬门槛

1. 先读取上游 r5/r6/r7 及 formal6 P4.1/P4.2.1 的**真实**脚本、checkpoint manifest、`effective_config`、训练配置、数据划分、prototype hash、训练 seed、状态/动作规范、归一化、EMA/temporal ensembling/推理入口；不得按本文猜测补齐未知字段。
2. formal6 数据修订号历史记录 `83a829af6df245ed53596dbc22b18c19f8bd3315` 需服务器实际核验；Train90/Dev15/Final15 按冻结 episode ID 使用；不把 Dev15/Final15 用来生成原型或训练。
3. formal6 **Legacy** 与 L1 相同预处理；**Fixed** 与 F3 相同 `top_raw` 在线单次标准化，不能将 Fixed checkpoint 放入 Legacy 推理路径；原始 Top/Wrist CNN 路径必须不变；8D state / 6D action / 100-step chunk；F3/L1 各自历史 Prototype Bank 与 train-only stats 均须一一匹配（不要跨任务共用）。
4. LIBERO 复用 r5 的 **同一 Train35、Dev9、四个 Codex 辅助参考目标位置建成的 Train-only 4×384 bank**、已通过评审的控制/采样接口；Legacy/Fixed 两组共享相同 bank；不得重标注、重建、使用评估数据、偷换 LIBERO 控制频率。
5. 只有 Gate 初值不同；逐项生成机器可读 `diff` 与人工可读审核。验证 gate=+0.001 可训练，其他可训练张量初始化在可复原条件下字节一致，seed、采样顺序、优化器、batch、clip、scheduler、AMP、normalizer、相机处理完全一致。若旧 step0 无法精确重建，标注 `RECIPE_MATCHED_NOT_BYTE_MATCHED` 并降低因果结论强度。
6. 必须从头训练，不得载入旧 80k/10k 模型后仅重置 Gate；smoke steps 不计入正式 effective steps。控制基线不被覆盖。出现比较性阻断时只阻断该配对作业、交付 PROPOSAL，不得自行更改实验定义。

## 3. 唯一 Codex 的服务器并行与无人值守调度

**先 inventory**：查询正在运行的作业（尤其可能已有的 P4.8-r7/P4.9 任务）、GPU 实际利用率、显存、CPU、IO、剩余磁盘、预计可用墙钟。若某任务已启动且配置相同，**接管其作业 ID 而不重复启动**；若 r7/r0 已运行，保留运行日志并提交冲突方案，不能盲目执行覆盖。

**并行基准**：分别用短 smoke 在正式数据/模型上测 1/2/3/4 进程的聚合有效 updates/s、单位 job 时长和显存、CPU IO，确保测试采用临时目录、训练计数隔离、不污染正式步骤。仅当并行带来 >=10% 聚合有效吞吐增益且无明显 OOM/严重抖动，才提高并发；采用分组动态补位、进程隔离、独立数据缓存与日志。训练应使用系统可用持久化执行机制（例如 tmux + 作业 runner），断开 SSH 仍完整执行依赖队列和评估。

**调度优先级：**
- A. 开始前完成预检、Gate 初始梯度 smoke、历史匹配基线可加载。
- B. 两个 **formal6 80k** 是最长、回本地真机最有价值的作业，**先确保均能在可用时间内完成**，优先并行运行 L1 Gate+ / F3 Gate+；如果并行不划算，根据实测用最优吞吐方式排队。不能因 LIBERO 短作业抢占导致两个 formal6 都不完整。
- C. 利用测得的可用吞吐槽位安排两个 LIBERO 10k 配对作业；它们可以先后执行以避免 IO 和 GPU 互抢；formal6 暂不可启动时也不要让服务器空转，可先跑 LIBERO。
- D. 完成一个作业后立即释放槽位，按依赖运行它的轻量评估，继续填补另一个可行作业。实验必须保持每个配对组同等训练预算。
- E. 计划运行窗口约 10 小时。**最后 45–75 分钟优先收尾**：Checkpoint 完整性验证、Dev15、Dev9、LIBERO 配对闭环、诊断与结果打包。不要为了训练 GPU 占用率牺牲评估。若预计 10 小时无法完整完成四个模型，自动执行最高可完成配对组、保留 full-resume checkpoint 标 `PARTIAL`，明确列出未完成工作；**不允许悄悄改变 80k/10k 预算或者将 60k 标成 80k。**

## 4. 检查点与恢复

- formal6 每组保存完整 optimizer/scheduler/RNG/normalization/prototypes 来源 checkpoint，建议 steps 10k/20k/40k/60k/80k；LIBERO steps 1k/5k/10k。保存大小、SHA256 和 exact step。checkpoint 恢复需在早期验证权重、optimizer 与采样 RNG 可恢复；best-effort 非确定性因素如 cuDNN 必须披露。
- 分离四个 run 的目录，独立 PID、timestamp、config hash、source SHA、数据 revision；支持断线继续，避免 checkpoint 并发覆盖或进程重复。
- 无人值守只许可**服务器训练、静态评估、仿真**，不允许本地或远程真实机械臂动作。

## 5. 统一诊断与统计

### 5.1 formal6 — 真机模型离线筛选
- 分别在**相同 Dev15 观测采样及程序**下对 L1 Gate0 vs L1 Gate+（seed1）、F3 Gate0 vs F3 Gate+（seed3）做配对评估；arm/gripper normalized/raw MAE、逐 episode 统计、动作分布/异常、Gate 轨迹、Object Residual/Query 比、Object ON/OFF 动作差异、参考 prototype 响应。
- 如果历史 80k checkpoint 的评估器不一致，必须**两边重新运行同一评估程序**。避免在有意无意改变评价数据/数据标准化前提下比较历史 CSV 数字。
- 不能将 L1 Gate+ vs F3 Gate+ 的差异归因于 DINO 修复（seed 不同）。Final15 仍封存；不根据 Final15、临时真机反馈选择 checkpoint。
- 输出完整 `robot_candidate_handoff.md`，含部署需用的 Legacy vs Fixed 不同预处理契约、当前校准/单位复核、hash-pinned checkpoint 以及与历史 L1/F3 的可比性。

### 5.2 LIBERO — 模拟闭环
- 同一训练数据、同一 r5/r6 原型、同一控制协议；对 ACT Gate N/A、Legacy Gate0、Legacy Gate+、Fixed Gate0、Fixed Gate+ 统一 Dev9 与 Object ON/OFF，记录 Gate 轨迹、初始梯度、残差比例与响应热图。
- 复测 r5 用过的 init 10–19，仅作为探索性一致性检查；**另预先冻结合法未被 r5/r6 使用过的 init 列表**，对所有五个模型实施相同 init 和评价标准，输出逐 init 结果和成功率（样本有限不得夸大）。保留控制频率、10FPS 数据与 20Hz 仿真接口的真实溯源，不能擅自修补时间基准。
- ON/OFF 与强制 Gate 剂量响应只属于模型机制诊断，不能冒充原始 checkpoint 的真实策略表现。

## 6. 真人机追加测试：未来计划，不属于无人值守授权

当用户回来、查看离线报告且在场监督后，分别进行 **L1 Gate0 vs L1 Gate+** 和 **F3 Gate0 vs F3 Gate+** 的受控真机比较。建议分别冻结相同 3 source 位 × 2 光照 × 2 水量 × 2 repeat × 2 model = **每对 48 次**，两对总 **96 次**，采用交替/配对顺序；先少量安全 Smoke，确认 SO-101 标定、关节角单位、夹爪范围、两个相机、模型推理类型、实际执行频率、Home、负载传感器。无杯仅 Shadow。不将 assisted recovery 计为自主成功。

主要指标是完整搬运成功率、首个失败阶段、稳定悬停、夹爪滑移、目标区放置、洒水与空抓风险；不得仅凭离线 MAE 或 Object ON/OFF 放大声称提升。任何真机部署必须另经现场安全确认；本 decision **不授权无人值守真机运行**。

## 7. Builder / Red-Team 审查、产物

结果目录：

```
experiments/p4_final_p1_training/p4_9_unified_gate_init/index1/
  decision.md
  result/
    job_registry.json
    preflight_manifest.json
    architecture_diff_and_init_contract.md
    throughput_benchmark.csv
    checkpoint_manifest.csv
    formal6_train_metrics.csv
    formal6_dev15_paired.csv
    formal6_object_path.csv
    libero_train_metrics.csv
    libero_dev9_paired.csv
    libero_rollouts_per_init.csv
    libero_object_path.csv
    model_provenance.json
    robot_candidate_handoff.md
    builder_review.md
    red_team_review.md
    execution_complete.json
    unresolved_proposals.md
```

- Builder：查验四组配方、Gate 唯一差异、checkpoints 实际存在、训练计数、完整恢复、输出检查、不同预处理推理路径、模拟器闭环复现；独立审查未完成须如实标记 `PENDING`。
- Red-Team：核查 Legacy/Fixed 非同 seed 的混淆、r5 init 的重复使用、人工目标先验声明、训练/评估泄漏、DINO 预处理一致性、GPU 并行吞吐及 CPU IO 干扰、SO-101 单位/校准风险。
- 对每个配对报告 `GATE_INIT_BENEFIT_SUPPORTED` / `GATE_ACTIVATED_NO_CLOSED_LOOP_GAIN` / `GATE_INIT_HARMFUL` / `INCONCLUSIVE`，**formal6 未做真机时只能称离线结果**；不能用该标签提前宣称真机优势。
- 禁止提交 Hugging Face token、SSH 凭据、隐私内容；禁止擅自上传模型。不要捏造服务器训练已开始/完成、可部署结果、评审已通过。

## 8. 交给 Codex 的执行边界

1. 先核对 P4.8-r7 与 P4.9-r0 **确实尚未执行**及服务器当前作业清单；本 r1 替代它们的待执行训练工作，而不是把它们重复跑两遍。
2. 找到配对性或资源的阻断时，输出含具体证据的 PROPOSAL；允许先运行不受阻断影响的配对组，不得修改冻结的科研变量。
3. 不要以训练成功率、Gate 变大或单任务高分为由自动更改下一轮论文结论或开启扩展任务；完成结果包等待 ChatGPT 审查。
