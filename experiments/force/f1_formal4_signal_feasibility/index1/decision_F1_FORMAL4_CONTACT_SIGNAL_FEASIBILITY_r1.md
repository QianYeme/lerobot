# Experiment Decision Contract

> Revision: r1 — 2026-10-15  
> 用户与 ChatGPT 根据已讨论的 F1 方向冻结；执行 Codex 对本文件只读。  
> 替代尚未产生科研结果的 r0；若仓库中 r0 已运行并查看科研指标，不得追认预注册，须保留旧轮并提交冲突说明，停止正式比较。

## 1. Identity & Mode

- Experiment ID: `F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY`
- Track: `FORCE`
- Stage: `FORMAL4_SIGNAL_FEASIBILITY`
- Index: `1`
- Mode: `DIAGNOSIS`
- Dataset: `formal4/kind_merged_nomaster`，实际根目录由执行端核验。
- Code State: `QianYeme/lerobot` 的 `det` 分支；开始时记录实际 HEAD、dirty diff、依赖版本，不沿用历史 SHA。
- Previous Experiment: D2 / D2.1 动作链诊断及其补证；历史 Formal3/F0 只作背景，不转用其结论、标签或阈值。
- Contract Path: `experiments/force/f1_formal4_signal_feasibility/index1/decision.md`
- Raw Evidence: `results/experiments/F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY/`

### Current Authorization

只授权读取冻结 Train36 / Dev12、原始信号审计、独立标签审核、因果历史特征、固定逻辑回归 probe 拟合、离线评估与文档更新。

不授权机器人 policy 训练、P1 或其他模型结构修改、Force head、控制器、自动收紧、部署参数修改、新采数据、真机、Final Test、push/publish，亦不授权自动进入 F2。

先读项目 `AGENTS.md`、`workflow/lerobot_research_workflow.md`、`workflow/STATUS.md`、本合同及必要历史证据。附件 STATUS 可能落后于当前对话；冲突逐项记录，不把历史 B3_ACT 名称当作架构事实。

## 2. Research Question

> **Primary RQ:** Formal4 中，夹爪 load 相比夹爪位置和已知运动/命令历史，是否提供跨 episode 的接触相关或任务阶段增量信息，值得后续设计最小 temporal auxiliary / conditioning 实验？

current 的独立与条件增量为固定次要问题，不得替代主问题。一次只解决信号可行性，不解决抓杯成功率。

## 3. Paper Relevance

### Candidate Claim

在明确独立标签、固定历史窗口和固定线性 probe 下，执行器读数包含超出位置/命令基线的离线阶段信息；若接触证据与混杂检查均通过，可进一步限定为接触相关辅助线索。

### Why It Matters

避免在信号价值未明时堆叠 Force 模块。P1 原本已经接收 load/current，本轮不证明 P1 已利用这些输入，也不是首次加入力信号的实验。

### Non-Claims

- load/current 不是已标定握力、接触力或牛顿值。
- probe 有增量不等于 policy 有收益；无增量不等于任何非线性模型都无法利用该信号。
- 不证明防滑、抓稳检测、安全力控、低抬升修复、长保持或真机成功率改善。
- 不验证透明物体专用识别，不建立纯 ACT 对照。

## 4. Baseline / Intervention

| Probe | 输入 | 比较角色 |
| --- | --- | --- |
| S0 | gripper.pos 及其历史变化、过去已发送/已知夹爪命令及变化 | 基线 |
| S1 | S0 + load | 主干预：S1−S0 |
| S_current | S0 + current | current 单独增量，次要 |
| S2 | S0 + load + current | current 在 load 之外的增量：S2−S1，次要 |

**Only Intended Scientific Difference:** 相同标签、样本、窗口、特征模板、标准化方法与 probe 下，是否增加指定执行器信号。不得为不同 probe 选择不同有效帧。

字段缺失或 current 退化为常量须如实报告；可继续 load 主比较，current 比较标 `NOT_EVALUABLE`，不得编造替代值。load 或 S0 必需字段不可用则主任务阻塞。

## 5. Frozen Contract

### 5.1 Data / provenance

- 严格沿用既有 episode manifest：Train36、Dev12、Final12；禁止重新随机划分。
- 核验 episode ID、互斥性、数据版本、schema、顺序、帧数、FPS、时间戳和视频对齐，保存 manifest 及哈希。
- 不读取 Final 的视频、数据行、分布或统计；只可读取既有 split manifest 中的 ID 以验证互斥。
- 只读原始数据，标签及派生特征另存。不按任务失败删除 episode。
- 按 episode 构造窗口，不跨集；不足完整窗口的开头帧不填充，并单列计数。

### 5.2 Signal audit / causal timing

实际核对 `gripper.pos`、`gripper.load`、`gripper.current` / `gripper.curr` 的字段、单位、符号、编码、裁剪/饱和来源、归一化和采集代码。

对 Train/Dev 分别报告 range、mean/std、分位数、零值率、唯一值/量化、缺失/非有限值、漂移、已知上下限命中率。未知硬件上下限写 UNKNOWN，不把样本 min/max 直接当作饱和阈值。

以观测 t 到达、尚未生成本次动作的时刻为特征截止点。读取源码确认同一行 action 是 leader target、已发送命令还是其他语义；通常使用 t−1 及更早命令，只有证明 t 行命令在截止前已知才可使用。记录时间对齐表及证据；未知时不得默认实时可得。不得使用未来动作或事后实际执行结果。

不以 Dev 搜索最佳滞后。发现延迟不明但仍能定义可靠的保守因果输入时记录局限；不能定义时 BLOCKED。

### 5.3 Label source / review

标签依据双视角 RGB 为主，位置与已知命令只用于定位候选、支持动作阶段解释。禁止向标注过程展示 load/current 曲线或 probe 分数。

事后审核可连续观看前后视频以判断事件；这不授权预测特征读取未来。标签必须记录来源、原始帧号、事件边界/不确定区间、可见性、审核者与证据图/片段。未知边界附近帧排除而非硬填。

独立定义：

- `PRECONTACT`: 可见夹爪与杯子尚未接触；包括可见的接触前闭爪过程。
- `CONTACT_SUPPORTED`: 视频支持接触/有效夹持，例如可见接触或杯子随后与夹爪共同运动；不得仅据“夹爪已闭合”判断。
- `CLOSING`: 闭爪动作阶段，不等于接触。
- `POST_CLOSE`: 闭爪结束至释放前的任务阶段，不等于抓稳。
- `RELEASE`: 张爪释放阶段，不等于已确认杯子稳定落桌。
- `UNKNOWN/OCCLUDED`: 无法确定；分开统计未知原因。

不具备滑落/失败/不稳定负例时，禁止构造 stable-vs-unstable 结论。

### 5.4 固定主任务与降级规则

禁止按 probe 精度选择任务。依下列顺序自动执行，冻结规则本身已由本合同授权：

1. 按冻结 Train manifest 顺序先审前 6 集的事件窗口，连续播放并逐帧核查，不以每秒一帧代替。
2. 若至少 4/6 集同时存在 PRECONTACT 和 CONTACT_SUPPORTED，各至少 5 个可靠目标帧，继续审核全部 Train36 的接触任务；否则直接转阶段任务，不重复粗抽帧。
3. 接触任务只有在 Train 至少 24/36 集两类齐全、每类每集至少 5 个有效目标帧时，才冻结为主任务：二分类 PRECONTACT vs CONTACT_SUPPORTED。
4. 若 Train 接触覆盖未达标，主任务固定降级为三分类 CLOSING / POST_CLOSE / RELEASE；需 Train 至少 24/36 集三类齐全、每类每集至少 5 个有效目标帧。
5. 两种任务均不足则主结论 UNRESOLVED，不为凑结果训练主 probe；交付最小人工审核片段和阻塞说明。
6. 主任务、标签规则及 Train 标签版本冻结并哈希后，再审核 Dev；Dev 不改变任务。Dev 需至少 8/12 集包含主任务全部类别、各至少 5 个有效目标帧，否则主要推断 UNRESOLVED。

以上覆盖计数按主窗口可用帧计算。阈值是本轮诊断的操作标准，不是统计功效保证。人工审核困难时先交具体小片段；不可提交全 UNKNOWN 表后声称已完成视觉精审。

### 5.5 Observation / features

- 固定历史窗口 W = 1 / 9 / 15 帧，主窗口 15；依据真实 FPS 报告时长，约 30FPS 时分别约 0.033 / 0.30 / 0.50 秒。
- 每个原始信号固定特征：当前可用值、窗口 mean、std、末减首；不临时添加 min/max 或其他有利特征。
- 命令和位置变化先因果差分，再按相同模板汇总。W=1 的窗口 std/末减首置 0，差分仍须有合法历史。
- 所有 probe 取共同有效样本；跨窗口比较另报告共同帧敏感性分析，避免把样本变化当窗口收益。
- 不使用 episode ID、绝对时间、帧号、任务进度比例或视觉特征作为 probe 输入。它们只作索引和审核元数据。
- scaler 只拟合 Train；禁止逐 episode 归一化及 Train+Dev 统计。非有限输入排除规则须统一、记录原因及数量。

### 5.6 Model / training

所有 probe 固定 L2 Logistic Regression，C=1、class_weight=balanced、seed=20261014；solver=lbfgs、max_iter=2000、tol=1e−4。二分类阈值 0.5，多分类 argmax。类别顺序写入配置。

只授权诊断 probe 拟合；不授权 robot policy 训练。禁止树模型、神经网络、非线性核、超参搜索、Dev 调参。若不收敛，只允许检查输入/数值错误，不得更换模型或调 C；仍不收敛则报告阻塞。

### 5.7 Evaluation / statistics

每个可评估 episode 单独计算 Balanced Accuracy（各类 recall 平均）、Macro F1、各类 recall、confusion matrix；主汇总对满足全部类别条件的 episode 等权。其余 episode 仍交付计数、预测、矩阵和排除原因，不从证据中删除。

**Primary metric:** W=15 下，各 Dev episode 的 BA(S1)−BA(S0) 的等权均值。

固定实际意义阈值 δ=0.03，即 3 个百分点。使用配对 episode bootstrap，2000 次、seed=20261014，重采同一批 episode ID，所有 probe 共用重采样索引；报告原始点估计、bootstrap 均值及 percentile 95% CI。不做 frame bootstrap。

次要固定报告：S_current−S0、S2−S1，全部窗口结果，以及每集 delta；次要结果不替换 load 主终点，不作为正式多重检验确认性证据。

### 5.8 Contact confounding check

接触主任务若显示正增量，必须补充位置/运动匹配敏感性检查：仅使用 Train 的 S0 特征拟合固定 L2 逻辑回归作为匹配分数；对每个 Dev episode 内两类帧按预测分数最近邻、无放回配对，最大分数差 0.05，按原帧号排序确定并列。匹配只用 S0 和标签，不用 load/current。

每集至少 5 对、至少 8 集时，沿用原 probe 评估匹配子集的 BA 差及 episode CI；不重新训练。支持 contact 需此差点估计>0 且 CI 下界>0。覆盖不足或结果不支持时，不能归 CONTACT_AUX_SUPPORTED，至多说明接触相关标签可预测但混杂未排除。

匹配只能削弱可观测位置/运动混杂，不能证明因果接触感知。S0 线性模型有限，不得声称排除了所有运动解释。

### 5.9 Deployment / engineering freedom

部署不变，不运行机器人策略。允许工程缓存、批处理、只读脚本、可视化和保持科研语义的 bug 修复；记录脚本、CLI、依赖、日志、失败尝试与哈希。禁止修改 decision 或覆盖旧结果。

## 6. Assumptions / Open Questions

- `[FROZEN]` P1 已输入 load/current，本轮只检验可解码增量，不检验原 policy 的实际利用。
- `[FROZEN]` B3 历史命名误导，已有核验支持 OC-ACT，不能作为纯 ACT baseline。
- `[FROZEN]` D2/D2.1 未完全闭环，G1 正式真机评价不完整；不作为 F1 必须全部完成的前置条件。
- `[PROVISIONAL]` 历史短窗可能比单帧更有用，必须在 Formal4 重新验证。
- `[OPEN]` contact 可见性、信号完整性及 current 独立价值待本轮检查。
- `[REJECTED]` 闭爪=接触、阶段=抓稳、电流=物理握力、probe 有效=真机有效。

## 7. Hypothesis & Falsification

### H1

W=15 下，load 在 S0 之外提供至少 δ=0.03 的跨 episode 标签预测增量。

### Expected Mechanism

接触或阶段变化伴随可重复执行器读数变化；短历史描述读数动态；相同输入规则下 S1 比 S0 更能区分独立标签。

### Reject / do not support H1 if

- 有效主比较 95% CI 上界 < δ：排除本轮定义的最低有意义 load 增量，H1 不支持。
- CI 跨越 δ 或跨零，未满足正向标准：结论不确定，不将“不显著”写成无信号。
- 标签依赖 load/current、时间泄漏或分组污染：实验 INVALID，而不是科研反驳。
- 信号/覆盖不可评估：UNRESOLVED，不等于 FAIL。

### Forbidden Post-hoc Rescue

禁止换窗口/任务/阈值、删除失败集、挑最好 seed、更换模型、引入新特征、增预算、打开 Final 或新增 Force 模块来挽救结果。新科研语义只能 proposal，不能执行。

## 8. Gates & Metrics

### Mandatory Gates

| Gate | 本轮要求 |
| --- | --- |
| G0 Provenance / Contract | 当前代码、split、数据、合同、历史证据来源完整；确认 r0 未产生本轮科研结果 |
| G1 Data & Schema | 字段与时间语义、Train/Dev 互斥、信号完整性、独立标签和覆盖 |
| G2 Implementation | 因果窗口、命令对齐、Train-only scaler、共同样本、配对 bootstrap 的定向验证 |
| G3 Training Health | 仅 probe 收敛/有限输出；policy training=NA |
| G4 Mechanism | 仅检验信号在 probe 中的增量；不冒充 P1 policy 机制通过 |
| G5 Offline Behavior | 独立标签预测与逐集稳定性；不等于机器人行为或成功率 |
| G6 Deployment | 仅核验特征因果可用性；完整部署等价=NA |
| G7 Real Robot | NA |
| G8 Scientific Conclusion | 依下列规则输出，限定为诊断证据，最高 SUPPORTED |

### Scientific decision rules

定义某比较“正向支持”：点估计 ≥δ、95% CI 下界>0、至少 2/3 有效 Dev episodes 的 delta>0。

- **CONTACT_AUX_SUPPORTED**：接触主任务，load 主比较正向支持，标签独立及匹配检查通过。仅支持接触相关辅助信息。
- **PHASE_ONLY_SUPPORTED**：阶段主任务下 load 主比较正向支持。current-only 满足同标准时可报告此类别，但必须加 `LOAD_PRIMARY_SUPPORTED=false`、`SUPPORT_SOURCE=CURRENT_SECONDARY`，不得称 H1 通过。
- **NO_INCREMENTAL_SIGNAL**：可靠主任务及覆盖均通过，S1−S0、S_current−S0、S2−S1 的 95% CI 上界均 <δ。仅排除本合同 probe/标签/特征下的最低有意义增量；不是信号普遍无效证明。
- **UNRESOLVED**：其余情况，包括覆盖不足、CI 不精确、接触增量但混杂未排除。单列 load/current 每项状态，不隐藏次要正向信号。

接触主任务不可因匹配失败而事后另换阶段主任务；可以在 proposal 中建议下一轮阶段诊断。

### Overall acceptance

- `PASS`: 所有适用工程/审计项完整、交付可复核，按规则给出科学结论。结果为 C 或统计不确定的 D 仍可诊断 PASS，不能写 Force 有效。
- `PARTIAL_PASS`: 有可用证据，但授权内可完成项或关键交付仍缺失。
- `BLOCKED`: 数据/可见性/环境等使必要步骤不能执行；写已尝试方法和最小缺口。
- `INVALID`: 泄漏、循环标签、事后改科学语义等使比较失效。
- `FAIL`: 必要实现检查未通过且未修复；与 H1 不支持分开记录。

`F1_SIGNAL_FEASIBILITY_COMPLETE=true` 仅当授权范围的必要步骤或预设降级/停止分支已完整执行、证据和文档齐全；不表示科学确定。否则 false。

## 9. Authorization

### Codex MAY

实现只读诊断与固定 probes；按本合同的确定性规则降级标签任务；修复工程错误、恢复任务、缓存数据、写 proposal。

### Codex MUST

保护用户已有修改，开始/结束检查 workspace；不使用 `git add .`。保存原始证据和失败尝试。同步更新 `workflow/STATUS.md` 与 `docc/日志/工作日志.md`，记录本轮授权、结果、科学边界、下一步仍待决定；保留 B3 身份更正、D2/D2.1 未闭环和 G1 真机不完整事实。完成后停止。

### Codex MUST NOT

修改 decision、split、主任务选择规则、特征、probe、主指标或判定阈值；训练 policy；改架构/Force head/部署；自动收紧；真机/新采；读取 Final；自动进入 F2/Confirmation；push/publish；覆盖无关工作。

### Requires User Approval

新研究结构、policy 训练、下一科研轮、正式确认、Final Test、正式真机、重要采集协议变化和 push/publish。本合同内普通编程、审核及预定义分支不需反复询问。

## E. Force / Control Section

- Signals: load/current 为执行器读数，单位与符号以源码核验为准。
- Contact / Grasp / Hold / Release: 按第 5.3 节区分；POST_CLOSE 不是稳定保持真值。
- Controller Ownership: 不新增控制器，不接管任何关节。
- Fallback: 按接触→阶段→UNRESOLVED 的冻结规则。
- Threshold Source: δ 和覆盖/匹配阈值来自本合同操作定义，不是硬件力阈值。
- Calibration Evidence: 未独立标定则记录 NOT_CALIBRATED，不输出牛顿值或握力估计。

## Required Deliverables

紧凑交付放在合同同目录 `result/`：

```text
summary.md
validation.json
signal_integrity.json
signal_timing_audit.md
label_manifest.json
label_coverage.json
task_freeze.json
leakage_audit.json
probe_config.json
metrics.json
bootstrap.json
per_episode_metrics.csv
confusion_matrices.json
contact_matching.json
figures/
evidence_samples/
proposal.md
```

不适用文件写 NA 及理由，不伪造数值。图至少包含信号分布、窗口比较、各 probe 比较、逐集 delta 与 bootstrap CI。提供能复核标签的少量双视角证据及精确片段索引，不能只提供 unknown 表。

Raw evidence 在 `results/experiments/F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY/` 按 run ID 保留：标签逐帧表、原帧索引、特征与预测、scaler/probe、脚本版本、CLI、日志、hash manifest、审核证据和失败尝试。紧凑交付须能定位 raw，关键小证据随包，不仅引用执行机绝对路径。

汇总表必须有 Task / Window / Probe / BA / Macro F1 / Per-class Recall / N episodes / N windows；单列 15-frame 三项增量、CI、逐集正向比例、排除计数及 UNKNOWN 偏差局限。

summary / validation 最后记录：

```text
OVERALL_STATUS = PASS | PARTIAL_PASS | FAIL | BLOCKED | INVALID
F1_SIGNAL_FEASIBILITY_COMPLETE = true | false
SCIENTIFIC_OUTCOME = CONTACT_AUX_SUPPORTED | PHASE_ONLY_SUPPORTED | NO_INCREMENTAL_SIGNAL | UNRESOLVED
LOAD_PRIMARY_SUPPORTED = true | false
SUPPORT_SOURCE = LOAD_PRIMARY | CURRENT_SECONDARY | NONE
NEXT_POLICY_TRAINING_AUTHORIZED = false
```

提出建议只写 proposal，下一轮由用户与 ChatGPT 讨论，不生成下一轮 decision，不自动执行。
