# Experiment Decision Contract

> Revision: r0 — 2026-10-15  
> 用户与 ChatGPT 按已讨论的小规模诊断方向冻结；执行 Codex 对本文件只读。  
> 本轮为新的探索诊断，不追认原 F1 完成，不替代正式确认。

## 1. Identity & Mode

- Experiment ID: `F1P_CONTACT_SIGNAL_PILOT`
- Track: `FORCE`
- Stage: `CONTACT_SIGNAL_PILOT`
- Index: `1`
- Mode: `DIAGNOSIS`，所有科学结果标记 `EXPLORATORY_ONLY`。
- Dataset: `formal4/kind_merged_nomaster` 的既有 Train36 子集。
- Code State: `QianYeme/lerobot`，`det` 分支；执行时记录真实 HEAD、dirty diff、依赖版本。
- Previous Experiment: `F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY` r1，当前 PARTIAL_PASS、科学 UNRESOLVED。
- Contract: `experiments/force/f1p_contact_signal_pilot/index1/decision.md`
- Compact Results: 合同同目录 `result/`。
- Raw Evidence: `results/experiments/F1P_CONTACT_SIGNAL_PILOT/<run_id>/`。

### Current Authorization

只授权复用已有 6 集 RGB 接触标签、固定选 6 集额外 Train、准备最小人工审核、固定诊断分类器拟合和独立离线评价。追加人工审核预算最多 6 集，只要求两类接触区间；不要求全 36 集精标。

不授权 policy 训练、Force head、架构或部署修改、自动收紧、真机、新采集、Dev/Final 查看、扩展样本、push/publish 或自动进入 F2。

先读取 AGENTS.md、工作流、STATUS、本合同、原 F1 合同和必要结果。以本合同定义新的小规模实验；保留旧 F1 合同和原始证据，不修改旧指标，不把旧 F1 标为完成。STATUS 标明其后续扩展暂停，改由 F1P 先探索。

## 2. Primary Research Question

> 在固定 6 集拟合、额外 6 集独立验证的低标注预算下，load 是否在夹爪位置与历史命令之外，提供值得扩大验证的接触标签预测增量？

current 单独增量及在 load 之外的增量为次要问题，不取代主问题。

## 3. Paper Relevance

### Candidate Claim

固定的小规模 RGB 独立标注与线性 probe，可为执行器信号的进一步研究提供初步正向或不确定证据。

### Why It Matters

先验证是否值得投入全量标注和后续策略实验，避免在尚无信号价值证据时堆叠模块。

### Non-Claims

不属于论文最终确认；不证明标定握力、抓稳/防滑能力、因果接触感知、P1 已利用 load/current、长保持或真机收益。P1 本来已输入 load/current，本轮不是首次加入信号。

## 4. Baseline / Intervention

| Probe | Inputs | Role |
| --- | --- | --- |
| S0 | 位置、位置差分、过去可用命令、命令差分的固定历史特征 | baseline |
| S1 | S0 + load 特征 | 主比较 S1−S0 |
| S_current | S0 + current 特征 | 次要独立增量 |
| S2 | S0 + load + current | 次要条件增量 S2−S1 |

唯一预期科学差异：在共同样本、标签、标准化、窗口及同一模型下增加指定执行器读数。不针对各 probe 挑样本、特征或阈值。

## 5. Frozen Contract

### 5.1 固定子集与使用边界

1. Pilot-fit = 既有人工标注的 Train episodes `[0,2,4,6,7,9]`；核对其确实属于既有 Train36。复用审核 CSV 的 PRECONTACT / CONTACT_SUPPORTED，两类之外的已标内容保留，不要求重标。
2. Pilot-eval 从既有冻结 Train manifest 中按其原顺序剔除上述 6 集，剩余应为 30 集。选择剩余列表中零基索引 `[0,5,11,17,23,29]`，共 6 集；这是固定分散抽样，不保证空间代表性。
3. 先输出 subset_manifest.json，记录具体 ID、选择代码、来源 manifest 哈希与两组互斥；在新增标签审核及任何 probe 成绩前冻结。不得按信号、成功表现、可见性或 probe 成绩换集。
4. 不足预期数量、fit 标签丢失或 split 冲突则 BLOCKED，不能私自重选。
5. 两组仍都属于原 Train，不把 pilot-eval 重命名为正式 Dev/Test。原 Dev12 与 Final12 均不读取，不从其统计选样。
6. 原 F1 的已做信号审计、时序审计、特征单元测试可以核验版本后复用，但不能宣称重跑；本轮 scaler 仅用 Pilot-fit，不沿用 Train36 scaler。

### 5.2 最小标注与人工负担

只要求两类：

- PRECONTACT：RGB 可支持夹爪与杯子尚未接触；可包含接触前的闭爪。
- CONTACT_SUPPORTED：RGB 可见接触或有后续共同运动等证据支持接触；不等于稳定夹持、离桌或标定力。

无需追加 CLOSING、POST_CLOSE、RELEASE 三类，不要求逐帧点击。审核者观看双视角，定位可靠区间端点；程序展开区间为标签。允许每类多个区间，不要求完整覆盖一集。候选可以用位置/动作定位，但接触判断不能只由闭爪位置决定。

连续播放辅助理解，在不清楚的边界逐帧核查。无法确定时留不确定区间，排除这些帧；不为凑覆盖硬填。事后可看未来视频确定标签，但预测特征不得看未来。禁止向标注者展示 load/current 或模型分数。

新 6 集尽量一次批量交付同一审核页面，自动保存、备份并校验；只把必要的边界问题交给用户，不要求重复已完成 6 集。工具无法观看视频时如实说明，不能把工具限制包装成所有事件不可辨认。

旧标签只做来源、语义、范围和冲突核验；如出现明确错误，在 probe 之前记录并修正来源证据，不能看成绩后改标签。CONTACT_SUPPORTED 不自动外推到整段遮挡帧。

### 5.3 有效性与停止规则

主窗口 W15 的合法历史后，一集若两类各至少 5 个可靠目标帧则为可评估 episode。Pilot-fit 与 Pilot-eval 均至少 4/6 集满足此要求才拟合并给出主增量。所有 6 集的审核状态、类别计数和不可评估原因必须保留。

任一组不足 4 集则科学 UNRESOLVED，交具体审核缺口并停止；不自动加集或切换阶段任务。不同 probe 使用共同有效目标帧。小样本无论帧数多少，独立重复仍最多是 6 集。

### 5.4 因果输入与固定特征

复用原 F1 已审计的保守时序，核验当前源码：obs[t] 在本次 action[t] 生成前到达；输入只用 obs≤t 和已知命令≤t−1。

S0 原始通道固定：pos[t]、pos[t]−pos[t−1]、action[t−1]、action[t−1]−action[t−2]。核实 action 是保存的已发送目标还是其他语义，不能把同一行未知命令用作输入。历史采集源码/固件不可核实的局限须保留。

固定窗口 W=1/9/15，主窗口 15；每通道 current/mean/std(ddof=0)/末减首。load/current 使用同样模板。W1 的窗口 std 和末减首置 0。不得跨集、补历史或加入未来、绝对帧号、时间戳、episode ID、任务进度与视觉特征。真实 FPS 与对应时长另报告。

统一非有限值处理，完整窗口及差分历史不足的帧排除并计数。各 probe 同样本；跨窗口附共同目标帧比较，避免把帧数差异当作窗口优势。主结论不改为表现最佳窗口。

### 5.5 模型与训练预算

固定 L2 Logistic Regression，C=1、class_weight=balanced、solver=lbfgs、max_iter=2000、tol=1e−4、seed=20261015；二分类阈值 0.5。Train-only scaler 这里专指 Pilot-fit，禁止拟合 Pilot-eval 或逐集 scaler。

最多 4 probes × 3 windows = 12 次拟合，除修正可证明的工程错误外不重复搜索。只允许线性诊断 probe，不允许 policy 训练、树模型、神经网络、特征搜索或调参。非收敛如实 BLOCKED，不增预算挽救。

### 5.6 评价与混杂审查

主指标 = W15 下 Pilot-eval 各有效 episode 的 BA(S1)−BA(S0) 的等权均值。BA 为两类 recall 平均。

固定报告各 probe BA、Macro F1、各类 recall、混淆矩阵、每集 delta、原始计数。次要比较 S_current−S0、S2−S1 与其他窗口全报，不因表现差删除。

配对 episode bootstrap=2000，seed=20261015；所有 probe 共享重采索引，报告 percentile 95% CI。仅为探索性不确定度展示，不做正式显著性、无效等价性或独立成功率结论；相邻帧不能当独立样本 bootstrap。

复核 S0 是否已近乎完全区分类别；展示接触前闭爪和接触后帧的 pos/历史运动分布及预测错误案例。若两类位置/运动几乎无重叠，或接触边界仅由闭爪阶段区分，说明 contact 与 motion 无法分离。本轮不硬性追加匹配模型，不把图表审查当因果排除。

load/current 是未物理标定的执行器读数；不同字段的原始寄存器计数不直接换算为握力。current 缺失/常量时次要比较记 NOT_EVALUABLE；load 或 S0 必需通道不可用则主问题阻塞。

### 5.7 Deployment / engineering freedom

部署完全不变。可复用固定特征代码、扩展独立审核页面、缓存、批处理、修 bug、做定向回归；不得改变研究语义。保留所有失败尝试，保护无关工作区内容。

## 6. Assumptions / Open Questions

- `[FROZEN]` 六集旧标签由用户审核，文件格式/覆盖通过不等于本轮已独立复核所有视频。
- `[FROZEN]` 原 F1 未完成，停止全量标注扩展并不等于负结论；本轮独立保存。
- `[PROVISIONAL]` 小规模诊断可能发现值得扩大验证的线索，无法可靠排除小收益。
- `[OPEN]` load 与 current 是否额外编码接触、以及运动混杂，仍待检查。
- `[REJECTED]` P1 有信号输入就已用好信号；短保持必能靠 Force 解决。

## 7. Hypothesis & Falsification

### H1

load 在固定历史特征下，相比 S0 呈现跨 episode 的初步正向增量。

### Expected Mechanism

执行器读数在接触过程中的变化，为位置与历史命令之外提供可解码线索。

### Frozen exploratory decision rule

- `PROMISING_LOAD`：主 delta≥0.03（3 个百分点），且至少 ceil(2N/3) 个有效 eval episodes 的 delta>0。该阈值只决定是否值得扩大研究，不是方法被确认有效。
- `CURRENT_ONLY_LEAD`：load 不满足上项，但 S_current−S0 满足同样探索阈值；明确主假设未获支持，次要信号可再讨论。
- `NO_CLEAR_GAIN_IN_PILOT`：有效覆盖和工程检查通过，但两项均不满足探索阈值。只是本小样本未见明确增量，不等于 NO_INCREMENTAL_SIGNAL。
- `UNRESOLVED`：覆盖不足、数据/时序问题、独立标签不可判或工程不能完成。

混杂审查与 CI 必须随结果一起解释；即便 PROMISING_LOAD，CI 很宽、S0 饱和或运动混杂严重时也只能建议补证，不能直接认为可训练 Force。所有结果标 EXPLORATORY_ONLY。

H1 不被支持的可观察情形包括 load delta≤0、多数 episode 不改善、增量仅由明显阶段/位置差异区分；不改变阈值救结果。标注循环/泄漏使实验 INVALID，不算反驳信号。

### Forbidden Post-hoc Rescue

不得换六集、加样本、选最好窗口、改任务、改阈值、调 C、改特征或打开 Dev/Final。改变科研变量只能 proposal，新合同后再做。

## 8. Gates & Metrics

| Gate | 要求 |
| --- | --- |
| G0 Provenance | 新旧实验分开、合同/代码/manifest/旧标签来源冻结 |
| G1 Data | 两组六集互斥、标签独立、有效覆盖、字段和时间语义 |
| G2 Implementation | 历史截止、共同帧、scaler 范围、episode bootstrap 的定向验证 |
| G3 Training Health | 只检查 probe 收敛与有限输出；policy training=NA |
| G4 Mechanism | probe 增量和运动混杂局限；不称 P1 机制通过 |
| G5 Offline | 独立 pilot-eval 标签预测；不等于机器人动作改善 |
| G6 Deployment | 因果特征可用性检查；完整真机部署等价 NA |
| G7 Real Robot | NA |
| G8 Conclusion | 仅 OBSERVED / 探索线索，不能升级正式确认 |

总体 PASS 表示授权诊断完整、证据可复核，不表示信号有效；PARTIAL_PASS 表示尚有授权项未完成；BLOCKED 表示外部缺口无法继续；INVALID 表示泄漏/循环/研究语义违规；FAIL 表示必要实现验证失败且未修复。

准备好六集审核页面但等待人工标签时提交 PARTIAL_PASS，不能宣称 pilot 已完成。达到冻结停止分支、记录缺口并完成交付，也可结束本轮，但科学仍 UNRESOLVED。禁止自动追加任务。

## 9. Authorization

### Codex MAY

复用核验后的信号/特征审计，准备最多六集新增审核、处理区间标签、固定拟合和评价、修工程错误、生成 proposal。

### Codex MUST

读取当前项目规则；保护已有修改并记录开始/结束状态；保留旧 F1 与失败证据。更新 workflow/STATUS.md 及 docc/日志/工作日志.md，注明从全量 F1 转为小规模 F1P、旧 F1 PARTIAL_PASS、新轮预算和证据边界。保留 D2/D2.1 未闭环、G1 正式真机评价未完成、B3 实为 OC-ACT 的更正。完成后停止。

### Codex MUST NOT

修改 decision、split/子集选择、主指标、阈值、模型、预算；训练 policy；改部署、Force head 或自动收紧；真机、新采集、Dev/Final、push/publish；自动恢复全 36 集标注或进入 F2。

### Requires User Approval

扩大人工标注或科研预算、修改子集/方案、policy-level 实验、新结构、正式确认/Final/真机、新采集协议及 push/publish。合同内正常调试不需反复请示。

## E. Force / Control

- Signals: gripper.pos/load/current，字段和单位实际核验。
- Contact: RGB 支持的物理接触候选，不等于抓稳。
- Grasp / Hold / Release: 本轮不新增这些标签与控制逻辑。
- Controller Ownership: 不接管任何真实关节。
- Calibration: 未标定，不输出物理握力；原 F1 单位审计注明复用来源。
- Fallback: UNKNOWN/局部可靠区间/UNRESOLVED，不循环推标签。

## Required Deliverables

```text
result/
├── summary.md
├── validation.json
├── subset_manifest.json
├── label_manifest.json
├── label_coverage.json
├── probe_config.json
├── leakage_audit.json
├── metrics.json
├── per_episode_metrics.csv
├── bootstrap.json
├── confound_review.md
├── evidence_samples/
├── figures/
└── proposal.md
```

保存拟合和验证两组标签/版本哈希；少量真实双视角证据与精确帧索引随包。raw 保存脚本版本、CLI、fit-only scaler/probes、逐帧特征和预测、复用审计引用、失败尝试及哈希。不适用/未执行写 NA/NOT_RUN 和原因，不填假数。

summary 用通俗中文回答：标注实际花在什么地方；是否看到增量；是否被位置/运动解释；不确定性有多大；是否值得扩大验证。不得只给分类准确率。

输出：

```text
OVERALL_STATUS = PASS | PARTIAL_PASS | FAIL | BLOCKED | INVALID
PILOT_COMPLETE = true | false
PILOT_OUTCOME = PROMISING_LOAD | CURRENT_ONLY_LEAD | NO_CLEAR_GAIN_IN_PILOT | UNRESOLVED
EVIDENCE_SCOPE = EXPLORATORY_ONLY
ORIGINAL_F1_COMPLETE = false
NEXT_POLICY_TRAINING_AUTHORIZED = false
```

结束后停止，不自动生成下轮合同；由用户与 ChatGPT 基于实际结果讨论下一步。
