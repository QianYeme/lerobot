# Experiment Decision Contract

> Revision: r1（替代此前未冻结、不得执行的草案）  
> 日期：2026-10-14  
> 用户在完整讨论后于 2026-10-14 回复“继续”，批准本轮零训练诊断。本合同供 Codex 执行；Codex 只读。

## 1. Identity & Mode

- Experiment ID: D2_P1_ACTION_CHAIN_DIAGNOSIS
- Track: POSITION / TASK_BEHAVIOR（不是 Force Integration）
- Stage: LOW_LIFT_SHORT_HOLD_DIAGNOSIS
- Index: 1
- Mode: DIAGNOSIS
- Repository: https://github.com/QianYeme/lerobot；branch: det。
- 项目内路径：experiments/position/d2_action_chain_diagnosis/index1/decision.md
- Dataset: formal4/kind_merged_nomaster；使用既有冻结 Train 36 / Dev 12。
- Previous Experiment: G1_POSITION_REAL_ROBOT_GENERALIZATION。
- G1 状态：PARTIAL_PASS，定性真机观察已发生，正式评价未完成；POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE=FALSE。
- Code State: 执行前记录实际 commit、dirty diff、依赖与 checkpoint/processor/prototype 哈希。报告中的 det / 1a4ef845 只是历史线索，不冒充实际执行版本。

### Current Authorization

只授权：已有数据与模型的只读审计、分析脚本、固定 checkpoint 离线推理、无硬件动作的部署后处理回放，以及证据和文档更新。

不授权训练、微调、模型结构修改、实际控制参数修改、新数据采集、追加真机、Final Test、模型融合或 Force 集成。

## 2. Primary Research Question

> 为什么当前 P1 真机抬升与空中停留弱于现有示范？在冻结模型的离线证据下，能否区分预测退化与部署后处理影响，并确定恢复现有示范水平的最小下一步？

本轮目标是缩小不确定性，不承诺离线诊断能唯一解释真机根因。

### 已达成的用户共识与最新现场事实

- 当前目标是先恢复现有示范水平，不要求本轮创造更高抬升或更长保持。
- 示范本身抬升不高、停留不长，但真机更低、更短，像连续拿起后放下。
- P1 抓起后保持夹紧并平稳放下；放下松爪后退回并停止。尚无证据说明本问题主要是滑落或提前松爪。
- P1 为主候选，B3 为 baseline，R3 为行为参考；不将定性排序写成正式性能结论。
- 长期顺序：现有数据诊断与单变量优化 → 小规模带记录真机验证 → 更好数据集采集与最终训练/独立测试。后面各阶段仍需新合同，本轮不自动授权。
- “优化到最佳”不承诺数学最优；当前阶段关注示范复现、动作链稳定和当前杯位范围内可重复改善。
- 新数据若参与方法调参属于开发数据；最终测试须独立、提前冻结。当前 Final 12 继续封存。
- 尚未排除自主运行中的观测/状态分布偏移；离线预测较好不能直接推出部署是唯一原因。

## 3. Paper Relevance / Non-Claims

本轮可形成：冻结数据中的动作阶段描述、固定模型的分阶段预测误差、后处理对录制状态下动作输出的影响。

即使 PASS，也不能声称：

- P1 真机泛化优于 ACT，或已达到可靠任务成功率；
- FiLM 显式坐标产生了 R3 追踪；
- 已确定真实执行低抬升的唯一原因，或凭离线正常直接排除闭环分布偏移；
- 已实现力控或证明 load/current 被有效利用；
- teacher-forced 回放等同自主闭环；
- 无标定像素位移或关节角变化等同毫米高度。

## 4. Baseline / Intervention

### Baseline

主诊断对象：G1 冻结 P1_OCACT 100000 checkpoint。B3_ACT / R3_FILM 的 G1 冻结 100000 checkpoint 为同协议参照。

优先从原 G0/G1 manifest 解析模型身份；P1 历史目录 P1_ocfix2_s1/100000 仅作定位线索。禁止使用首轮未实际注入 OC-ACT 的无效 P1。禁止按本轮误差选择 checkpoint。

### Intervention

无训练干预。比较相同录制观测上的示范目标、模型原始输出，以及经过明确后处理配置的离线输出。

### Only Intended Scientific Difference

部署敏感性对照只切换时间集成或限幅一个因素；模型比较只使用既有冻结模型。其余数据、观测、索引、归一化与推理协议保持一致。

## 5. Frozen Contract

### Data / Observation / Action

- 冻结既有 episode split；统计仅 Train；不得读取 Final 的 parquet、视频、标签或执行推理。
- Top/Wrist RGB；8D 状态含五关节位置与 gripper.pos/load/current；6D action。
- 核验实际字段顺序、单位、帧率、action 语义、时间戳对齐和 processor。缺失即明确报告，不能凭名称推定。
- Formal3 标签不得直接套用 Formal4；旧数据的保持时长不作为当前示范证据。
- 不改变数据内容、现有标签、split、prototype、tau=0.20 或归一化。

### Models

- B3：标准 ACT，无 DINO/object tokens/FiLM。
- P1：ACT + frozen DINOv2-S + 4 prototypes + 5 object tokens + gated cross-attention。
- R3：保留 OC-ACT object branch，额外对 state token 施加 spatial FiLM。
- 三者结构关系不等于 checkpoint 初始化关系；实际训练 provenance 必须记录。
- eval/no-grad；固定推理随机性；保留 checkpoint 原配置。路径修复仅允许指向哈希相同资产，单独记录。

### Evaluation

主评估 h0，并报告动作关节与夹爪分项；完整 chunk 仅在显式重建合法同 episode 未来目标和 padding mask 后评价。

全 Dev 12 episode 按可用合法帧执行；Train 36 用于数据分布、规则制定和必要参照。指标按 episode 汇总，同时报告所有 episode，不能只展示成功片段。

推理需记录 latent 使用方式、chunk 长度、n_action_steps、动作队列、推理频率；不得把旧离线 L1 当成本轮分阶段结果。

## 6. Execution Plan

### A. Provenance / 数据与部署审计

1. 核验 split、checkpoint、原型、processor、输入顺序和实际控制脚本。
2. 核查 Formal4 action 是否为限幅后实际发送目标，state 是否为实际反馈；核查录制延迟与可达性，发现错位只报告，不擅自重标或平移。
3. 解析并记录 G1 指南、checkpoint 配置及最新运行命令之间的差异，尤其 temporal ensemble、max_relative_target、n_action_steps。
4. 没有真实运行日志时，将现场实际参数记为 UNVERIFIED。不得以指南配置充当实际解析配置。

### B. 示范阶段审计

1. 先检查 Formal4 是否已有经人工审核的阶段标签。无则建立可追溯候选事件表，不沿用 Formal3。
2. 在全部 Train/Dev 可读取视频中标出：有效闭爪、杯子离桌、下降开始、杯子重新接触桌面、释放；遮挡与无法判断记 unknown。
3. 在固定视角下区分：闭爪后抬升前等待、实际离桌后停留、整个离桌持续时间。三者分别记录，禁止合并为 hold。
4. 接触/离桌是视觉事件；load/current 仅辅助线索，不能单独作为成功抓取或离桌真值。
5. 有人工审核则标 HUMAN_REVIEWED；Codex 视频判断标 VIDEO_REVIEWED_BY_CODEX；自动信号规则标 AUTO_PROXY。保留来源、帧号、证据图及不确定区间，不冒充人工标注。
6. 自动规则仅由 Train 制定并冻结后应用 Dev；Dev 不调阈值。无合法精确事件则报告时长区间或 unknown。
7. 无可信运动学/标定时只报可解释的关节运动范围、图像轨迹及秒数，不报物理高度。图像坐标须说明相机、遮挡与透视限制。
8. 审计夹爪在抬升/运输中的重新张开；以数据单位报告变化，不能自动称为掉杯或夹持力不足。

### C. 模型分阶段误差

1. 三模型在相同合法观测上推理，比较 approach/grasp/lift/airborne_hold/lower/release；无法可靠分阶段的帧不进入阶段指标，但必须保留 unknown 计数及全任务误差。
2. 报告 h0 action MAE：五关节和 gripper 分项、归一化及反归一化单位；统计示范与预测动作相对当前 state 的差值和符号，不把单关节当作笛卡尔抬升。
3. 输出 per-episode 阶段样本数、误差、夹爪重开趋势及预测/目标曲线；描述短阶段在训练中的占比。
4. 合法 full-chunk 标签按 episode 边界重建，明确 action 索引和 padding；没有未来标签的样本只报预测分布，不能伪造 ground truth。
5. 时间集成可跨录制观测产生多 chunk 重叠输出，但所有 state/video 仍是专家轨迹。明确 teacher-forced，不称为策略自主状态或真实保持时间。

### D. 部署后处理敏感性

1. 必须调用或等价复现当前控制代码，核验时间集成权重方向、索引、队列更新、单位和限幅位置；不得凭通用 ACT 实现猜测。
2. 检查以下 2×2 配置：temporal ensemble off / coeff=0.01；max_relative_target=5 / 2。其余冻结；按固定另一因素的配对结果解释单因素变化。
3. “off + 5”仅为指南/配置参考，不认定为历史实际运行配置；“0.01 + 2”仅为报告参数，实际运行仍未核验。
4. 输入完全相同的原始模型 chunk、录制 state 和时间顺序；每 episode 重置集成和队列。不向机器人发指令。
5. 报告每关节裁剪率、处理前后差值、集成与限幅各自影响、阶段分布。若共同作用不能相加，单独报告交互。
6. 缺失代码或参数时对应项 BLOCKED。不能把离线被裁剪解释为“真机一定抬不起来”，也不能修改部署来验证。

## 7. Hypothesis / Falsification

H1：低抬升与短停留存在可定位的任务链瓶颈，现有只读审计至少能缩小下一轮干预范围。

竞争解释与证伪：

- DATA：本轮示范是恢复参照，时长短本身不构成失败。只有事件含混、严重不一致或目标与反馈明显不匹配时，才提出数据质量优先；不得因为用户示范抬得低就擅自转向新采集。
- PREDICTION：示范有明确行为而模型对应阶段持续偏离，支持模型/监督优先；模型较好复现目标则该解释减弱，但不排除闭环偏移。
- POSTPROCESS：离线后处理明显改变相关阶段输出，支持部署敏感性；差异很小则该因素在回放条件下解释力减弱。
- MIXED / UNRESOLVED：允许多个因素共存或证据不足，不强行选择唯一根因。
- CLOSED_LOOP_UNRESOLVED：离线预测和后处理均不能解释时，记录闭环分布偏移等未核验因素；可建议后续最小带动作/state 日志的真机验证，但本轮不执行。

禁止事后换 checkpoint、删除难例、调 Dev 阈值、将未量化观察转换为成功/失败计数、以离线 replay 修复宣称真机修复。

## 8. Gates / Metrics / Outcomes

### Primary Diagnostic Endpoint

诊断类别及可追溯证据覆盖：DATA_QUALITY_PRIORITY / PREDICTION_PRIORITY / POSTPROCESS_PRIORITY / MIXED / UNRESOLVED。闭环未核验作为独立 limitation 记录。

这是诊断终点，不是成功率或方法优越性指标。必须同时提供支持证据、反证、覆盖 episode 数及证据局限。

### Mandatory Gates

| Gate | 要求 |
|---|---|
| G0 Provenance | checkpoint/代码/数据来源可追溯，冻结条件无违规 |
| G1 Schema | 8D/6D 字段、processor、时间及 episode 对齐通过 |
| G2 Implementation | 标签重建、mask、事件计时与后处理有针对性验证；不添加镜像实现的无意义测试 |
| G3 Training | NA，本轮无训练 |
| G4 Mechanism | 仅核查已知模块身份及后处理实际进入路径；不宣称新增目标/力控因果证据 |
| G5 Offline | 固定 Dev 12 完整列出；阶段 unknown 和缺失如实报告；三模型同协议 |
| G6 Deployment | 可核验静态语义与离线后处理；真实执行等价性因无日志保持 UNVERIFIED |
| G7 Real Robot | NA，本轮禁止追加真机；G1 不因本轮诊断升级 PASS |
| G8 Conclusion | 归因不超证据，下一步只提出一个主要干预建议或明确缺失证据 |

### Secondary Metrics

示范事件/时长与不确定区间；各阶段 frame 占比；每 episode h0 MAE；夹爪重开变化；合法 chunk 误差；后处理裁剪率与输出差值；unknown/可评估覆盖率。禁止虚构毫米高度。

### PASS

G0/G1/G2 与可执行的 G5/G6 静态审计完成，无泄漏；提交全 Dev 12 的记录；结论可追溯且下一步选择有证据或明确反证。这里 PASS 仅指 DIAGNOSIS_COMPLETE，不指 P1 性能 PASS。

### PARTIAL_PASS / BLOCKED

部分审计可用但标签、视频、代码或 checkpoint 缺失；准确列出不可完成项。材料不足时不得生成“唯一根因”。

### FAIL / INVALID / STOP

- FAIL：实现校验持续失败，诊断结果不可信。
- INVALID：读 Final、换 checkpoint、篡改数据、伪造标签/真机指标等破坏合同。
- STOP：出现资源或输入问题，先保留已产证据；不自行采集、训练或联网重建未知模型。
- 合法审计完整但 UNRESOLVED 可以为诊断 PASS；优化选择仍 BLOCKED，不硬凑原因。

## 9. Resource Budget

零训练；固定三份 checkpoint；优先复用合法 raw 推理缓存。单 GPU 顺序/受控并行均可，OOM 可降推理 batch，不变样本与指标。

先完成 P1 全 Dev 12 及数据/后处理审计，再做 B3/R3 同协议参照。不得为了对照而延误 P1 主诊断；预算不足则明确对照未完成，不偷偷改成更小子集再宣称公平全量比较。

GPU 推理累计预算 4 小时；达到上限保存中间结果并报告 PARTIAL_PASS，不擅自加训练或扩大搜索。无 GPU 时先完成 CPU 数据、配置审计。不得申请未提供的远程凭据或假定旧服务器仍可访问。

## 10. Codex Authorization

### MAY

编写诊断与报告脚本；修正常规工程错误；缓存特征/推理；增加验证和图表；并行独立只读分析；列出需要人工审核的视频片段；提交 proposal。

### MUST

- 执行前阅读本合同、workflow 和最新 STATUS；冻结 provenance。
- 主线围绕可靠抓起与动作链，不回到未经验证的横向 dim0 硬门槛。
- B3/R3 是参照，P1 是诊断重点；不得预先宣布 P1 最佳。
- 正式训练集统计与 Dev 明确区分；原始证据与解释分开。
- 更新 workflow/STATUS.md 和 docc/日志/工作日志.md：记录 G1 正式评价未完成、用户暂停真机、本轮授权范围、真实结果与阻塞，不覆盖历史。
- 更新 G1 result 的当前 summary/validation 或新增复核文件并明确旧硬件 BLOCKED 已过时；保留旧文件与原 decision，不伪造 trial_results。
- result/proposal.md 只给待讨论候选，不产生执行权限。
- 完成后停止，不能自行写下一轮 decision 或开始优化训练。

### MUST NOT

修改本合同；重训/微调；改结构、prototype、tau、数据/标签真源、split 或实际控制设置；读 Final；追加真机；融合 P1/R3；自动进入 Force/Confirmation；未经授权 push/publish。

### Requires New User Authorization

任何新训练、新监督/采样干预、新模型结构、实际部署参数变更、新采集、真机验证、Final Test、Force Integration 或发布。诊断证据形成后先交付验收和推荐，再冻结新合同。

## 11. Required Deliverables

result/ 必须包含：

- summary.md：通俗结果、证据等级、诊断类别、下一步单一推荐及反证。
- metrics.json / validation.json：状态、覆盖、资源、Final=false、training=false、real_robot=false。
- provenance.json / schema_audit.json / deployment_contract_audit.json。
- demonstration_events.csv：split/episode/event/frame/time/来源/置信或不确定区间。
- demonstration_audit.json：可用/unknown、示范时长与阶段统计、标签合法性。
- per_episode_phase_metrics.csv / prediction_diagnosis.json。
- postprocess_replay.json：四配置与配对比较、静态核验、真实执行未核验说明。
- figures/：有用的示范/预测曲线、阶段误差和后处理差异。
- evidence_manifest.json：raw 路径、哈希、调用命令、失败/缺失清单。
- proposal.md：待批准的单一主干预及最小验证；证据不足则优先补证据。

Raw evidence 保存至 results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/，交付紧凑 result 包。所有报告遵守 G1 原始 trial 数与成功率未知，不把 null 写成 0。

最终输出 DIAGNOSIS_COMPLETE=TRUE/FALSE、diagnostic_category、acceptance_status；POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE 仍为 FALSE。停止并等待网页验收。
