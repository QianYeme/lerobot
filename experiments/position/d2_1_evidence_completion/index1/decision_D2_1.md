# Experiment Decision Contract

> Revision: r0  
> 日期：2026-10-14  
> 用户在 D2 部分验收后回复“可以”，认可补完证据方向。本合同由用户与 ChatGPT 冻结；Codex 只读。

## 1. Identity & Mode

- Experiment ID: D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION
- Track: POSITION / TASK_BEHAVIOR
- Stage: ACTION_CHAIN_EVIDENCE_COMPLETION
- Index: 1
- Mode: DIAGNOSIS
- Repository: https://github.com/QianYeme/lerobot；branch: det
- 项目内路径：experiments/position/d2_1_evidence_completion/index1/decision.md
- Previous: D2_P1_ACTION_CHAIN_DIAGNOSIS，PARTIAL_PASS / UNRESOLVED / DIAGNOSIS_COMPLETE=FALSE。

已有结果产生后发现 B3 科研身份冲突，故建立独立补证轮，不改写原 D2 科研合同。原结果保留并引用，新的核验、纠正与阶段分析写入本轮。

### Current Authorization

仅授权已有 Train/Dev、冻结模型与代码的审计，视觉事件审核，合法推理缓存分析，必要的固定 checkpoint 补充离线推理，后处理等价验证和文档更新。

零训练；无新采集；不向机器人发送指令；不改实际部署参数；Final Test 继续封存。

## 2. Primary Research Question

> 在核验身份及回放正确性、补齐 P1 抬升/保持阶段事件后，现有证据能否解释 P1 相对示范的行为衰减，并确定一个恢复示范水平的最小下一步？

用户目标：逐步恢复现有示范水平；现有数据优化成熟后另行设计更好数据集和独立最终测试。本轮不以超过示范为目标。

## 3. Frozen Facts / Open Questions

- 用户现场观察：P1 拿起后保持夹紧并平稳放下，松爪后退回停止；抬升和停留弱于本来就较低较短的示范。定性观察不等于正式成功率。
- D2 报告 P1/R3 各 Dev12 / 5748 帧，48 集信号审计，240 事件 unknown；阶段覆盖 0。
- D2 报告时间集成增加全任务回放误差，但未证实抬升/保持阶段因果；真实 runtime 配置未核验。
- B3 checkpoint 被报告为 model_type=oc_act、object_conditioning_enabled=true；尚需配置原件、模型实例和历史 manifest 三方核验。
- 配置字段不能单独证明训练/推理实际启用目标分支。此前将 B3 描述为纯 ACT 的结论暂时撤回。
- P1 是主候选，B3/R3 是既有比较参照；不据此预定真机性能排名。

## 4. Baseline / Intervention

### Baseline

D2 冻结 P1 原始预测及同 episode 示范 action/state。重用 D2 P1/R3 raw cache 前必须核验模型、processor、原型、样本索引、推理协议及完整性。

B3 为 IDENTITY_UNVERIFIED 的历史候选，不称为纯 ACT，不自行替换。

### Intervention

补充可追溯视觉阶段事件和回放等价验证；在相同冻结预测上进行阶段化分析。

### Only Intended Scientific Difference

无模型或训练干预。后处理保持 D2 的 off/0.01 × limit 5/2 四配置，每个配对只变一个因素；阶段分组来源明确。B3 身份纠正是事实核验，不是授权改模型。

## 5. Frozen Contract

- formal4/kind_merged_nomaster；原 Train36 / Dev12，原 train-only stats，不读取 Final12 的行、视频或标签。
- 8D 状态：五关节、gripper.pos/load/current（实际字段可能为 curr）；6D action；双 RGB。
- 固定 P1/R3 G1 checkpoint、prototype、tau、processor、模型结构、随机性和推理配置。
- 不以旧 Formal3 阶段标签解释 Formal4。
- no-grad/eval；不挑 checkpoint/seed/episode，不调 Dev 阈值。
- 单位、action 语义、时间对齐、视频帧索引先验证；没有可信 FK 不报告毫米抬升。
- 教师强制回放始终以专家 state/video 为条件，不能声称自主闭环或真实保持时间。
- 原 D2 数据、decision 与历史报告保留；不得回填 G1 成功率或置 G1 complete=true。

## 6. Execution Plan

### A. B3 身份核验

保存 G1 三候选 manifest、B3 原 config/train_config、checkpoint key/module 摘要及哈希；对照 T3 的训练构造代码与运行记录。

区分 CONFIG_ENABLED、MODULE_PRESENT、FORWARD_PATH_ENABLED、TRAINING_PROVENANCE_VERIFIED 四类证据。可作无硬件 forward hook，不改 checkpoint。不因仅有字段就断言实际使用。

结论：PLAIN_ACT_VERIFIED / OCACT_VERIFIED / UNRESOLVED。身份冲突不得阻塞 P1 事件与回放工作。

若为 OC-ACT，报告 B3/P1 比较的科学含义需重审，不再声称标准 ACT vs object-aware；允许固定原 B3 作“既有 OC-ACT 候选”辅助参照，但纯 ACT baseline 仍标缺失。不得换 B0、重新训练或关闭分支冒充纯 ACT。

### B. 视觉事件审核

先完成 Dev12 的完整视频审核和 P1 阶段分析，再完成 Train36；这是执行顺序，不改变全48集交付要求。部分结果必须分 split 报覆盖。

每集核验：有效闭爪、杯子离桌、下降开始、重新接触桌面、释放；另记录最大抬升附近的停留区间。双视角互证，遮挡或无停留则分别记 UNKNOWN / NO_HOLD_OBSERVED，不补造事件。

区分闭爪后抬升前等待、离桌总时长、顶部停留；不能用离桌总时长代替 hold。事件须带 frame、time、来源、可见性、证据图或片段及不确定区间。

有能力查看视频则由 Codex 自主审查，标 VIDEO_REVIEWED_BY_CODEX；人工确认才标 HUMAN_REVIEWED；信号规则为 AUTO_PROXY。不得将“缺人工标签”当作停止所有可执行视觉审核的理由。

自动候选规则只能 Train 制定，再固定应用 Dev。人工/视觉事件审核直接依据可见行为，不用 Dev 误差选择边界。负载/电流不能替代离桌真值。

若运行环境无法解码或查看，先完成解码/抽帧工程修复；仍无法判断时交付精确到 episode/片段的审核清单及原因，不能只交全 unknown 占位表。

### C. 阶段分析

在可靠事件上分别评价接近、抓取、抬升、顶部保持、下降、释放。unknown 单列并保留全任务误差。

报告 per-episode 原始 h0 / 四配置后处理输出的五关节与夹爪 MAE、相对录制 state 的动作差值、裁剪率及处理差值。保持指标必须有可靠区间；无停留集不得凭关节低速度自动称为视觉 hold。

关节空间运动幅度/趋势只是代理量，不以单关节解释笛卡尔高度。预测 chunk 的保持结构仅是预测序列特征，不是真机停留。

合法未来 action 按 episode 索引重建并正确 mask，补报不同 horizon 误差，不能只用 full-chunk 平均掩盖 h0 与远期差异。

如复用缓存缺关键字段，固定 P1 优先补推理，R3 次之；不得扩大模型搜索。

### D. 回放验证与报告一致性

核验真实代码的时间集成权重顺序、重叠 chunk 时间索引、n_action_steps、队列及 reset。明确 off 与 on 是否改变取 action 语义，不能将队列差异全归因于平滑。

独立验证 normalized 集成后反归一化与反归一化后集成是否等价，覆盖真实样本、边界及浮点误差。若 normalization 非线性或权重不同，解释差异并按真实代码路径重算。

验证限幅调用与实际 follower helper 一致，记录离线使用的是录制 state，而真实 helper 使用 fresh Present_Position 的限制。不擅自积分出模拟机器人状态。

完成 cache/frame/chunk shape 和未来标签数量检查。报告模型/processor/prototype 哈希、代码状态及成功/失败运行时间。

D2 provenance 的本地 torch=false 与服务器推理分开记录，不能混成一个环境；补服务器代码 commit/diff/依赖。

新报告引用 D2 各文件时间与作用，明确旧 BLOCKED_CHECKPOINTS_AND_TORCH 已过时；统一 full_chunk_ground_truth_evaluated 等状态。不得覆盖旧证据来掩盖矛盾。

## 7. Hypothesis / Falsification / Non-Claims

H1：补完阶段和回放证据后，可比 D2 全任务误差更明确地排序下一步优先级。

- 示范清晰且 P1 原始输出在抬升/保持明显退化：支持 PREDICTION_PRIORITY。
- 原始输出较好，而正确复现的后处理明显改变相关阶段：支持 POSTPROCESS_PRIORITY，仍非真机因果确认。
- 示范边界不可辨、标签错位或目标/反馈冲突：支持 DATA_QUALITY_PRIORITY；示范短本身不是缺陷。
- 多因素并存：MIXED；不足以区分：UNRESOLVED。
- 原始预测和回放均正常：不能直接断言部署问题，记录自主闭环分布偏移尚未核验。

没有假设被支持时也应如实完成诊断，不强凑根因。不能声称力控有效、FiLM 追踪因果、位置泛化成功率或真机恢复。

## 8. Gates / Primary Endpoint / PASS–FAIL–STOP

Primary endpoint：诊断类别、可追溯证据覆盖与下一步单一建议；不是性能 PASS 指标，不设事后成功阈值。

| Gate | 必要检查 |
|---|---|
| G0 | 冻结 provenance、B3 身份核验与纠正、不读 Final |
| G1 | 字段/单位来源、时间及视频索引、事件来源和 unknown |
| G2 | 回放等价、队列/时间索引、mask/标签计数定向验证 |
| G3 | NA，无训练 |
| G4 | 模块路径静态/forward 核验；后处理影响，不夸大训练使用因果 |
| G5 | P1 Dev12 全列、Train36事件覆盖、阶段及 horizon 分项 |
| G6 | 离线语义核验；真实执行等价与 runtime 仍未核验 |
| G7 | NA，无新增真机 |
| G8 | 归因受证据限制，单一下一步或精确补证需求 |

PASS：可执行核验完成，全48集均有审查记录、可判与不可判来源可追溯；P1 Dev12全列；回放正确；结论不超证据。诊断可诚实 UNRESOLVED，不能因此伪造明确归因。

PARTIAL_PASS / BLOCKED：缺资产或确实无法判定，交付已完成证据和精确缺口；不能将未执行视觉审查等同已审但不可判。

FAIL：实现校验失败致输出不可信。INVALID：泄漏、换模型、篡改标签/合同或虚构证据。STOP：达到预算或资产不可用，保存进度，不自动启动训练/采集。

## 9. Budget / Authorization

额外 GPU 推理累计不超过4小时；优先复用 P1/R3缓存，避免重复全量推理。实际审核时间如实记录，不要求靠全量重训练解决。

Codex MAY：自主工程修复、视频解码/审核、可追溯候选标注、合法缓存、验证与图表、必要离线推理。

Codex MUST：完成可独立执行项；保留 D2 历史；更新 workflow/STATUS.md、docc/日志/工作日志.md、候选模型说明的身份更正；用复核附注更新 G1 当前状态，不改原 G1 decision 或伪造 trials。

Codex MUST NOT：改本合同、训练、换 checkpoint、替换纯 ACT baseline、实际改部署/操作机器人、读 Final、融合、进入 Force/Confirmation、未经授权 push/publish。

需要新授权：具体优化、实际部署变更、少量带日志真机验证、更好数据采集或最终训练/独立测试。完成本轮后先验收讨论，不自行生成下一轮 decision。

## 10. Deliverables

本轮目录 result/：

- summary.md、metrics.json、validation.json、provenance.json。
- b3_identity_audit.json 和引用的配置/模块/manifest 证据；candidate_identity_correction.md。
- demonstration_events.csv、visual_review_manifest.json、demonstration_audit.json。
- per_episode_phase_metrics.csv、prediction_diagnosis.json、horizon_metrics.json。
- replay_equivalence_validation.json、postprocess_replay.json、deployment_contract_audit.json。
- report_consistency_audit.json、evidence_manifest.json、figures/。
- proposal.md：一个主要下一步及其反证，无法选择则明确补证需求。

Raw: results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/。

结束输出 DIAGNOSIS_COMPLETE=TRUE/FALSE、acceptance_status、diagnostic_category、B3_identity_status。POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE 仍 FALSE；停止。
