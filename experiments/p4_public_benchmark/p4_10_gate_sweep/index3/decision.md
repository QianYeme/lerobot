# P4.10-r2 — 解除 LIBERO 确认集容量阻断，继续 Gate 机制验证与 formal6 小预算筛选

日期：2026-10-31
状态：研究方案冻结、交由唯一的本地 Codex 按门槛执行；不授权最终80k或无人监督真机运动。
替代：P4.10-r1 中不可实现的“原LIBERO单任务 >=50 全新官方 init 确认集”要求；保留 r0 和 r1 的全部历史成果与审计记录，不能覆盖。

## 1. 原因与实验边界

- 已核实原 LIBERO 单任务官方 init bank 仅50个；init10–29 与30–49 已用，init0–9是否由早期 Smoke 使用未确定。不得把 reset seed、新训练 seed、重复 rollout 或人为扰动称为新的官方独立初始状态。
- 原 r0 固定P1 Gate+.05 的14/20 vs 独立训练 ACT12/20属于探索性结果；配对3赢1输，p=.625（未作多重比较校正）；不据此宣称最优或统计显著。
- 保留 formal6 120 episodes, Train90/Dev15/Final15，L1 Legacy Seed1、F3 Fixed Seed3，真实杯子原型库与各自在线预处理。LIBERO 单任务结果不能直接指定真机最优Gate。
- Gate 配置的定义是可学习、带符号标量 alpha 的**初始化值**，不是推理阶段硬改 alpha；固定alpha对照是单独机制实验。

## 2. 阶段 A：已完成结果与语义审计（不训练新模型）

A0. 继承 r1 已导出的 +.05 与 NOOBJ 在 init35/38/42/43/48 的真实视频、轨迹及哈希；检查源视频真实性、任务成功标记、配对索引与训练来源。不得将局部抽帧看图当作连续精确动作标注。

A1. 必须为 +.05 对应 on-policy 关键时刻重新产生帧同步的DINO patch similarity maps/overlay（参考目标、机械臂、背景、搬运后原位），保存可审计的帧号、坐标映射、权重、模型权重/原型hash。检查接近、接触、抬升、运输、放置。Codex视觉判读只能记作执行者审阅，若无法实现则标记 SEMANTIC_REVIEW_INCOMPLETE，不编造。

A2. 固定 Gate alpha=.05 与可学习 alpha_init=.05 作相同seed、同非Gate初始权重、训练流、统一5k预算、同Dev9/旧探索init的机制对照。仅在完整来源可验证时启动；记录alpha曲线、梯度、Residual/Query、ON/OFF、训练稳定性；旧init只用于探索，不可称确认集。

## 3. 阶段 B：多seed稳定性 + 另一个LIBERO任务独立测试

B0. Gate候选先冻结为 [0,+.01,+.05,+.1]，主比较 Fixed P1 vs 匹配NOOBJ ACT。禁止根据新测试结果即时扩增Gate取值。先核验r0历史5k checkpoint和可恢复训练配方，找不到则所有比较组统一新建5k配对，不将历史10k混作同预算。

B1. 在原任务上至少两训练seed且统一5k预算；每seed共享非Gate初始权重、数据顺序和正式评估的官方init集合。旧init结果全部标记 REPLICATION_ON_KNOWN_STARTS，不是 unseen-start validation。输出seed内配对、seed间方向、一致性、不确定性，不以多rollout次数伪装独立初态数。

B2. 在开始任何选Gate结果检查前，选择**第二个 LIBERO-Object 任务**作为跨任务验证，检查能否获得合法独立的 task init bank，列明具体任务、合法官方init数量、历史使用审计；若任务初始状态此前已被查看则仅保留确实未用于选择的部分。冻结目标任务、Train/Dev demo划分、可复现新目标原型的四Train图与Codex采样坐标、统一训练seed与预算以及未用官方确认init名单/hash，严禁用确认起始图像/测试标签调原型。

B3. 在第二任务上以相同构建协议训练匹配 NOOBJ/Fixed Gate0/+0.05（是否添加+.01/+0.1由原任务探索结果先锁定最多一名候选；不允许看新任务确认表现后再增补）。Train-only构建新的原型，不能迁移罐头原型或使用模拟器GT分割。先在任务内探索/Dev选择，封存确认init仅一次验证；如果合法独立起点不足，明确报告 LIMITED_CONFIRMATION，不能假造50个独立初态。

B4. 解释两个任务本身难度不同与任务特定原型：这是小规模跨任务证据，不是完整LIBERO Object 10任务benchmark；只在任务内比较Gate，不能把两个任务混为同一模型的逐init配对。

## 4. 阶段 C：formal6 20k 小预算候选筛选（不强制等待B的完整统计胜利）

C0. 只有当A来源与数值检查通过，且Gate候选已在C启动前冻结，才启动formal6小预算。B出现NO_RELIABLE_GAIN可作为不确定性记录继续C，前提不是目标语义明显失败；如果确定原型错误或训练不合法应停并提交Proposal。

C1. 分别以 L1 Legacy Seed1 与 F3 Fixed Seed3 对照 Gate0和Gate+.05；如果B证据支持另一个更稳健值，可在C启动前替换+.05或者新增**最多一个**非零配置，写明筛选前证据、时间及成本。各模型20k从头训练，原版20k可复用须有审计充分证明配方/初始化/数据相同，否则统一从头配对训练。

C2. 固定formal6 Train90、Dev15、封存Final15、历史4×384杯子原型、Top/Wrist预处理、8D状态、6D action、训练调度。各架构内Gate之外的初始权重逐张量相等；写入SHA和每step数据顺序证据。L1 Seed1 vs F3 Seed3不能作为单变量DINO预处理因果对比。

C3. 评估chunk_valid Arm/Gripper MAE、动作阶段分解、Gate/Residual/Query/ON-OFF、梯度、异常输出/数值稳定性。20k只是离线候选筛选，不宣称最终真机有效，Final15不参与选参。

## 5. 选择 / 停止规则

- 分开报告`L1_gate_candidate`和`F3_gate_candidate`，候选可以相同、不同或均为0；若无法支持则写INCONCLUSIVE/NO_GATE_CHANGE，不表示已证明0最佳。
- 只将真实闭环收益、跨seed方向一致性、formal6离线与阶段诊断、语义证据综合形成候选；Object ON/OFF变大、alpha变大不单独构成入选。
- 任何候选只有经后续**单独获批**的80k、在本地RTX4060完成Smoke和受监督配对真机表现后，才能称为最终优胜模型。
- 若第二任务容量或训练来源不可验证，生成blocking_proposal但允许不依赖此问题的A以及合规C继续，只需清晰报告各分支状态，不再将全部研究无限期阻断。不得擅自更换数据、初态规则、训练步数或方案。

## 6. 执行/交付

唯一的本地Codex经SSH使用远程RTX5090；先查有无现存任务，不能破坏P4.7或旧checkpoint；实测多进程有效steps/s后决定并行度。持久会话、可恢复训练、日志、checkpoint hash、原型与数据来源留痕。未获得权限的真机活动禁止执行。

建议目录：`experiments/p4_public_benchmark/p4_10_gate_sweep/index3/decision.md` 与并列 `result/`。

必须交付：`status.json`, `lineage_audit.md`, `semantic_review/`视频与on-policy热图、`fixed_vs_learnable_gate.csv`, `known_start_seed_replication.csv`, `cross_task_freeze.json`, `cross_task_confirmation_pairs.csv`（如合法执行）, `formal6_20k_comparison.csv`（如合法执行）, `gate_selection_report.md`, `blocking_proposal.md`（仅有阻断时），`builder_review.md`, `red_team_review.md`, `summary.md`。Builder/RedTeam未独立完成时标PENDING。
