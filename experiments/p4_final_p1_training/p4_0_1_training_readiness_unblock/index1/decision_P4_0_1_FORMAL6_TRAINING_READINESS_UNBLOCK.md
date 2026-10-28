# decision.md — P4.0.1 Formal6 Training Readiness Unblock

> **实验编号**：`P4_0_1_FORMAL6_TRAINING_READINESS_UNBLOCK`  
> **状态**：`FROZEN / NOT_RUN`  
> **执行角色**：Codex（仅执行）；科研语义与验收由用户 + ChatGPT 决定  
> **性质**：训练前解阻 / 数据接口与 split 冻结；**不训练、不运行真机、不改变 P1 体系**  
> **建议仓库位置**：`experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/decision.md`

## 1. 输入依据与边界

本轮唯一数据源为 Hugging Face `QYyyyyyyy/formal6_merged`，必须 pin 审计所依据的 revision：

`83a829af6df245ed53596dbc22b18c19f8bd3315`

复用上一轮 P4.0 的产物：`summary.md`、`schema_and_interface_audit.md`、`split_manifest.csv`（**候选，不是最终**）、`split_method.md`、`trajectory_nearest_neighbors.csv`、`collection_session_confounds.md`、`training_preflight.md`、`validation.md`、`episode_manifest.csv`、`prototype_source_plan.md` 与用户已批准的区域标签要求变更记录。不要覆盖 P4.0 的证据。

P4.0 已确认：120 episodes、93,391 frames、双相机完整解码，粗筛无明显失败；**不代表每条物理阶段都精审通过**。实际环境为机械臂左抓右放，约 10.5×13.5 cm Source、6.5×6.5 cm Target；相机投影方向可能与物理方向相反。不要凭图像左/右误判任务方向。

本轮要解决的是“能否按可信、可复现的合同开始 P1 从头训练”，不是证明 `formal6` 已优于历史数据。不得以缺少 R01–R15 精确区域标签或 120 集逐帧人工标签作为默认阻塞。

## 2. 唯一科研问题

> 在不更改 P1_OCACT 主架构、原始数据及实验目标的条件下，能否冻结 `formal6_merged` 的 **9D→8D 输入映射、状态/动作单位合同、无明显泄漏的 90/15/15 划分与训练入口配置**，从而可靠进入下一轮 P4.1 Pilot？

结论只能是：`TRAINING_PREFLIGHT_READY`、`READY_WITH_DECLARED_LIMITATIONS` 或 `BLOCKED_WITH_EVIDENCE`。绝不靠猜测给出 READY。

## 3. 不变项

- 主模型：`P1_OCACT`；历史 P1 最优模型的架构语义不变。
- 原有 P1 state **8D**：5 关节 + `gripper.pos` + `gripper.load` + `gripper.curr/current`。
- 动作 **6D**：5 关节 + `gripper.pos`。保持 ACT chunk 接口，不改预测范围。
- 原始 ACT Top/Wrist 视觉骨干 + Top-only Frozen DINOv2-S + 4 Train-only Cup Prototypes + soft pooling τ=0.20 + 1 Global/4 Local tokens + cross-attention/gated residual，不改结构。
- 原始数据 revision、episode 集合及素材不可改写；若需要训练映射，优先非破坏性 dataset adapter，必要时生成独立命名的派生 manifest，带 hash 和谱系。
- 历史 Force 分支和 Reactive Load Auxiliary 不纳入本轮；P1 **仍保留** load/current 输入。
- 不训练 P1、Original ACT、New-GL；不跑任何 optimizer step；不运行真机；不执行最终测试。

## 4. 工作包 A — schema 与 9D→8D 映射冻结

依据数据集**实际 features/字段名称**而非仅数组切片顺序，确认原始 state 顺序（P4.0 审计为）：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
gripper.load
gripper.curr
master_gripper.pos
```

冻结 P1 输入映射：

```text
P1.state[0:8] =
  shoulder_pan.pos,
  shoulder_lift.pos,
  elbow_flex.pos,
  wrist_flex.pos,
  wrist_roll.pos,
  gripper.pos,
  gripper.load,
  gripper.curr
```

`master_gripper.pos` **只作为历史遥操作元数据**，不得输入自主 P1。冻结动作 6D 字段名与顺序，并核对 camera path：`observation.images.top`、`observation.images.gripper`。接口字段在不同数据集/代码中可能称为 `gripper.current` 或 `gripper.curr`，必须明确实际取数映射，不可悄悄创建一条新维度。

执行：检查远端 metadata / features、当前 P1 dataset loader、训练 transform、模型构造、runtime schema，完成无训练的静态/小批量读取校验；核对 batch shape、dtype、有限值、归一化前后 shape、相机分辨率与通道顺序。不得向模型输送 9D 再赌运行时隐式裁剪。

输出：`schema_mapping_contract.md`、`schema_mapping_validation.json`，包括前/后字段列表、原始与派生 revision/hash、只读测试证据、匹配的配置与代码位置。

**A gate：** `SCHEMA_PASS` 仅在 8D/6D 名称、顺序、加载器/模型一致且所有验证成功时成立。否则 `SCHEMA_BLOCKED`，附最小修复提案。

## 5. 工作包 B — 单位、标定与归一化合同

优先取证：

1. `formal6` 采集时的机器人配置（尤其 `use_degrees`）、leader/follower 标定文件及版本、dataset metadata、采集日志 / 实际调用参数。
2. 当前 P1 训练入口与 runtime 对 joints/gripper 的具体单位约定：degrees/radians/raw servo range、归一化方向及计算方式、action 与 state 的语义。
3. `gripper.load`/`gripper.curr` 的原始意义和可能的数值范围，不允许写成已校准牛顿力。
4. 训练 transforms 中是否对 state/action 重复归一化、单位重复转换，或对 load/current 施加与关节同样不适用的换算。

优先采用有来源文件路径、字段值和哈希的实际证据。**不能根据数值看起来像角度或 range 就断言单位**。

如果历史采集设置无法恢复：允许提出独立的 `FRESH_TRAIN_SELF_CONSISTENT_CONTRACT`，仅限**使用新数据从头训练、部署时同一实际 follower 及严格对应的状态/动作配置**；需要列出能够验证的字段/transform/部署接口一致性，以及未证实的历史绝对标定。若连新训练与部署接口的单位一致性都无法验证，仍必须 BLOCK。

不得以“从头训练所以不需要管单位”为由跳过检查。未经证明不得把新模型 checkpoint 当作旧 P1 checkpoint 的无损 continuation。

输出：`unit_and_normalization_contract.md`、`calibration_evidence_manifest.csv`、`transform_roundtrip_checks.json`。

**B gate：** `UNIT_CONTRACT_VERIFIED` 或带明示局限且部署闭环一致性经过验证的 `UNIT_CONTRACT_FRESH_TRAIN_LIMITED`；否则 `UNIT_CONTRACT_BLOCKED`。

## 6. 工作包 C — 候选 90/15/15 split 审核与冻结

先复用 P4.0 的候选 `split_manifest.csv`，原则上保留其确定性生成逻辑：`default_rng(20261028)`，`ep113` 优先 Dev（初始手入镜，不视作无效 episode），其余候选分配满足 `Train90/Dev15/Final15`。**禁止因为某一集难/容易就基于模型性能重抽划分。**

审查：

- 120 ID 完整覆盖、无交集；episode 一致性与零文件重复；
- `trajectory_nearest_neighbors.csv` 中高相似对，配合初始画面/动作过程/采集 session 信息进行**有证据的近重复审查**；不能仅凭最近邻分数断言数据泄漏；
- 同一物理操作的重复编码或派生副本不得跨 split；相近空间位置但不同演示允许分别入 Train/Final；
- 对用户授权取消精确 15 区标注保持尊重，103/120 近似杯位检测不是 120 个 GT；
- 如发现可靠的 session/group 证据，优先提出组级调整；若确需改变候选划分，仅作 `PROPOSAL_SPLIT_CHANGE.md`，**不得擅自冻结新方案**；
- 记录 Dev ep113 和可能的其他手入镜对验证估计的影响。Final 不得因结果好坏再次筛选。

若未发现需改变 split 的实质证据，可**直接批准冻结 P4.0 候选**，形成 immutable `train90_episode_ids.json`、`dev15_episode_ids.json`、`final15_episode_ids.json` 和 `split_freeze_manifest.json`，记录 SHA256、seed、划分算法、源 revision、审查结论。

冻结前不调用 Final 做模型推理、prototype 选择、超参选择；冻结后 Final 只用于未来一次性独立评估。

**C gate：** `SPLIT_FROZEN` 或 `SPLIT_CHANGE_REQUIRES_REVIEW` / `SPLIT_BLOCKED`。禁止为了提高表面指标反复挑 split。

## 7. 工作包 D — Train-only prototype 与训练入口预检

P1 的 4 个 Cup Prototypes 必须来源于 Train90；明确：

- prototype 选取的候选来源及预先确定的取样策略；
- 不访问 Dev/Final 图像来建立、筛选或修订 prototype bank；
- 当前轮**不构建原型，不训练模型**，只检查准备流程与 Train-only 强约束；
- 固定 dataset revision、8D/6D schema、归一化合同、camera keys、split IDs、训练代码 commit 和 checkpoint lineage 规则；
- 检查 P1 的 deterministic 开关、seed 注入位置、输出/log checkpoint 目录，无需运行 optimizer。

允许一个纯数据加载 smoke test（例如有限个 batch、前向前的 shape/finite 检查）；如果所谓“前向”将执行随机初始化 DINO/ACT 导致显存很大，可不做。本轮**绝不跑训练更新**。

输出 `training_input_contract.yaml`、`prototype_source_plan_frozen.md`、`preflight_smoke_test.json`、`reproducibility_manifest.md`。

**D gate：** `TRAINING_PREFLIGHT_PASS` 仅在 A/B/C 和 Train-only prototype 来源约束满足时成立。

## 8. 不得擅自扩展范围

禁止：

- 开始 40k/80k/200k 训练、P1 pilot、多 seed、ACT baseline；
- 自动修正物理单位、自动猜测标定，或修改原始 HF dataset；
- 按近邻值自动删除 episode、按粗筛评论删除有手入镜片段；
- 为了 split 强行补 R01-R15 人工精确标注或重新采集；
- 用 Dev/Final 建立 DINO cup prototypes；
- 修改 P1 架构、8D 输入语义、视觉通路、ACT chunk 参数、原有 load/current；
- 将尚无证据的问题标成 PASS。

发现未覆盖问题须输出 `PROPOSAL.md`，而不是自行修改本 decision。

## 9. 报告要求与验收逻辑

全部成果存入与本 `decision.md` 同目录的 `result/`。最低交付：

```text
result/
  summary.md
  validation.md
  evidence_manifest.md
  schema_mapping_contract.md
  schema_mapping_validation.json
  unit_and_normalization_contract.md
  calibration_evidence_manifest.csv
  transform_roundtrip_checks.json
  split_review.md
  split_freeze_manifest.json              # 仅在确实冻结时存在
  train90_episode_ids.json               # 仅在确实冻结时存在
  dev15_episode_ids.json                 # 同上
  final15_episode_ids.json               # 同上
  training_input_contract.yaml           # 可为 BLOCKED 草案，须标状态
  prototype_source_plan_frozen.md         # 未冻结时标草案
  preflight_smoke_test.json
  reproducibility_manifest.md
  builder_review.md
  red_team_review.md
  PROPOSAL.md
```

`summary.md` 须单独记录：

```text
P4_0_1_COMPLETE = true/false
SCHEMA = PASS/BLOCKED
UNIT_CONTRACT = VERIFIED/FRESH_TRAIN_LIMITED/BLOCKED
SPLIT = FROZEN/CHANGE_REQUIRES_REVIEW/BLOCKED
PROTOTYPE_TRAIN_ONLY = GUARANTEED/BLOCKED
TRAINING_PREFLIGHT = READY/READY_WITH_DECLARED_LIMITATIONS/BLOCKED
TRAINING_RUN = false
ROBOT_RUN = false
RAW_DATASET_MODIFIED = false
```

其中 `READY_WITH_DECLARED_LIMITATIONS` 仅适用于单位/历史标定未完全可恢复、但**从头训练 + 部署接口自洽**经可复验的情况；不得用于 schema 或 split 未就绪。

## 10. Builder Review

核查是否严格 8D/6D、字段按名称映射、保留 load/current、排除 master、数据只读、单位与 transform 有可追溯证据、候选 split 审核合理、hash 可复现、Train-only prototypes、无训练与真机动作。

## 11. Red-Team Review

专门挑战：

1. 9D 的最后一列是否真的仅是 leader metadata？
2. `curr` 与 `current` 命名是否造成漏读/重复维度？
3. `use_degrees` / range 与 normalized action 是否混淆？
4. 是否对 dataset 和 model 各归一化一次？
5. 是否把历史 P1 checkpoint 的标定假设错误沿用到新数据？
6. 最近邻高相似是否只是杯子相近但正常的重复示教？
7. ep113 及其他手入镜是否被错误排除？
8. 是否在建立 prototype 时意外访问 Final？
9. 是否宣称完成逐帧动作 QC，而实际上只有 P4.0 粗筛？
10. 有没有将训练前提不成立却写成 READY？

## 12. 下一轮入口

仅当 `TRAINING_PREFLIGHT=READY` 或 `READY_WITH_DECLARED_LIMITATIONS` 且 Builder/Red-Team 独立审查接受后，才进入 **P4.1：P1_OCACT Train90 Single-Seed Pilot & Training Budget Selection**。

P4.1 才会决定 checkpoint cadence、预算上限、Dev-only 离线评估和 object-path collapse diagnostic。P4.0.1 不提前冻结 80k/133k/200k 为最佳步数。

## 13. 对执行端的最终指令

严格只做训练前合同解阻，**最小修改、保存证据、保留原数据**。无法确定就标 BLOCKED 并提交可执行 PROPOSAL；不得替科研决策方自行新增变量或改变训练目标。
