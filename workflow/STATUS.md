# LeRobot 项目当前状态

## 2026-10-13：G1 真机位置泛化已完成试验准备，但硬件阻塞

- G1 决策已读取；冻结 shortlist 为 `B3_ACT`、`P1_OCACT`、`R3_FILM`，不增加其它模型、不重训、不读取 Final。
- 已生成 81 次固定试验计划：Phase A 为每模型 3 次 nominal-center safety/function check（9 次，不计主成功率）；Phase B 为 12 strata × 2 repeats × 3 models（72 次）。
- 当前工作站虽检测到 RTX 4060 Laptop GPU，但未检测到 SO101 串口、top/wrist 相机，且三份冻结 checkpoint 不在本地；因此状态为 `BLOCKED_HARDWARE`，0 次真机 trial 已执行。
- 已生成 `experiments/position/g1_real_robot_generalization/index1/result/trial_manifest.json`、`preflight.md`、`validation.json`、空结果表和资源占位文件，以及 `scripts/prepare_g1_trial_manifest.py`。
- 接通真实硬件并完成相机/串口/冻结 checkpoint 校验后，才能由用户现场确认启动 Phase A；本轮未启动真机。

## 2026-10-13：G0 位置泛化筛查完成，shortlist 已冻结

- 固定 Formal4 Dev 12 episodes 对 9 个既有 100k 候选做统一离线筛查；未训练、未改架构、未读取 Final、未操作真机。
- 使用 P0.1 冻结的 x/y tertiles 与 Edge top-25% 分组，报告 h0 teacher-forced L1、Left/Center/Right、Near/Center/Far、Edge/Non-edge，以及 first-1/5/10/full chunk 的预测分布差异。
- 位置泛化排序（位置稳健性优先、无横向 dim0 硬门槛）：B3_ACT、B2_ACT、P1_OCACT、P3_FIXED_SPATIAL、P4_JOINT_SPATIAL、R1_RELATIONAL、B0_ACT、R3_FILM、R2_ACTION_LEVEL。
- 候选 shortlist 已冻结为：`B3_ACT`（ACT baseline）、`P1_OCACT`（最佳 object-aware h0 误差）、`R3_FILM`（结构不同且 Left/Right 首动作分离诊断最强，但整体误差较高）。
- G0 的 full-chunk 后续步没有伪造 ground truth：Formal4 单样本只暴露 h0 target，5/10/full 仅作为预测分布诊断；Dev matched-state 只有 2 对，明确标为 underpowered。
- 结果：`experiments/position/g0_generalization_screening/index1/result/summary.md`、`candidate_ranking.json`、`center_bias.json`、`stratified_metrics.json`、`proposal.md`；执行脚本：`scripts/eval_g0_generalization.py`、`scripts/complete_g0_report.py`。
- 当前仅达到 `POSITION_GENERALIZATION_SHORTLIST_READY=TRUE`，下一步必须由用户 + ChatGPT 新建 G1 真机泛化 decision；本轮不自动进入真机。

## 2026-10-13：H0 横向控制信号审计完成（H0-C / unresolved）

- 已在远程 GPU 服务器完成只读 `H0_HORIZONTAL_CONTROL_SIGNAL_AUDIT`，固定 Formal4 Train/Dev 划分，未使用 Final、未训练新模型、未进行真机测试。
- 按闭爪前 reach/fine-alignment 规则选出 Train 1,501 帧、Dev 556 帧；无可用 SO101 URDF/Jacobian，采用 Train-only 五关节 ridge 经验投影，并在冻结映射后复评既有 token-transplant 结果。
- H=1/3/5/10 的 Dev 方向准确率分别为 0.401/0.448/0.475/0.514，Pearson 分别为 -0.171/-0.142/-0.102/0.041；Train 状态匹配 33 对，但独立 Dev 仅 2 对，不能构成稳定验证。
- 结论为 `H0-C / unresolved`：不能把“起步向中间”归因于单一 action dim0；当前多关节映射也未显示可独立复现的横向控制信号。停止基于该假设的架构扩展。
- 证据与脚本：`experiments/position/h0_horizontal_control_audit/index1/result/summary.md`、`model_re_evaluation.json`、`horizontal_mapping.json`、`dev_mapping_validation.json`、`scripts/audit_h0_horizontal_control.py`、`scripts/complete_h0_report.py`。
- 下一步仅允许改善诊断边界：取得可信 SO101 运动学标定，或采集带显式 reach 阶段且横向变化充分的新轨迹后重跑 H0；不得据此直接启动新训练、Final Test 或真机测试。

## 2026-10-12：N0 三候选 preflight 通过，已排队正式训练

- `N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH` 的 R2 action-level、R1 relational、R3 FiLM 三个候选均已完成远程 Formal4 2-step smoke；三路均 exit=0、loss 有限，且分别生成了对应结构的训练参数（action spatial MLP、relational projection、FiLM generator）。
- 已冻结并记录训练/Dev/Final episode 列表、seed=1、100k steps、每 10k 保存；当前框架使用 `eval_freq=0`，Dev 评估在训练后单独执行，不能把它报告成训练中每 5k 自动评估。
- 当前服务器只有一张 RTX 5090；独立的 T3 B3/P4 仍在运行。因此已上传 `scripts/run_n0_overnight_queue.sh`，等待 T3 两个 `train.done` 后按 R2 → R1 → R3 顺序串行启动，避免 GPU 争抢和结果污染。正式 N0 尚未开始。
- 本批次仍禁止读取 Final Test、禁止真机测试；训练完成后只对固定 Dev 做 C0–C5 与 T0.2 token-transplant，最后生成批次排名，不自动宣布赢家。证据清单见 `experiments/position/n0_overnight_batch/index1/result/training_manifest.json`。
- 远程数据盘清理已完成：删除 11 个已结束 smoke/preflight 的 `checkpoints/`，保留其 metrics、日志和运行记录；未触碰 Formal4 数据、正式 checkpoint、T3 当前运行目录或 N0 队列。数据盘从约 50 GB 已用降至约 43 GB，剩余约 58 GB。

## 2026-10-13：N0 正式训练完成，固定 Dev 评估已启动

- R2、R1、R3 均已完成 100,000 steps，并生成 `checkpoints/100000/pretrained_model`；三路 `train.done` 存在，训练日志未检出 Traceback、NaN、Inf。
- T3 B3/P4 也已完成；本轮仍不读取 Final Test、不进行真机测试。
- 远程 detached `screen n0_eval` 已启动固定 Dev 12 episodes 的 C0–C5 评估，并在每个候选后执行 T0.2 token-transplant；结果写入 `experiments/position/n0_overnight_batch/index1/result/`。评估结束后再生成批次排名，不自动宣布赢家。
- 另有 detached `screen t3_eval` 等待 N0 评估结束，随后评估 T3 P4 的固定 Dev 与 token-transplant；两套评估均不读取 Final Test。

## 2026-10-13：N0/T3 固定 Dev 与机制评估完成

- N0 Dev first-action L1：R2=`0.135205`、R1=`0.143740`、R3=`0.126878`。R2 的空间反事实最大变化约 `0.109`，但 whole token-transplant 横向方向一致率 `0.000`；R1 横向 `0.100`、纵向 `1.000`，仍未达到横向机制门槛 `0.70`；R3 空间变化约 `4.2e-06`，基本未使用空间输入。三者均不能进入 Confirmation 或真机。
- T3 对照报告已补齐 B3/P4：B3 Dev L1=`0.085466`，P4=`0.126716`；P4 空间反事实最大变化仅 `3.71e-05`，whole token-transplant 横向/纵向均为 `0.000`，机制判定 `FAIL`。
- 结论：本轮没有满足横向空间控制门槛的候选；不能因为 R2/R1 的纵向响应或 R3 较低 Dev L1 就宣布成功。下一步应冻结本轮证据，停止绝对坐标架构扩展，另建新的相对目标—状态或融合抑制 decision；Final Test 与真机仍禁用。

## 2026-10-12：T3 联合 State-Spatial 条件 preflight 完成，等待正式训练

- 按 `T3_JOINT_STATE_SPATIAL_CONDITIONING` 实现唯一结构差异：P4 将 8D robot state 与 3D `(x,y,confidence)` 拼成 11D，经 `Linear(11,128)-LayerNorm-GELU-Linear` 生成唯一 state token；B3 保持 OC-ACT 原始 8D state projection。
- P4 不创建 `explicit_spatial_beta`、`explicit_spatial_scale` 或额外 spatial residual；P4 smoke checkpoint 已确认存在 `joint_state_spatial_projection` 且不存在 beta/scale。
- B3 2-step smoke 通过；P4 首次 smoke 暴露提取联合空间特征条件遗漏，已完成纯工程修复，P4 第二次真实 Formal4 2-step smoke 通过，loss 有限并保存 checkpoint。
- 已同步 T3 训练、Dev counterfactual、汇总脚本与 manifest；正式 100k 训练尚未启动，仍禁止 Final Test/真机。
- 已通过 preflight 验收并启动远程 screen `t3_b3`、`t3_p4` 并行 100k；约 10 step/s，预计 2.5–3 小时。训练日志当前无 NaN/Inf/Traceback。

## 2026-10-12：T2 固定尺度空间条件 Discovery 完成（FAIL）

- B2/P3 两路 100k 均 exit=0，均有 `100000` checkpoint，无 NaN/Inf/Traceback；P3 checkpoint 确认固定 `explicit_spatial_scale=0.5` 存在，trainable `explicit_spatial_beta` 不存在。
- 固定 Dev 12 episodes 离线结果：B2 首步 L1=`0.088282`，P3=`0.097128`，P3 明显劣于 B2；P3 S0–S4 最大 Dev-L1 变化=`0.002868`，说明路径数值上已活跃但不是可靠坐标控制。
- P3 token transplant whole directional consistency：horizontal=`0.10`，longitudinal=`0.55`（纵向配对子集为 `0.90`，但横向核心门槛失败）；横向远低于冻结 `0.70`，机制判定 `FAIL`。
- 结果见 `experiments/position/t2_fixed_scale_spatial/index1/result/summary.md`、`checkpoints.json`、`mechanism.json`、`dev_eval/`、`transplant_P3/`。按决策不得继续单纯增大 lambda，不进入 Confirmation、Final Test 或真机；下一步需另建单变量结构性 decision（如 fusion suppression 或 relative target-state representation）。

## 2026-10-12：T2 固定尺度空间条件训练已启动

- 按 `T2_FIXED_SCALE_SPATIAL_CONDITIONING` 只改变显式空间残差的尺度：P3 使用固定、不可训练 `lambda=0.5`；B2 为冻结的 OC-ACT 基线，显式路径关闭。
- B2/P3 均通过真实 Formal4 2-step smoke，loss 有限并成功保存 checkpoint。P3 配置确认 `explicit_spatial_fixed_scale=0.5`，未创建可学习 beta。
- 远程 RTX GPU 已通过 screen 并行启动 `t2_b2` 与 `t2_p3`，目标各 100000 steps、每 10000 steps 保存；当前禁止 Final Test 和真机。
- 训练记录与冻结 split manifest 位于 `experiments/position/t2_fixed_scale_spatial/index1/result/`；训练结束后只在固定 Dev 12 episodes 做 S0–S4 离线评估和 token transplant 机制检查。

> 更新时间：2026-10-11（M0 OC-ACT架构工程完成，`ARCHITECTURE_READY=TRUE`；正式训练前仍需P0 prototype calibration）。这里保存当前快照，不保存完整历史。历史过程见 `../docc/日志/工作日志.md`；长期执行规则见 `lerobot_research_workflow.md`。

## 当前阶段

- Model M0：`M0_OC_ACT_ARCHITECTURE_BUILD` 全部 M0-G0--G9 PASS，`ARCHITECTURE_READY=TRUE`。原 ACT baseline 默认路径不增加参数；OC-ACT 在 top camera 上增加 frozen pretrained DINOv2-S、冻结 prototype cosine matching、1 global + 2×2 local 共5个包含 semantic/x/y/confidence 的 object tokens、adapter、pre-encoder cross-attention 与零初始化 scalar gated residual。Formal4 episode 0真实双相机/8D NOMASTER/6D action smoke中，DINO backward grad 全None，adapter/cross-attention/gate/ACT head梯度非零，`alpha=0`对baseline max abs diff=`0.0`，object zero/shuffle/spatial-shift均改变hidden，ACT与OC-ACT checkpoint往返diff均为`0.0`。本轮未正式训练、未读dev/final-test、未执行真机；smoke prototype明确为`NOT_FOR_FORMAL_TRAINING`。证据见 `../experiments/model/m0_oc_act_architecture/index1/result/summary.md`。下一步只能由新decision进入P0 prototype calibration。
- Formal4 D1：`D1_FORMAL4_TRAINING_READINESS` 全部D1-G0--G6 PASS，`TRAINING_READY=TRUE`。已建立60集/28,739帧的8维 `kind_merged_nomaster`，全量state删除主臂通道逐值一致、action不变、视频未重编码；60/60 QC为VALID，稳定闭爪窗口range中位0.913°/最大1.217°且无>1° reopen event；冻结36/12/12 split（各组kind1/kind2为18/18、6/6、6/6），仅train 17,243帧计算normalization。`load=-500`饱和314帧/25集，后续Force建模必须显式处理。证据见 `../experiments/data/d1_training_readiness/index1/result/summary.md`。正式训练仍需新的模型decision，本D1不得自动进入训练。
- Data 主线：`D0_NEW_DATASET_V1` 已生成 seed=`20261008` 的冻结采集表（5 rounds × 12 strata = 60 episodes），并完成采集 schema 与 recorder 代码审计。当前机器未检测到机械臂串口或摄像头，故状态为 `BLOCKED_HARDWARE / READY_FOR_COLLECTION`，真实采集仍为 0/60，D-G0--D-G5 均未通过。连接 leader/follower 与 top/wrist 双相机后，必须先完成单 episode smoke 并核验 `state/action/load/current/video`，才可进入正式采集；QC PASS 前禁止生成 split 或训练模型。结果见 `../experiments/data/d0_new_dataset_v1/index1/result/summary.md`。
- Force 主线：`F0_SIGNAL_CALIBRATION` 已按 `../experiments/force/f0_legacy_calibration/index1/decision.md` 完成，状态 `PARTIAL / OBSERVED`。原始 Formal3 的 `gripper.load` 在闭合与释放转换中有60/60跨集一致的符号结构，15帧窗口比单帧更可分；但 stable grasp / lift-place 的单帧 `load/current` 高度重叠，`current` 仅18个离散值且大量为0。expert action 在 lift/place 中重新张开中位4.11°，15/60集达到至少5°，说明旧数据本身可能促成真机 Lift 后重新张爪。S1/S2与S4/S5仍无法可靠细分。结果见 `../experiments/force/f0_legacy_calibration/index1/result/summary.md`；未授权 F1、训练或真机。
- Formal3 主线：旧 reset-pair 与修复候选 reset-preroll 均为`FAILED / DIAGNOSED / REAL_ROBOT_BLOCKED`。B1 reset-preroll 100k训练健康、评估完整，但100k仅Gate3/5/6通过：train48首步方向11/48（需≥33）、dev12为3/12（需≥9）、train48 `|bias|=4.02°`（需≤2.5°）；幅度比1.133、任务区间arm MAE比0.849和完整性通过。完整结论见 `../outputs/formal3_preroll_p4_100k_20260929/summary.md`。
- B1没有在100k解决真实复位frame0方向问题，且20k–100k无节点接近方向Gate，因此按冻结选择矩阵淘汰，不续训、不做seed变体、不进入真机。主线回退到已通过60/60离线闭环回放的B2 `DET 100k + kick-start`；下一阶段是部署dry-run与安全Gate，而不是继续训练。
- Formal3 四模型 100k 与部署资产已完成；现阶段不以既有 teacher-forced 指标解除 reset-pair 的真机阻塞。
- C50 真机主线：`P6` 首轮探索性测试已完成，但没有按杯位配对，也没有判定表，不能作为正式 round 2 成功率对照。
- C0 显式杯框条件支线：`P3 FAILED / DIAGNOSED`。10k检测坐标正确、坐标MLP确有更新，但策略从1k起始终几乎不使用独立box token；完整置零/反转、全horizon与坐标-only反事实均确认，不晋升100k。
- C1/C2 显式杯框条件对照：C1 已进入 Formal3 P4 并完成 100k，但反事实确认坐标对动作的因果影响过弱（coordinate-only pan span `0.00561`）；状态为 `CONFIRMED_BLOCKER`，不得通过续训或无约束放大推进。C0/C2 仍冻结。
- 水面关键点支线：`P2`，代码与真实数据工程联调完成，人工标签不足。
- 完整透明杯 MASK 支线：暂停晋升，现有画质和遮挡下标签语义/时序稳定性不足。

## 已完成

- 新机械臂数据采集SOP已冻结为`docc/指南/透明水杯_SO101_数据集采集指南.md`：单episode采用短复位（0.3–0.5秒）→直接定位→连续闭爪→稳定保持（0.2–0.5秒）→抬升搬运→接触后释放→立即结束；明确`action=限幅后实际发送目标`、`state=从臂实际反馈`、主臂信号仅诊断，并配有可编辑draw.io时间轴和现场判废Gate。
- 建立 `formal1_C50_nomaster`：50 episodes、35,900帧、8维state、6维action；移除部署不存在且与动作标签泄漏的`master_gripper.pos`。
- ACT-NM、DET-NM、I0-NM、I1-NM四组同预算100k训练完成，退出码和检查点完整。
- 固定val10的100k离线评估，以及20k/40k/60k/80k/100k趋势捕获完成。
- 四个100k模型已发布到`QYyyyyyyy/C50_NOMASTER_{ACT,DET,I0,I1}_100k`。
- 水面关键点可见性门控、Gaussian热图分支、旧checkpoint兼容和真实双相机smoke通过；尚未正式训练该分支。
- 新增 C0：从 FCOS 预测中解码 `[cx,cy,w,h,confidence,visible]`，阻断动作梯度后编码成独立 Transformer token；默认关闭，不改变旧模型路径。MASK、水面、I0/I1 和 Diffusion 本轮冻结。
- 新增 `scripts/smoke_c0_explicit_box.py`：原生 C0 严格加载；旧纯 DET 只能进入明确标记的接口 smoke 模式，并记录随机初始化项，避免把兼容复制误报为 C0 效果。
- 新增 C1/C2 与三模式统一 10k 合同、初始化哈希检查、并发资源预检、坐标反事实评估和训练编排。321 个共有张量逐元素一致；三路 batch 8 并发 2-step 均正常退出且真实批次 smoke 通过。

## 当前冻结结论

- ACT-NM在teacher-forced离线复现精度上四组最好：h0 arm `2.6008`，gripper `1.6170`。
- I1-NM是检测组综合最好，但相对ACT-NM没有通过预注册筛选线；I1抓取阶段优势受ep16单集明显影响，post阶段弱于ACT。
- I0/I1相对纯DET有稳定收益；唯一从20k到100k未反转的早期关系是I1优于DET。
- 20k–40k出现关键排名反转，10k/20k短训不能可靠预测100k终点排名。
- 以上都是离线证据，不能替代真机闭环成功率和扰动鲁棒性。
- round 2 探索性观察为：四模型起步先向工作区中部，I0/I1 后续偶尔逐渐追杯；闭爪→抬起仍不流畅，抬起后策略可能重新张爪。缺少杯位/阶段映射，以上保留为现场定性证据，不冒充配对统计。
- C50 首帧覆盖五个离散方位（left 8、left_front 15、front 3、right_front 15、right 9），具备左右监督但不是连续随机杯位。C0 第一轮仍保持 C50 不变以隔离架构变量。

## 下一步

0. Model M0 已完成并停止。当前 smoke prototype 禁止用于正式训练；由用户与 ChatGPT 验收 M0 `result/` 后，创建独立 `P0 Prototype Calibration` decision，冻结 train-only/reference image、杯区域标注、prototype数量与可视化Gate。P0通过前不得启动ACT/OC-ACT正式训练。
0. Data D0：连接真实采集硬件并确定 leader/follower 串口及 top/wrist 相机路径；严格按 `../experiments/data/d0_new_dataset_v1/index1/result/collection_runbook.md` 先录制 1 条 smoke。仅当 schema、实际发送 action、双视频和力信号核验通过，才按冻结表采集60条。
1. Force F0 已完成并停止。只有用户与 ChatGPT 验收本轮 `result/`、生成并批准新的 F1 `decision.md` 后，Codex 才能设计或训练最小 Force Module；当前不得自行进入 F1。
1. ✅ h0动作辅助权重线已按合同完整执行并停止：实现（默认关闭）+本地5/5测试+服务器真实batch smoke通过，2k双候选train48诊断完成，预注册筛选BOTH FAIL（S1 |bias| 5.45/5.46 vs 需≤3.86；S2 cosine+ 10/8 vs 基线15/48更差），不延长训练。
2. ✅ 阶段B选择已冻结：B1 reset-preroll 100k训练/评估完整但overall FAIL（Gate3/5/6通过，Gate1/2/4失败），按合同淘汰；B2 kick-start全量离线PASS（60/60接管、闭环arm MAE train48/dev12为1.63°/3.29°），`DET 100k + kick-start`成为唯一部署候选。规划文档：`docc/方案与规划/Formal3_起步方向修复_完整规划_2026-09-29.md`。
3. 下一步只做B2部署准备：把离线mode B固化为显式`HOLD/KICK/NORMAL/ABORT`状态机，完成无电机dry-run和动作边界/超时/相机失联测试；随后采集夹爪`pos/load/current`标定并通过安全Gate。任何真机动作仍需用户另行授权。
4. 可在上述固定离线合同稳定后实现受限的 `robot-autoresearch` 原型，只允许改候选配置并执行短训/固定评估/保留回退；禁止修改数据语义、启动正式100k或操作真机。
5. 真机前仍需夹爪 `pos/load/current` 标定与部署安全 Gate；正式 round 2 必须恢复杯位配对、判定表和完整阶段记录。

## 当前阻塞与限制

- reset-preroll 10k仅Gate4/6通过：真实复位frame0方向0/12、pan 5/12、幅度比中位数2.672；共同任务区间总arm比1.197虽贴线，但approach为1.399，故Gate5亦失败。2k也仅4/12方向正确，五个节点没有可晋升候选。阶段 A 已重新解释该失败：10k 预算不足（原始结构 10k 同样方向错误 8/48，100k 才收敛到 42/48）；固定偏置在 100k 首运动行处仅 1.5–1.8°，不再是表征上限证据。
- train48根因诊断排除“仅dev协议泛化失败”：10k在训练集也仅17/48方向、24/48 pan，幅度比2.416；2k/8k方向亦仅15/48、18/48。h0预测在43/48训练集和12/12 dev中仍最接近truth chunk第0步，故不是简单错取后续horizon；主要表现为跨关节固定偏置，train均值`pred-target=[+0.53,-10.43,+2.49,+10.53,-7.37]°`。该偏置的事后校正只用于诊断，禁止作为部署补丁。
- reset-pair 10k 的预注册 Gate 仅 2/6 通过：复位方向、pan、幅度和裁剪起点交叉泛化均失败，因此不得进入100k或正式真机。
- 超调在dev12中同时与协议分组和运动起点 `S` 相关；即使完成全60集协议审计，现有观察数据仍不能声称唯一根因。
- 全60集协议审计已确认 `S` 范围有42--99重叠且组间标准化差异仅-0.196；复位pan与杯位分布漂移更大。当前仍不能从dev12辨认唯一因果变量，但不再把`S`与协议组视为完全混淆。
- C0 P3的检测器与坐标MLP已学习，但动作策略基本忽略显式box token；当前C0结构不得进入100k。10k结果只用于工程诊断，不与正式100k模型排名。
- 夹爪接触阈值尚无真机 `load/current` 标定数据；不得复制外部阈值或直接启用自动收紧。
- 水面关键点现有可见人工点数量不足，不启动正式100k训练；遮挡中心不得插值或使用SAM漂移中心代替。
- 固定val10已用于正式比较，后续不应继续把它当开发调参集。
- 服务器大型raw输出受租用实例生命周期影响，关键汇总已同步本地，但仍需保持证据清单。

## 2026-10-11：P0 prototype calibration 结果

- P0 已按冻结 decision 执行：8 个 train-only reference episode、12 个 train-only check episode，远程 RTX 5090 上使用冻结 `dinov2_vits14`，未读取 dev/final-test，未训练正式模型，未操作真机。
- 原型峰值定位通过（12/12 在粗杯框内，cup-vs-corner proxy=1.0，inside/outside ratio=2.786），但 M0 的未加温度 softmax pooling 产生近似固定的画面中心 soft center：中位误差 78.28 px、P90 97.77 px、inside softmax mass 0.0508。
- 因此 P0 状态为 `FAIL / DIAGNOSED`；`cup_prototype_bank_v1.pt` 仅作诊断候选，`formal_training_eligible=false`，不得用于正式 ACT/OC-ACT 训练或部署。完整证据见 `../experiments/position/p0_prototype_calibration/index1/result/summary.md` 与 `validation.json`。
- 下一步必须新建并冻结一个 decision，专门验证空间聚合/目标 token 修复（例如温度或局部归一化），然后重新执行 train-only P0；不得用 dev/final-test 调参。

## 2026-10-11：P0.1 spatial pooling fix 结果

- P0.1 严格复用 P0 的 4 个 prototype、8 个 reference、12 个 train-only check 帧和所有坐标/粗框定义，只比较冻结的 `tau=1.00/0.20/0.10/0.05/0.02`；未读取 dev/final-test，未训练正式模型，未操作真机。
- `tau=1.00` 的 soft-center 中位/P90 误差为 78.28/97.77 px；温度缩放后 `tau=0.20` 为 24.75/32.73 px，Pearson x/y=`.964/.931`，Spearman x/y=`.888/.895`，peak/corner 两个 Gate 均为 1.0，且有效 patch 数均值 120.1，无 one-patch collapse。
- 按“满足全部 Gate 的最大 tau”规则选择 `tau=0.20`，P0.1 状态为 `PASS`。配置见 `../experiments/position/p0_1_spatial_pooling_fix/index1/result/object_pooling_config_v1.json`；该结果只证明空间表征校准通过，不自动授权正式训练。
- 下一步必须新建正式训练 decision（baseline 与 OC-ACT 的训练合同、固定 split/预算和离线 Gate）；Codex 不得从 P0.1 自动启动训练。

## 2026-10-12：T0 ACT vs OC-ACT Discovery 已启动

> 复核更正（2026-10-12）：B0 训练和开发集评估有效；首轮 P1 虽然保存了 `oc_act=True` 配置，但远程训练代码缺少 `object_conditioning/__init__.py` 与 `modeling_act.py` 注入实现，故首轮 P1 仅作无效工程记录，不纳入科学结论。已补齐代码并通过构造级 smoke test，P1 正在以 `P1_ocfix2_s1` 重新训练；Final Test 仍未使用。

> T0 结果（2026-10-12）：修复后 P1 完成 100k steps、退出码 0；固定 Dev 评估全部 10 个检查点成功。B0 最优 40k，L1=0.241336；修复后 P1 最优 100k，L1=0.230687（相对下降约 4.43%）。位置分层中 P1 误差整体低于 B0；置零目标分支使动作变化 0.07452，批内打乱 0.02036，但空间交换仅 0.00103。当前判定为 PARTIAL：注入有效且 Dev 有改善，但绝对空间敏感性仍弱，需用户确认后才可进入 Final Test。

> T0.1 结果（2026-10-12）：在冻结 P1-100k 与 12-episode Dev 上完成 D0–D9。D1/D2 证明 object branch 被使用（h0 arm delta=0.03832/0.01141）；D3/D4 显式 x/y 干预几乎无效（0.00044/0.00002）；raw token 层的 D5/D6/D7 local layout 干预有效（0.00429/0.01901/0.01041）；D8/D9 显示 global 与 local 均贡献，local-only 更大。当前 T0.1 判定为 PARTIAL：模型使用 object/local layout，但没有直接显式坐标机制证据；未运行 Final Test、真机或重训。

- T0 按冻结 decision 启动 B0 ACT 与 P1 OC-ACT 两路并行 Discovery，均为 seed=1、36 train episodes、batch=4、chunk=100、n_action_steps=1、100,000 steps、20k checkpoint；Final Test 和真机禁止。
- 新服务器 `connect.westd.seetacloud.com:23089` 已完成 Formal4 60 episodes / 579,892,449 bytes 数据校验；远程安装项目依赖并完成 B0/P1 两步真实 batch preflight（均 exit=0）。P1 使用冻结 P0 bank 与 P0.1 `tau=0.20`，温度已接入 tokenizer。
- 当前 screen：`t0_train_b0`、`t0_train_p1`；训练约 10 step/s，预计约 2.5–3 小时。当前仅有 train loss/health，尚无 Dev 或 Discovery 结论；训练完成后才运行允许的 Dev 离线评估，Final Test 仍不读取。

## 2026-09-27：Formal3 四路线训练准备

- `formal3_kind_merged_nomaster_fit48` 已发布到 Hugging Face 数据集 `QYyyyyyyy/formal3_kind_merged_nomaster_fit48`：60 episodes、35,917 frames，包含双相机视频、NOMASTER 数据、训练48集统计、top检测标注、固定48/12划分和最终人工阶段标签。
- 四路线合同位于 `outputs/formal3_mixed_p4_20260927/experiment.md`，状态为 `PREPARED`：纯 Diffusion、DET基线、C1 state-box显式坐标注入、DET+phase(weight=0.10)。C1沿用已通过Gate的 state token加法，不使用未通过Gate的C2 action residual。
- 四路 batch=8 的真实数据并发2-step预检全部退出0，无OOM/NaN。用户确认后，正式100k训练已在服务器 `screen -S formal3_mixed_p4` 启动；四个主训练器均越过首批步骤，总显存约27.5 GB，预计以最慢ACTDet计约10–11小时。

## 2026-09-28：C1显式坐标弱使用阻塞

- C1-100k在Formal3快速反事实中确认“检测准但动作弱使用坐标”：左/中/右框可见、顺序与方向均正确，但coordinate-only pan跨度仅`0.00561`（要求`>=0.05`），zero/reverse最大影响为`0.00973/0.00342`，box-effect Gate失败。
- 该问题记为`CONFIRMED_BLOCKER`；不得通过延长原C1训练、单纯增大检测loss或无约束放大残差处理。下一轮先做带反事实敏感性损失与特征尺度监控的5k–10k短训，过Gate后才允许100k。详见`../outputs/formal3_mixed_p4_20260927/c1_weak_conditioning_diagnosis.md`。

## 2026-09-28：Formal3部署资产发布

- 四个100k部署目录已发布并完成Hub端文件/大小校验：`QYyyyyyyy/F3_P4_{DIFFUSION_BASE,DET_BASE,DET_C1_STATE,DET_PHASE_W010}_s1000`。只发布`pretrained_model`部署文件，不包含优化器或中间checkpoint。
- 真机只需从`QYyyyyyyy/formal3_kind_merged_nomaster_fit48`下载`meta/**`；checkpoint自身pre/postprocessor携带训练归一化参数。指南见`../docc/指南/Formal3_四模型真机测试指南_2026-09-28.md`。

## 2026-09-29：h0 动作辅助权重 2k 筛选 FAIL

- 已实现默认关闭的 ACTDet h0 动作辅助权重（`use_h0_action_aux=false`、`h0_action_weight=1.0`）：仅给 chunk 行 0 的 L1 损失加权，`h0_l1_loss` 监控在行加权前取值（跨 W 可比），旧 checkpoint 兼容、旧路径逐位不变。本地定向 5/5 通过、项目回归 68/68 通过。
- 损失恒等式 `loss_on−loss_off == (W−1)×h0_l1_loss/chunk_size` 在服务器真实 batch smoke 中精确成立（W3/W10），梯度有限非零；双路 2-step 预检 PASS（checkpoint config 记录 h0 flags）。
- 双路 2k 训练（W=3/W=10，其余与 preroll 10k 同配置、仅 top 相机）20:35 完成 exit=0；train48 frame0 双路评估 48/48 npz exit=0；三组 bias 分析完成。证据链在 `outputs/formal3_h0_aux_p3_20260929/`（experiment.md/screening.{json,md}/diagnose_train48/）。
- 预注册筛选判定 **BOTH FAIL，按合同立即停止**：S1 |bias| arm W3=5.446°/W10=5.455° vs 基线 5.513°（目标线 ≤3.859°）基本未动；S2 方向指标反而更差（cosine+ 10/48、8/48 vs 基线 15/48；pan 25/24 vs 26）。closest_truth_row 0 为 45/45/46、幅度比 1.83/1.78 靠近 1 只是微弱向行 0 靠拢迹象，不改变判定。
- 结论：2k 行 0 加权不能消除跨关节固定偏置——该偏置是数据主导模式，非 horizon 错选。h0-aux 线终止；下一候选回到数据侧（pre-roll 短序列/窗口缩短/动作拍平），待用户拍板。
- 运维：AutoDL 数据盘出现瞬时 ENOENT（numpy 原子写 rename 后元数据滞后，W10 npz 已存在却报 FileNotFoundError），链式诊断中断一次，改绝对路径+重试后补跑成功；后续服务器分析脚本统一绝对路径+重试。

## 2026-09-29：起步方向修复——阶段 B（preroll 100k + kick-start）已授权并执行

- 用户授权 B1+B2 并行。B1：`scripts/train_formal3_reset_preroll_100k.sh`（与 10k 同配置，仅预算/存点/输出名变化，复用 preroll P2 preflight），screen `formal3_reset_preroll_100k`，输出 `F3_P4_DET_RESET_PREROLL_100K_s1000`，每 20k 存点；与 B2 并行时 ~1.3 步/s，ETA 约 18–22h，完成后按预注册 Gate 评五节点（frame0 方向/幅度/偏置 + time-trim 交叉）。
- B2：新增 `scripts/replay_kickstart_offline.py`（mode A open-loop 从 r*；mode B kick-start 混合时钟闭环：行 0 <5° 则执行 r* 行，首次 kick 后时钟跳到 S+1 以录制帧代理后 kick 状态）。smoke ep7/11：**kick 仅 1–2 次模型即接管**（首个运动观测行 0 delta 6.8/7.4°），闭环 100 步 arm MAE 1.67/3.37°；dev12 ep26 open-loop 漂移 10.2° vs 闭环 3.29°。全量 train48/dev12 双屏并行（约 70/20 分钟）。
- 用户决定 **diffusion 出局**：4060 8G 部署带不动，远程推理不满足闭环控制时延；阶段 A 已证其 100k 采样坍缩（pairwise 0.98）。DET 为唯一部署候选，DIFFUSION_BASE 仅留离线对照。
- 运维：服务器换端点 `connect.westb.seetacloud.com:33228`（5090 32GB）；SFTP 不可用，文件传输走 exec tar/base64；phaseA 全量产物已同步本地。
- **B2 kick-start 全量 PASS（60/60 集）**，证据本地 `outputs/formal3_direction_fix_20260929/phaseB_kickstart/`：mode B 混合时钟闭环——train48 首踢方向 42/48（cos 0.615）、3.25 次 kick、**闭环 100 步 arm MAE 1.63°**、接管 48/48；dev12 首踢方向 12/12（0.845）、2.17 次 kick、**arm MAE 3.29°**、接管 12/12。open-loop（mode A）5.51/5.65° 对照证明闭环重规划是关键。**"现有 DET 100k + kick-start"成为已证部署路径**（离线回放证据，真机仍需部署+安全 Gate）；preroll 100k（B1）降为训练侧备选/对照，Gate 通过后二选一。
- 用户训练规划决策：先单路 B1 看证据再定 4 路化（本机 1×5090，4 路 65–75h）。
- 服务器清理（用户确认）：死线 F3_P3 四线 10k + h0-aux 2k 双路 + DIFFUSION_BASE 100k（HF 已发布）+ 旧 tar.gz，共约 30G；metrics.csv 留证 `outputs/cleanup_20260929/metrics/`；磁盘 25G→54G 可用。保留 F3_P4_DET_BASE/C1_STATE/PHASE_W010、campaign 证据、数据集、lerobot-main 旧副本。

## 2026-09-29：起步方向修复——阶段 A 零成本诊断（根因修正）

- 授权范围：阶段 A 零成本诊断三项（diffusion 100k reset frame0 采样 N=16 看多样性、DET 100k 首运动方向、pan 符号 vs 杯框 cx 信息流审计）。新增 `scripts/eval_reset_frame0_samples.py` + `scripts/analyze_phaseA_direction_info.py`，服务器 GPU 并行执行，62 文件全部 exit=0；证据在服务器 `outputs/formal3_direction_fix_20260929/phaseA/`（实例离线，raw 待同步）。
- DET 100k frame0 chunk 首运动行 r*≈62.5：方向 cosine+ 42/48 train48、12/12 dev12；r* pan 与杯框 cx 相关 −0.753/−0.667（条件化通路通）；幅度比 1.190/1.091、|bias| 1.754/1.484°。
- DET ORIGINAL 10k frame0：r*≈0（行 0 即动）但方向 8/48、0/12、框相关 −0.233——方向需要 100k 级预算；10k 固定偏置=预算不足的均值回归表现，非表征上限（"多模态回归塌缩"假设修正）。
- Diffusion 100k before_S：16 条采样覆盖正确方向 42/48、12/12（best cos 0.832/0.994）但 pairwise cosine 0.978/0.981≈无多样性——目标分布近乎单峰，BeT 模式头降级为后备。
- 根因定论：唯一阻塞=计划结构（100k 模型正确首运动在行 62–66，部署 n_action_steps=1 只执行行 0 → 自锁）；preroll 结构已解自锁，只差预算。两条路径均已进入执行：路径 2 kick-start 离线 replay 全量 PASS；路径 1 preroll→100k 训练中（约 19h）。

## 2026-09-30：B1 训练中巡检 + 20k 趋势/100k Gate 无人值守编排

- 服务器巡检：B1 已 ~10k/100k（10%），10K 节点 loss 0.768（10k 短训同节点 0.748，量级健康）、grad norm 44.4，无 NaN/Inf/Traceback；~1.6 step/s，ETA 今晚 ~16:00。五个 screen 存活：训练、posttrain eval 等待器、trend20k、summary、jupyter。
- 资源画像：208 核只用 ~5 核（4 dataloader worker 各满 1 核做 pyav 解码 + 主进程）；每步分账 `updt_s:0.107 data_s:0.524`——瓶颈是数据管线（num_workers=4、CRF18 高码率重编码视频、新服务器单核解码慢；旧服务器+原始视频硬链接曾 5.5 step/s）。GPU 83% 空闲。本轮不动 num_workers（与 10k 合同严格可比、重启无 resume 会丢 10k 步）；下次正式化提速项：num_workers 16–24 + torchcodec/decord 后端。
- 用户决策：并行训练（3 路 seed 变体 / time-trim 100k）**等 20k 趋势再定**。
- 新增 `scripts/summarize_formal3_preroll_100k.py`（预注册五 Gate + gate6 完整性；`--timetrim-ref-step` 冻结对照到 time-trim 10k；train48 集取 fit48 manifest）、`scripts/run_formal3_reset_preroll_100k_summarize.sh`、`scripts/run_formal3_reset_preroll_100k_trend20k.sh`（20k 单点 dev12+train48 max-frames-2 评估 + h0 bias 分析，输出 `trend_20k/`，不碰正式评估目录）。冻结的 10k 汇总脚本未改动。
- 口径对照 smoke PASS：新汇总器跑 10k 轮产物（steps 2000）与冻结 `summary.json` 七项逐位一致（dev12 方向 4/12、pan 7/12、幅度比 1.8672、trim-start 12/12 vs 7/12、任务区间 arm 3.6064/5.6186）；train48 侧 15/48、|bias| 5.513°、幅度比 2.097 与 10k 诊断记录一致。smoke 产物留证 `outputs/formal3_preroll_p2_20260929/smoke_100k_summarizer.{json,md}`（冻结 summary 未动）。

## 关键入口

- 工作流：`lerobot_research_workflow.md`
- 工作日志：`../docc/日志/工作日志.md`
- 阶段E报告：`../docc/报告/C50_阶段E评估报告_2026-09-19.md`
- round 2指南：`../docc/指南/C50_真机测试指南_round2_2026-09-19.md`
- 水面关键点联调报告：`../docc/报告/C50_水面关键点训练分支联调_2026-09-18.md`
- C0 P2实验合同：`../outputs/c50_c0_explicit_box_20260921/p2_interface/experiment.md`
- C0 P3短训合同：`../outputs/c50_c0_explicit_box_20260921/p3_short/experiment.md`
- C0 P3诊断报告：`../outputs/c50_c0_explicit_box_20260921/p3_short/diagnosis.md`

## 仓库状态提醒

- 工作分支：`det`；当前同步基线提交：`0944086d`。
- 工作区存在用户未提交的指南格式修改和未跟踪材料；任何代理都必须先检查`git status --short`并保护这些内容。
- `AGENTS.md`是项目代理规则真源；仓库不再携带工具专用权限、skills或入口文件。

## 2026-09-28：Formal3 时间裁剪派生集准备

- 已确认真机起步停滞的主要数据机制：原数据开头含较长静止动作，ACTDet 的首个显著动作通常位于预测 horizon 的约第 62--66 步；部署每次仅执行 `n_action_steps=1` 时会不断重规划并重复执行静止的第 0 步。
- 新增 `scripts/prepare_formal3_time_trim.py`，从 `kind_merged_nomaster` 非破坏性生成时间裁剪派生集。60 个 episode 全部保留；每集起点为相对前 15 帧基线、机械臂五关节 L∞ 偏移至少 5° 且持续 3 帧的首帧，终点为人工审核 `release.end_frame` 的后一帧。
- 真实数据冒烟结果：35,917 帧保留 23,177 帧（64.529%）。数据行、episode/global 索引、时间戳、阶段标签与 top 检测 XML 同步过滤和重编号；原视频不重编码，使用硬链接并平移 episode `from_timestamp`。
- 项目环境定向 pytest 2 项通过；LeRobotDataset/PyAV 对 ep0/10/37/59 的 top/gripper 裁剪首帧与源帧逐像素对照均为 max diff=0。
- 已生成 `kind_merged_nomaster_time_trim_fit48`：沿用原冻结 train48/dev12，不重新抽样；训练/开发帧数为 18,482/4,695，state/action 统计仅由 train48 重算。真实数据双相机加载与统计复算通过。
- P3 两路 10k 合同位于 `outputs/formal3_time_trim_p3_20260928/experiment.md`，只比较同架构 DET 原始数据与时间裁剪数据。
- 远程两路并发 batch=8、2-step 真实数据预检均 exit=0，正确加载原始28,736与裁剪18,482个train帧，无NaN/OOM；自动 Gate 已生成 `preflight_ready.json`。合同状态升级为 `PREFLIGHT_PASS / AWAITING_TRAIN_AUTHORIZATION`，10k未启动。
- 用户已授权合同内两路10k，远程 `screen -S formal3_timetrim_p3` 已启动；原始/裁剪组均越过首100步，约5.5 step/s/路，GPU约12.7GB，无OOM/NaN，预计30--40分钟。不授权自动扩展100k。
- 两路10k已于2026-09-28 20:03完成，exit均为0，耗时约29分55秒/29分57秒，五个checkpoint节点齐全且`train.done`存在。当前进入离线评估待执行状态，不能依据训练loss判断时间裁剪有效。
- 两族五checkpoint dev12完整轨迹评估已在 `screen -S formal3_timetrim_eval` 启动；裁剪模型读取同一原始部署frame 0的五checkpoint交叉测试已完成。汇总器将比较相同初始观测及相同裁剪时间区间，避免用不同帧段制造收益；当前尚未形成结论。
- 时间裁剪P3评估为`PARTIAL_PASS / REAL_ROBOT_BLOCKED`：10k共同保留区间arm/gripper MAE由3.345/1.831降至2.469/1.456，四阶段均无系统退化；但真实原始frame 0上总体运动方向仅8/12同向、pan仅7/12正确、幅度中位数为示范1.59倍。裁剪起点观测为12/12同向，指向复位frame 0到裁剪起点的输入分布偏移。下一步做state/双相机交叉替换诊断，禁止直接正式真机或100k。
- 10k模态交叉替换进一步定位：原始frame0的motion cosine=0.081；换裁剪state/top/gripper分别为0.123/0.133/0.249，双图像为0.283，裁剪全观测为0.318。gripper视觉偏移最大但非唯一来源，且裁剪全观测仍仅8/12 cosine为正，表明模型方向学习也未稳定。下一候选是显式构造少量“真实复位观测→首运动目标”配对或短pre-roll重标记消融，不直接真机/100k。

## 2026-09-21：P4 四路训练已启动

- P4 合同：`outputs/c50_c1_diffusion_p4_20260921/experiment.md`，状态 `RUNNING`。
- 四路配置：`C1_BASE`（state box）、`C1_DROP`（box dropout=0.10）、`C1_NOISE`（box noise std=0.03）、`DIFFUSION_NM`（纯 Diffusion baseline）。共同使用 `formal1_C50_nomaster_fit32`、seed=1000、batch=8、100k steps、AMP 关闭。
- 四路并发 2-step 真实数据预检全部通过；Diffusion 暴露的 fit32 EpisodeAwareSampler 越界已通过仅在 `drop_n_last_frames>0` 时启用 sampler 修复，修复后预检通过。
- 正式训练由远程 `screen -S c1_diffusion_p4` 运行，日志在 `outputs/c50_c1_diffusion_p4_20260921/logs/`；目前四个进程均存活，无 OOM/NaN，C1 约 2.7 step/s，Diffusion 约 4--5 step/s。
- 当前仅证明训练启动和健康性，不代表离线评估或真机效果；完成后必须按固定 val10 和部署一致协议比较。
- 2026-09-22：P4 四路 100k 全部完成并生成 100000 checkpoint，退出码均为 0，`train.done` 已生成。固定 val10（7,11,13,14,16,27,35,38,40,43）部署一致离线评估已在远程 `screen -S p4_eval_val10` 顺序启动，结果将写入 `outputs/c50_c1_diffusion_p4_20260921/eval_val10/`。
- 2026-09-22：首轮评估发现 `offline_eval_act_det.py` 只读取 ACT 的 `chunk_size`，导致 Diffusion 退出；已改为兼容 `chunk_size/horizon`，并修正 JSON 保存完整 episode 列表。旧结果保留在 `eval_val10/`，修复后评估在 `eval_val10_v2/` 运行。
- 2026-09-22：P4 固定 val10 已完成。C1_NOISE 的 inference L1=0.337049，C1_DROP=0.340389，C1_BASE=0.382571；C1_NOISE 在 C1 三变体中最好。Diffusion=0.019265，但其零噪声首动作指标与 ACTDet 的 chunk 输出不直接可比，不能据此宣布 Diffusion 胜出。完整表见远程 `outputs/c50_c1_diffusion_p4_20260921/eval_val10_v2/summary.md`。
- 2026-09-22：统一轨迹评估已在新远程实例启动，screen `p4_trajectory_val10`，四路并行处理固定 val10。2 帧 smoke 已通过 C1_BASE 和 Diffusion；Diffusion 初次 smoke 暴露采样 RNG 不固定，已重置 RNG 后通过。当前四路均存活，输出在 `outputs/c50_c1_diffusion_p4_20260921/trajectory_val10/`。
- 2026-09-22：统一轨迹 val10 完成。Diffusion 的 h0 arm/gripper MAE=1.559/0.378、二阶差分 arm=0.306、方向一致率=0.692，但网络中位耗时 49.8ms；C1 三变体方向一致率 0.603/0.603/0.616。C1_DROP 初始 pan 偏置接近 0，C1_NOISE 方向略好但偏置 +1.972。结果仅为 teacher-forced 诊断，尚未证明闭环或真机效果。
- 2026-09-22：阶段拆分（reviewed core phase annotations，unknown 区间排除）显示 Diffusion 在 approach/grasp_transport/place 的 arm MAE=`2.002/1.468/1.429`，gripper MAE=`0.395/0.473/0.496`；C1_DROP 对应 arm=`3.158/2.762/2.736`、gripper=`2.494/0.935/1.253`。Diffusion 阶段误差均较低，但仍是 teacher-forced，下一步只授权低速真机 smoke 前的部署一致性检查。
- 2026-09-22：GitHub `det` 已推送提交 `7c6b471e`。四个 P4 100k 模型已通过 `hf-mirror.com` 上传并用 `config.json` 下载校验：`QYyyyyyyy/C50_P4_C1_BASE_s1000`、`C1_DROP_s1000`、`C1_NOISE_s1000`、`DIFFUSION_NM_s1000`。
- 2026-09-22：新远程实例 `connect.westd.seetacloud.com:12533` 已连通。完整数据位于 `数据集/formal1_C50_nomaster`，约 221 MB；`数据集/formal1_C50_nomaster_fit32` 仅有约 16 KB 的 fit32 manifest/stats。
- 2026-09-22：已将完整 dataset 上传至 [QYyyyyyyy/formal1_C50_nomaster_fit32](https://huggingface.co/datasets/QYyyyyyyy/formal1_C50_nomaster_fit32)，并加入 `meta/nomaster_manifest_fit32.json`；`hf download` 已校验 `info.json`、`stats.json`、`tasks.parquet` 和两份 manifest。真机推理必须使用匹配的元数据根目录，不要指向旧 `formal1_C`。
- 2026-09-24：检查 `数据集/formal3/kind_merged`：60 episodes、35,917 frames，元数据、Parquet 索引、top/gripper 视频帧数全部一致，完整性检查通过。已按 episode 边界生成 `annotation_segments/videos/observation.images.top/` 与 `...gripper/` 各 60 段，各目录包含 `segments.csv`；两路合计均为 35,917 帧，可供目标检测标注。
- 2026-09-24：另生成 `annotation_segments_crf18/`版本供 CVAT 使用；两路各 60 段、合计 35,917 帧，样本检查为 H.264/640×480/30 FPS，无空文件。输出约 374 MB，保留原始 CRF 0 无损版本不覆盖。
- 2026-09-27：力控/相位建模状态：已从 `formal3/kind_merged` 的 load/curr 和夹爪指令推导四档相位 `approach/grasp/lift-place/release`，并完成 60 集离线信号分析。接触检测建议以每次部署的 load 基线为参考，load 主判断、curr 备份；不使用固定外部阈值。
- 2026-09-27：方案受 Bi-HIL（[arXiv:2603.13315](https://arxiv.org/abs/2603.13315)）的双向力感控制、子任务进度率和相位条件化思想启发，但不复用其代码。当前已完成方案设计和离线信号证据，尚未完成相位标签导出、相位条件化 ACTDet 训练、真机力控闭环或三态标定。
- 2026-09-27：相位建模升为研究主线；部署侧 HOLD 状态机保留为真机安全兜底和消融对照。下一步应先实现和验证阶段标签导出，再新建训练合同；不得将当前方案报告为已完成力控。
- 2026-09-27：四阶段标签导出 P2 v2 已完成，合同与证据位于 `outputs/formal3_phase_labels_20260927/`。真实 `formal3/kind_merged` 导出 60 episodes/35,917 frames；命令/实际位置主导，load 仅作诊断。ep6 已确认是 load 规则误报，现正常标注 251–260 grasp、260–405 lift_place。候选有效标签 29,407 帧，`unknown` 6,510 帧，无坏 episode；60 张审核联系表已生成，ep0/6/38/59 用户确认，余 56 集待审。状态 `PARTIAL_PASS`、`training_ready=false`。
- 2026-09-27：新增本地四步骤视频审核台 `scripts/run_phase_review_ui.py` + `tools/phase_review_ui/`。每个 episode 分别审核闭爪前、闭爪开始、稳定窗结束/抬升起点、释放开始；支持逐帧、当前帧回填、阶段独立判断/备注、筛选、快捷键和 JSON/CSV 自动保存。60 集真实接口 smoke 通过；当前 ep0/6/38/59 四步骤继承用户确认，其余 56 集待审。
- 2026-09-27：修复审核台视频进度条无法任意拖动：本地服务原先忽略 HTTP Range 并始终返回完整文件 200；现支持 `Accept-Ranges`、206/`Content-Range`、开放/后缀范围及非法范围 416。真实 ep6 视频中段 1000-byte Range smoke 通过；右侧新增四阶段含义说明。
- 2026-09-27：标注规范升级为 v3 动作区间：删除信息重复的“闭爪前单帧”，不再要求从连续闭合中挑唯一帧；人工分别审核闭爪区间、实际抬升开始、释放区间。闭/开区间候选由实际夹爪位置的每集相对行程 10%–90%生成，load 仅诊断。60集 v3 导出 `PARTIAL_PASS`：29,407 有效帧、6,510 unknown、无坏集；新版审核结果独立写入 `review_ui_v2_intervals/`，当前需按新语义重新审核。
- 2026-09-27：审核 UI schema 升为3并简化：删除事件级“通过/需修正/无法判断”（旧按钮还存在点击重绘缺陷，现连同无效语义一起移除）；点击“确认通过并进入下一待审”才设置 episode 级 `reviewed=true`，任何编辑自动撤销确认。结果独立写入 `review_ui_v3_reviewed/`，当前60集均未确认。
- 2026-09-27：用户完成60段人工检查但旧按钮未落确认状态；磁盘边界与自动建议逐集一致，无未保存的帧号/备注差异。经用户明确确认全部通过，已通过 schema 3 API 写入60/60 `reviewed=true`；JSON/CSV各60集、边界顺序错误0。保存前备份位于 `review_ui_v3_reviewed/backup_before_all_reviewed_20260927_1533/`。下一步是从审核记录生成最终 reviewed 逐帧标签并做冻结校验，尚未自动启动训练。
- 2026-09-27（最终更正）：上一条“边界与自动建议一致”是服务器磁盘旧副本，不是浏览器内存，结论作废。通过 schema-2 兼容抢救接口从未刷新 Edge 页面捕获60集人工结果；最终 JSON/CSV逐字段一致，60/60 `reviewed=true`、五边界顺序错误0，且60集均与自动建议至少一处不同。闭爪稳定→抬升保持窗口为1–44帧、中位14帧。最终真源：`review_ui_v3_reviewed/phase_step_reviews.{json,csv}`；下一步生成最终 reviewed 逐帧标签并冻结数据 Gate。
- 2026-09-27：最终 reviewed 逐帧标签已冻结，数据 Gate=`PASS`、`training_ready=true`。`final_reviewed_labels/` 覆盖60集/35,917唯一帧：approach 13,140、grasp 3,114、lift_place 9,034、release 1,947、unknown 8,682；有效监督27,235帧。全部episode/行 reviewed、区间非空、JSON/CSV一致；release结束后保持unknown。hold严格中位数为15.5帧（约0.52秒），更正上一条误写的14帧。下一步只授权建立相位辅助ACTDet训练合同与真实批次smoke，尚未授权正式训练。
- 2026-09-27：相位辅助 ACTDet P2 接口与真实批次 smoke 已通过。新增默认关闭的四类视觉辅助头；相位预测不回灌动作 decoder，unknown 完全屏蔽，旧配置默认结构不变。20项定向回归测试通过；formal3 ep0 的 approach/grasp 两帧各完成一次更新，相位损失单独反传时分类头和共享 backbone 梯度均有限非零；无标签部署批次仍输出 `(1,100,6)`。合同见 `../outputs/formal3_phase_aux_p2_20260927/experiment.md`。正式训练仍被两项前置条件阻塞：生成 formal3 NOMASTER 派生集、接入并校验 formal3 检测框。
- 2026-09-27：已生成并校验 `数据集/formal3/kind_merged_nomaster`。60集/35,917帧保持不变，state从9维降为8维且只删除`master_gripper.pos`，action保持6维；其他逐帧列精确一致，双相机3个视频文件硬链接复用且哈希冻结，最终人工相位标签逐帧一一对应。8D真实数据上的相位辅助两步更新与无标签部署推理通过；数据转换及ACTDet定向测试共24项通过。当前正式短训的剩余前置项是：接入formal3 CVAT检测框，以及冻结互斥P3 train/dev划分并生成train-only统计量。
- 2026-09-27：formal3 CVAT检测标注接入Gate完成。top 60集/35,917帧全覆盖、每帧唯一合法cup框，1,820个人工关键帧与34,097个CVAT插值框；标准化后安装到NOMASTER数据的`annotations/top/`。18帧跨包叠框抽检无可见偏移；真实两步联合smoke产生49/42个FCOS正样本且det cls/reg/ctr均有限非零，26项定向测试通过。gripper标注中ep4全空、另13集覆盖不完整，当前明确排除，不作为背景监督。正式P3仅剩冻结互斥train/dev和train-only stats，之后才能建立短训合同。
- 2026-09-27：formal3 P3四路100k已获用户授权，合同为`outputs/formal3_phase_p3_20260927/experiment.md`。seed=1000冻结48 train/12 dev（28,736/7,181帧），state/action统计仅由train重算；四路为DET_BASE及phase weight=0.03/0.10/0.30，其他参数一致。当前本地数据视图与26项回归通过；下一步推送代码、在RTX 5090服务器独立干净工作区同步数据并运行四路并发2-step资源Gate，Gate通过后才由screen启动正式训练。
- 2026-09-28：formal3 四模型100k混合训练完成并部署 HF（`F3_P4_{DIFFUSION_BASE,DET_BASE,DET_C1_STATE,DET_PHASE_W010}_s1000`）。真机起步卡死根因复核定案：**首尾静止段占数据 30.5%（初始中位 86 帧/末尾 97 帧）+ 部署 `n_action_steps=1` 每帧重规划 → receding-horizon 自锁**（chunk h0–h60 保持复位目标、h90 才动；DET/Phase/C1 在 dev12 初始帧上 `n=1/5/10/30` 离开复位率均为 0%）。"杯体阻挡 shoulder_lift"初判已撤销（−103.7° 为每集复位姿态）；另确认采集保存的是 leader 动作而非 `send_action` 实际发送值（真问题、非本次卡死首因）。动作-状态对齐诊断：各关节最佳响应延迟一致 4 帧（约133 ms），shoulder_lift 另有 −4.31° 饱和（10,367 帧不可达标签）。
- 2026-09-28：时间裁剪 10k 短训（原始 DET vs 裁剪 DET，同构同预算）：共同任务区间 arm MAE 3.345→2.469、gripper 1.831→1.456，四阶段无系统性退化；但真实复位 frame 0 上 h0 运动向量仅 8/12 同向、pan 7/12、幅度 1.59×示范首运动，状态 `PARTIAL_PASS / REAL_ROBOT_BLOCKED`。模态交叉替换：gripper 相机贡献最大（cosine 0.081→0.249）但非单因，裁剪全观测 0.318 仍仅 8/12 为正——10k 方向学习本身不稳。
- 2026-09-28（续）：复位配对派生集构建完成（本会话接手 Codex 停点 20:51）。`数据集/formal3/kind_merged_nomaster_reset_pair`（23,177 帧）：obs[0:15]=原始复位帧 0–14、action=原始 S..E 不变、obs[15:]=原始 S+15..E；相位标签沿动作时间线、检测框按实际图像重映射（每帧一框）；双相机 120 段视频逐集重编码 H.264 CRF18。构建自检五项全过 + 解码级对齐验证（state 行级精确、解码帧 max diff ≤0.17、错位守卫 0.87）；59/60 集配对窗口纯复位（仅 ep37 S=11 混入 4 帧）。fit48 视图复用时间裁剪轮冻结 split（train 18,482/dev 4,695），train-only 统计逐位复算一致。10k 合同 `outputs/formal3_reset_pair_p3_20260928/experiment.md`=`READY_FOR_PREFLIGHT`：对照臂复用 DET_TIMETRIM_10K 既有评估；关键 Gate=复位 frame 0 总体同向 ≥11/12、pan ≥9/12、幅度比 ∈[0.5,2.0]、裁剪起点 12/12 不倒退。**训练未启动，待用户授权。**
- 2026-09-28（续二）：复位配对 10k 已完成（exit=0、final loss 0.751、5 checkpoint；预检/训练/两路评估协议全部按合同执行，dev12 评估按用户要求改为五 checkpoint 并行加速）。预注册评估 **FAIL（overall，4/6 Gate 未过）**：Gate1 复位方向 10/12（需 ≥11/12；时间裁剪为 8/12，方向确有改善且 2k/4k/8k 曾达 12/12）；Gate2 pan 8/12（需 ≥9/12）；Gate3 幅度比 2.864（需 ≤2.0，时间裁剪 1.592——方向对但幅度失控，六集 3.6–5.7×）；Gate4 裁剪起点交叉 8/12（时间裁剪 12/12，不倒退失败——复位配对数据观测不含原始帧 S..S+14）；Gate5 任务区间 MAE ratio 1.022 PASS（gripper 1.320 反优于对照 1.433）；Gate6 完整性 PASS。按合同不启动 100k，回诊断（优先级：配对窗口长度/起点规则/预滚短序列）。
- 2026-10-12：T0.2 OC-ACT token transplant directionality 完成。严格使用冻结 `P1_ocfix2_s1/100000`、Dev 12 episodes、无训练/无 Final Test/无真机；固定 confidence≥0.20、time_fraction≤0.75、state 标准化最近邻和 seed=17，每类 20 对。比较 whole/local-only/global-only 与 same-position/random controls。纵向主维 dim2 的方向一致率为 whole/local/global = 0.90/0.85/0.90，且 whole 大位移 arm delta=0.0691 高于 same-position=0.0183；random control 仅 0.15，支持纵向 object-conditioned steering。横向 dim0 仅 0.20/0.00/0.20，未通过两轴机制要求。判定 `PARTIAL`：冻结当前 P1 作为纵向机制参考，不进入 Confirmation/Final Test；下一决策只设计单变量横向 steering 干预。原始证据见 `experiments/position/t0_2_token_transplant/index1/result/`，脚本为 `scripts/eval_t0_2_token_transplant.py`，报告生成脚本为 `scripts/report_t0_2_token_transplant.py`。
- 2026-10-12：T1 `OCACT_EXPLICIT_SPATIAL_CONDITIONING` 已通过 B1/P2 真实 Formal4 batch=4、2-step smoke。B1 是重新训练的冻结 OC-ACT（explicit path off）；P2 仅增加 `[x_global,y_global,confidence] → Linear(3,128)-LayerNorm-GELU-Linear → beta*state token`，beta 初值 0.01；DINO/prototype/tau=0.20/object tokens/cross-attention/ACT 其余冻结。按合同已在远程 GPU 并行启动 `t1_b1` 与 `t1_p2`，seed=1、36 train episodes、100k steps、10k checkpoint；框架 `eval_freq` 不执行离线 Dev 推理，完成后按固定 Dev 12 episodes 离线评估并记录该限制。Final Test/真机禁止。

## 2026-10-12：T1 显式空间条件 Discovery 完成

- B1/P2 远程 GPU 100k 训练均 exit=0；无 NaN/Inf/OOM/Traceback。固定 Dev 首步 L1：B1=`0.132089`，P2=`0.145435`，P2 未优于 B1。
- P2 S1–S4 空间消融最大 Dev-L1 变化仅 `9.469e-08`；beta=`0.00318068` 非零，但显式坐标路径没有形成可观测动作影响。
- P2 token transplant whole horizontal=`0.263`、longitudinal=`0.556`，均低于冻结门槛 `0.70`。判定：`FAIL_EXPLICIT_PATH_NOT_USED`，不得进入 Confirmation、Final Test 或真机测试。
- 结果见 `experiments/position/t1_explicit_spatial_conditioning/index1/result/summary.md`、`mechanism.json` 与 `figures/`；该结果保留为显式 state-token 坐标注入的负基线。

## 2026-10-12：T1.1 显式空间路径健康诊断完成

- 在冻结 P2/100000 checkpoint 上执行只读诊断，未重训、未改架构、未使用 Final Test/真机。
- 主诊断为 `SCALE_STARVATION`，次诊断为 `OPTIMIZATION_REJECTION`：beta=`0.00318068`，state token norm=`24.487650`，raw spatial embedding norm=`8.552184`，effective residual norm=`0.027202`，effective spatial/state=`1.111e-03`。
- beta 覆盖证明路径本身可用：learned beta 下动作变化仅 `1.275e-07`，beta=`1.0` 时平均动作变化 `0.004146`；x/y/xy/zero counterfactual 在 beta=1.0 下均产生可测变化。
- implementation hook 捕获成功，融合公式误差 `0.0`，DINO 仍冻结；输入 x/y/confidence 在 train/dev 均有非零方差。中间 10k–90k checkpoint 未生成，梯度分量历史如实记为 `NOT_RECORDED`。
- `DIAGNOSIS_READY=TRUE`。当前不进入下一架构、不重训、不做真机；下一轮必须另建单变量 decision，优先处理 gate scale starvation。
- 证据见 `experiments/position/t1_1_spatial_path_health/index1/result/`。
- G1 真机测试指南已生成：`experiments/position/g1_real_robot_generalization/index1/result/REAL_ROBOT_TEST_GUIDE.md`。内容包含硬件预检、安全门、Phase A/B、现场记录规范及 B3_ACT/P1_OCACT/R3_FILM 三条推理命令；checkpoint 路径、串口和相机索引必须现场核验，当前仍未启动真机。
- 已补充 `lerobot-teleoperate` 主臂/从臂预检命令（`/dev/ttyACM1` leader、`/dev/ttyACM0` follower）；该命令仅用于手动复位和通信检查，策略推理时不与 `control_act_det.py` 同时运行。当前本地仅有 C50 三个 checkpoint，G1 冻结的 B3/P1/R3 不在本地，尚未上传错误模型。
- 已从远程服务器 `connect.westd.seetacloud.com:23089` 核验并上传 G1 三个冻结 100k checkpoint：`QYyyyyyyy/G1_B3_ACT_100k`、`QYyyyyyyy/G1_P1_OCACT_100k`、`QYyyyyyyy/G1_R3_FILM_100k`；每个仓库包含 7 个模型/配置/前后处理器文件。服务器上传前启用 `/etc/network_turbo`；真机前仍须下载并核对文件。
