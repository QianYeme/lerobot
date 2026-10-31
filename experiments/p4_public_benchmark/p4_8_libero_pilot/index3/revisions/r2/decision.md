# decision.md — P4.8-r2 LIBERO Legacy/Fixed P1 + ACT 无标注并行 Benchmark

**决策状态**：APPROVED_FOR_CODEX_EXECUTION（受 G2 证据门槛约束）  
**父任务**：P4.8-r0 G0/G1（已完成）、P4.8-r1（现予澄清修订），G2 Proposal（待处理）
**修订范围**：r1 的 `P1-LIBERO` 并未明确指定 Legacy 还是 Fixed；r2 将其拆分为两套必须分别从头训练的 P1，明确预处理单变量控制。**如果 r1 已经运行，不终止已有任务，先做 checkpoint/配置身份核对，将已运行模型正确归类，并通过新增作业补齐缺失组；严禁重命名伪装或覆盖结果。**  
**替代范围**：本 decision 取代 P4.8-r0 中 G2/G3 关于新增人工 ROI 标注与模型训练的安排；保留 G0/G1 既有证据，不回滚其环境、数据集和仿真验证。  
**运行约束**：只有一个本地 Codex，通过 SSH 管理服务器；P4.7 不受影响。**不允许 Codex 修改本 decision 的科研语义。**

## 1. 冻结目标

在无需实体机器人的 LIBERO-Object 公共数据和 MuJoCo 闭环仿真上，验证 **ACT 主干 + P1 原有 Object-Centric 增强**是否带来相对于 ACT 的收益。禁止新增任何人工目标 ROI、框、Mask 或利用仿真器目标真值、评估状态构建 Prototype。需要用 LIBERO 训练演示重新训练相同网络机制的策略；不得直接使用 SO-101 的 6D 动作 checkpoint。

**论文主结论必须来自闭环任务成功率**，离线动作误差只是辅助指标；这是新的机器人任务，不得声称验证了 SO-101 Force/Load 机制。

## 2. 模型组及优先级（r2 三训练作业）

| 标识 | 网络与 DINO 输入 | 训练 | 科学用途 | 优先级 |
|---|---|---|---|---|
| ACT-LIBERO | 原生 ACT；无 DINO/OC/Load/Local | 10k seed1 | 标准方法基线 | P0 |
| **P1-LIBERO-LEGACY** | 原 P1 Object-Centric；**在线 DINO 双重 ImageNet Normalize**（忠实复现旧版） | **从头 10k seed1** | 历史实现对照 | P0 |
| **P1-LIBERO-FIXED** | 与 Legacy 完全相同；**仅在线 DINO 改为单次 ImageNet Normalize** | **从头 10k seed1** | 修复版主对照 | P0 |
| P1-LEGACY-OFF / P1-FIXED-OFF | 分别从上述同一权重关闭 Object Residual | 无需重训 | 分支依赖性诊断，**非正式训练消融** | P1 |

**不能把 r1 的单一 `P1-LIBERO` 直接认定为 Legacy 或 Fixed。** 需要以实际输入处理代码、有效配置、运行 hash 与输入张量测试确认身份。默认 r1 第 3 节写的是单次标准化，但源码执行证据优先于文档文字。

对两组 P1：固定同一 Train35/Dev9、训练 batch 次序协议、seed、原型 Bank 文件/hash、DINO backbone、视觉和 ACT 权重初始化协议、8D state/7D action、优化器与预算。原型构建逻辑和 DINO 原型侧预处理**两组相同且单次**；唯一预先批准的变量为**策略在线 DINO 的单次 vs 二次标准化**。对于独立复现的随机性，保存初始化张量哈希和训练采样序列；如无法做到逐字节同初始，应如实说明配方受控而不是逐步完全相同。

ACT-LIBERO 仅作为方法基线；由于 P1 在 LIBERO 中的原型 bank 及可能不同的状态适配，ACT vs P1 的差异不能全部归因于 DINO 预处理。**Legacy vs Fixed 才是 DINO 预处理修复的直接受控比较**。

不训练 New-GL/Force：LIBERO 无 SO-101 真实负载电流，不得造假。后续论文正式结论需足够预算、多 seed、闭环重复，而非 10k Pilot。

## 3. P1 架构不可变更项

1. ACT Top/Agentview + Wrist 原始路径完整，按照审计结果保持视觉 Backbone 参数共享语义；不擅改原有 ACT CVAE、Transformer、Action Chunk 和推理机制，只有必要的 8D LIBERO state / 7D action 形状适配。
2. DINOv2-S 为 Frozen，**只处理主视角**。原型构建侧固定单次标准化；在线策略侧 **Legacy 双重 / Fixed 单次**（刻意复现历史差异，且在论文中明确称为旧实现缺陷）。ACT ResNet 输入对两版 P1 和 ACT 均不可因修复而变化。禁止将 Legacy 双重标准化错误地描述为推荐实现。
3. Prototype 数固定 4，Top DINO features 匹配，保留原 P1 cosine max、tau=0.20 pooling、1 global + 4 local object tokens、2×2 local partitions、raw 387D token 到 256→512 adapter、slot embedding、encoder tokens 为 Q 的 cross-attention、可正负的 learnable gate（初值 0）。核对真实代码后记录精确配置；禁止因为 LIBERO 对象不同而重新设计模块。
4. 只允许机器人相关接口变化：LIBERO 图像 key、8D state 语义、7D action 与对应归一化；严格记录 shape、dtype、范围、频率、时间窗和所有 override。
5. 原型**来源规则**与原 P1 一致至关重要。若原 P1 实际使用了隐含人工 ROI、手工点位或专门采集先验，必须如实记录。不能为了“无标注”把它重写为新发明的目标发现算法。不能复用透明杯 bank。

## 4. G2 原型无标注来源的核验门槛

Codex 必须先查证原 P1 cup Prototype Bank 的真实构建脚本、版本、训练帧选择、patch 选择、缓存文件与 hash，输出 `prototype_lineage_audit.md`。**原始模型没有逐帧目标标注，不等于原型生成天然无任何目标先验**。

通过路径：
- A（优先）：可验证原脚本及其无新增人工标注的输入要求 → 对 LIBERO train split 原样复用选帧/patch选择/聚类/四原型规则，锁定 seed 与 artifacts；提供可视化覆盖和 train-only provenance。
- B（需停下来提案）：若原机制依赖杯专属先验且无法无标注等价迁移，或原始脚本不可恢复，**只提交最小兼容性 Proposal**（包括新算法变更点、潜在不公平性），不得擅自采用自动交互分割、KMeans、随机 patch、仿真 GT、人工 ROI 当作原 P1。

验证包括：Prototype 的 train-only 证据、归一化一致性、相似度热图与抓取目标的**定性**对应、排查机械臂/背景主导、故意空场景时的响应；禁止把可视化当作定量标签或据此调参到 test。明确区分无新增人工标注与无监督。

## 5. 数据、环境、评估协议冻结

保留 G1 已验证的 LIBERO-Object task index 24（alphabet soup can into basket；用脚本中实际 task id 核对），44 条演示、6,867 帧的当前数据源；保留已冻结 Train35/Dev9 episode id。若 G1 并未正式冻结划分，以不可变 seed 一次生成，写 manifest，禁用隐藏评估数据；不得为了提高结果更换演示/任务版本。

- 使用相同两路 RGB 与 8D state、7D action；ACT vs P1 改变 OC 分支，**Legacy vs Fixed 仅改变在线 DINO 预处理**；时间戳数据约 10FPS 与模拟控制 20Hz 的接口契约必须先核验，且两模型相同。禁止未经批准盲目插帧、重放动作或改控制周期。
- 评估必须使用 MuJoCo 闭环，固定任务 init states、seed、control_mode、reset 策略、最大步数、相机、成功判据。官方 LeRobot LIBERO eval 提供 `libero_object`、task_ids、n_episodes 和固定 init states。遵循当前仓库版本 API，不盲抄网上命令。
- 可行性 Pilot：三个模型各 10k updates、同 batch/优化器/learning rate/gradient clip/训练统计与 seed；每模型**同一组** 10 个 rollout，仅评估运行通路，不以成功率差异宣称架构优劣。
- 正式阶段须另有批准的预算：LIBERO-Object 10 tasks、每 task 至少 10 个配对初始状态；推荐 3 seeds。若单任务全部 0 成功，先验证时间基准、动作执行和足够训练预算，不能仓促归咎于架构。
- Offline Dev9 MAE 只作辅助，不把它与真实闭环成功率混为一谈。保存训练墙钟时间、显存、参数量、推理延迟、GPU 类型。

## 6. 服务器并行执行（唯一 Codex）

0. SSH 只读查询现有 P4.7/P4.8 进程、磁盘空间与依赖。不得杀死、重启、覆盖或篡改已运行实验。为 P4.8 创建独立环境/运行目录。
1. G2 原型来源审计先行；不通过则停止 P1 训练，仅允许 ACT 工程 smoke 和形成阻断 Proposal，避免白白占用整夜服务器。
2. 三个模型各执行短 GPU smoke：真实 batch forward/backward/optimizer、可恢复 checkpoint、动作形状与 eval 接口；固定随机种子与代码 commit。
3. 先测单进程 ACT / Legacy / Fixed 吞吐、显存，再测双进程、必要时三进程聚合吞吐。只有在**聚合有效 updates/s 提升至少 10% 且无 OOM、资源干扰**时采用该并行度；否则回退到总完成时间更短的调度。比较各任务剩余 update、总 wall-time，不以 GPU 使用率为效率指标。不能把显存空余等同于计算资源空余。
4. 持久化调度（tmux/supervisor/可靠作业队列），每 job 独立 stdout、配置、checkpoint、心跳、pid、日志和异常标记；SSH 断开可继续；同一 Codex 后续通过 SSH 查询，不创建第二个 Codex。
5. 训练 10k paired Pilot。先 checkpoint 完整性/Dev9，再相同 init states 的 10×3 闭环 rollout；保存逐 rollout success/failure、录像（如可行）、恢复日志与 object-path on/off 诊断。
6. 到达期限 / 资源不足：先保存可恢复点，**不擅自降级模型和不强行启动完整 10 task**。G3 结果包独立交付，待后续决策。

## 7. Builder / Red-Team 验收与失败代码

**Builder PASS 条件**：原型来源有证据、无新增人工标注、配置差异受控、DINO 单次归一化、数据 schema 与动作时间基准一致、三个模型均可训练并恢复、成功执行匹配 init states 闭环评估、P4.7 未受干扰。

**Red-Team 检查**：原型是否偶然聚焦机械臂/背景；P1 特权输入；训练数据或 simulator ground truth 泄漏；将 10k pilot 结果当正式排名；将 evaluation-time Object OFF 当训练消融；新数据 10FPS 到 20Hz 错配；并行造成训练速率下降；不公平地把额外 object/prototype 构建成本排除在算力报告外。

状态标签：`G2_PROTOTYPE_PROVENANCE_BLOCKED` / `G2_SCHEMA_BLOCKED` / `PILOT_COMPLETED` / `PILOT_INCONCLUSIVE` / `READY_FOR_FULL_SUITE_PROPOSAL`。所有未证实事项显式 `UNKNOWN`，不自动宣布通过。

## 8. 交付产物和目录

建议仓库路径：`experiments/p4_public_benchmark/p4_8_libero_pilot/index3/decision.md`（不覆盖正在执行的 r1/index2）。

`result/` 至少含：`prototype_lineage_audit.md`、`effective_config_act.json`、`effective_config_p1_legacy.json`、`effective_config_p1_fixed.json`、`legacy_fixed_preprocess_probe.json`、`dataset_split_manifest.json`、`environment_manifest.json`、`normalization_and_timing_audit.md`、`parallel_benchmark.csv`、`train_metrics.csv`、`checkpoint_manifest.csv`、`dev9_metrics.csv`、`rollout_results.csv`、`object_path_diagnostics.csv`、`summary.md`、`builder_review.md`、`redteam_review.md`、`blocking_proposal.md`（若需），以及全部可复现代码 commit/hash 与可恢复 checkpoint 指针。

**执行交接**：单个本地 Codex SSH 管理服务器。若 r1 已启动，先只读检查进度、将已存在作业按真实配置归类，再仅补建缺失的 Legacy 或 Fixed 组；当前 r1 任务继续训练，不得为了切换 r2 停止已有有效工作。若原型来源 G2 仍阻断，则仅登记三组排队，不擅自越过 G2。

**不允许 Codex 自行修改 decision.md；科研语义有歧义或来源阻断时提交 Proposal，由用户与 ChatGPT 冻结后再执行。**
