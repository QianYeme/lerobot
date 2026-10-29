# decision.md — P4.3-r2 单 Codex 编排双轨：本地真机对照 + SSH 服务器 New-GL 并行训练

- **决策日期**：2026-10-29
- **状态**：FROZEN FOR CODEX IMPLEMENTATION；实体机器人运动另需现场操作者批准与安全预检
- **实验编号**：`P4_3_r2_SINGLE_CODEX_DUAL_TRACK_ROBOT_NEWGL`
- **建议存放**：`experiments/p4_final_p1_training/p4_3_dual_track_robot_newgl/index2/decision.md`
- **科研语义所有者**：用户与 ChatGPT；Codex 为执行者，不能修改、覆盖或隐含重解释本 decision。遇到冲突写 `PROPOSAL.md`。
- **相对于 P4.3-r1 的关系**：本文件为**执行编排纠偏的 r2**；完全继承 r1 的模型身份、训练变量、实验规模、评估指标和停止条件，仅将错误的“两 Codex 会话”假设替换为“唯一的本地 Codex 经 SSH 编排远端任务”。本文件为**重新规划的 r2**，保留原本三模型真机实验的科学设计，新增独立并行的 New-GL 服务器工作流。**若 r0/r1 已启动，保留现有日志与实验试次；不得重启、覆盖、重复计数或热切换计划**。审查冲突并先提交 PROPOSAL。若 r0/r1 未启动，以 r2 为唯一执行计划。
- **执行拓扑**：仅有**一个本地 Codex 会话/执行代理**，本地 Windows/笔记本经 SSH 连接远程 RTX 5090；A 在本地 SO-101 + RTX 4060 8GB；B 是由同一个本地 Codex 远程创建的服务器独立训练进程（不是第二个 Codex）。两轨计算运行可并发，Codex 的交互式调度按时序交替进行。两轨不可共享可变运行目录。

## 0. 本 decision 做什么

1. **轨道 A（本地，立即执行）**：冻结 Legacy P1 seed1 80k / Fixed P1 seed3 80k / Original ACT seed1 80k 的已验证模型包，在同一套 source-left → target-right 任务和相同物理条件下，执行统一部署预检、离线推理核对、集中监督真机对照。默认 3 模型 × 6 固定摆放位置 × 2 重复 = 36 个计分尝试。
2. **轨道 B（服务器，同时执行）**：以 *New-GL 本身* 为研究对象，分别构建 `NG-L`（历史/Legacy 双重预处理配方）与 `NG-F`（仅修复 DINO 单次标准化的配方）；分别训练 **Global → Frozen Global → Local Residual**；以同 Global 的 Local OFF/ON 和两个 Global Legacy/Fixed 比较，完成统一离线评估与可复现实验报告。
3. **两轨汇合**：A 先形成当前 Position 的真机事实，B 先形成 New-GL 是否值得真机测试的离线证据；B 的机器人测试不是本 decision 的自动授权动作。若符合条件，另作补充真机 decision，尽量安排在本地同一批次测试而非反复重启设备。

**绝对不做**：中止 A 等待 B；为了消耗 GPU 而训练不兼容的 New-GL；用历史旧任务 checkpoint 冒充 formal6 新任务模型；默默修复已有 Legacy checkpoint 再续训；修改 P1、ACT、New-GL 的冻结研究语义；用 Final15 选模型；未经现场批准运行机械臂。

## 1. 已知事实 / 待核验项

- 数据：`QYyyyyyyy/formal6_merged`，revision `83a829af6df245ed53596dbc22b18c19f8bd3315`，Train90/Dev15/Final15 固定。最终任务是左侧区域随机杯位抓取半满透明杯、抬起、稳定停留约 0.8–1.0s、搬运至右侧目标区并放置释放。相机 Top/Wrist 640×480@30fps。Source 和 Target 实际几何边界现场测量，不能将历史估计当成精确定位真值。
- P4.2.1 统一 Dev15 参考：Legacy P1 s1 80k arm/grip `0.06436/0.07739`；Fixed P1 s3 80k `0.07836/0.08906`；Original ACT s1 80k `0.06830/0.08046`。**它们不是成功率**，不意味着实际排名。
- P1 的 Legacy DINO 重复 ImageNet 标准化已经证实，Fixed P1 修复后 Preprocess 一致，但 Dev15 代理指标没有改善；不可据此推定机器人表现。
- **New-GL 当前留存实现**存在双重标准化路径；其历史实际完整执行文件和 prototype checkpoint 血缘尚未完全闭合，历史模型结论仍是 `NEWGL_HISTORY_UNVERIFIABLE`。本轮新配方 `NG-L` 只能称**依据现存证据重建的 Legacy 预处理实验**，除非找到可证明的精确历史版本；不得宣称精准复现历史训练。
- New-GL 与 P1 不同：Strong-G Object-Centric Global + 6D Position State / 单独 Load Token + Local Load-Aware Gripper Residual。**必须先从审计源和训练日志锁定真实 schema 和训练脚本**；不能套用 P1 的 8D state、100×6 输出和单阶段 80k 联训配方。

## 1.1 唯一 Codex 与 SSH 远端任务的执行架构（执行硬约束）

- **角色唯一**：用户只有一个在本地工作的 Codex。Codex 使用当前已经验证的 SSH 连接配置进入远程服务器，在远端执行 shell/训练脚本，并通过同一 Codex 读取远端日志、检查进程、拉取轻量评估结果。**禁止要求用户另开一个“服务器 Codex”会话，禁止假定远端有 Codex 代理**。
- **并发含义**：真正并行的是服务器上的 `NG-L`/`NG-F` 训练进程（必要时多个 GPU 作业），以及服务器计算与本地真人监督下的真机测试；**不是同一个 Codex 同时进行两场需要连续交互的任务**。Codex 应先把远端任务变成自运行、可恢复、可监测的受控作业，再切回本地真机流程。
- **首先做 SSH 预检**：只用用户现有 SSH host/端口/密钥和仓库路径，核验 `hostname`、GPU、远端工作目录、数据/checkpoint 可用性和未结束任务；未确认前不得新建训练或猜测 SSH 凭证。不得把密码写入 decision/日志/命令。
- **服务器后台生存性**：Codex 编写并检查远端受版本控制的 `launch.sh` / `supervise.sh` / `status.sh` / `stop.sh`（或现有作业管理器），使用 `tmux`/`systemd-run --user`/明确验证的 `nohup` 等方式让训练**脱离 SSH 连接后继续运行**；务必在 smoke 中测试 SSH 断开重连后进程存活、日志续写、checkpoint 可读取。单纯 `cmd &` 不能被默认认定为足够可靠。
- **任务隔离**：每个 Global/Local run 使用唯一 run_id、PID 或作业 ID、进程组、日志、环境变量、随机种子、只读数据和独立 checkpoint 目录；`status.sh` 报告每组的状态/step/日志路径/最近 checkpoint/错误原因，不依赖交互式会话的内存状态。
- **前置队列/依赖**：可同时启动两组 Global；必须等相应 Global 通过完成与身份检查后才自动触发其 Local。允许由同一远端 supervisor 按依赖图推进，**Codex 不需要长期在线盯进程才能完成远端队列**。与所有已冻结的训练 budget/科学变量保持一致；遇到科研语义问题不得自动选择替代方案。
- **本地切换**：远端 smoke、并发吞吐及自动执行经验证后，Codex 在本地运行 A 轨道部署预检；现场真机测试由用户确认安全条件并监督运动。期间远端训练继续运行；本地 Codex 可在自然测试间隙以同一 SSH 连接查询 B 状态，**不得在机器人运动过程中因远程操作失去现场安全监督**。
- **资源争用**：服务器 GPU 训练与本地 GPU 推理是物理隔离的，但本地数据同步、网络带宽、磁盘 I/O/CPU 负载可能影响本地真机控制；禁止在真机动作期间同步大型 checkpoint 或运行高负载转码。只在测试前后拉取结果摘要和校验清单。
- **会话中断与恢复**：本地 Codex 关闭、SSH 中断、机器休眠，不得被误报为服务器训练结束。重新进入时先读 `status.sh` / manifest、锁文件、真实 PID 和 checkpoint；**不得重复启动同一 run、覆盖输出或从错误 step 重跑**。无权限或无法确认远程作业状态时标 `REMOTE_STATUS_UNKNOWN` 并暂停重复启动。
- **控制权限**：自动训练/离线评估可无人值守，但不得远程或定时自动触发本地机械臂运动；机器人执行必须有现场用户允许与可用的紧急停止方法。

## 2. 两轨任务隔离、时间和资源规则

- 轨道 A：一台本地机器、一个真机测试会话。可独立于 B 的训练进度启动。只从只读已验证模型包加载权重；录视频、CSV、部署脚本日志。绝不从远程训练中途传一个未完成的 checkpoint 自动替换候选。
- 轨道 B：只读冻结 Train90/Dev15/prototype-bank，使用各自独立进程、独立结果目录、独立 optimizer/checkpoint/log、原子 checkpoint 落盘；数据并行读取可共享只读缓存。不得覆盖 P4.2/P4.2.1 的输出。
- 一次无人值守服务器窗口按 **9h 必须收尾、10h 弹性**规划；**不能保证**从零完成全部 Global/Local，先做工时/吞吐预测。时间预算根据真实测量而非凭空假设；保留至少 45min 用于离线评估、校验、摘要、完整权重落盘；若估计无法完成两个配方成对的同预算阶段，则优先完成可解释的成对比较（而不是一个完整、另一个未完成）。
- 单卡 5090 并行：先量化 1 个作业、2 个作业（必要时 3 个）短时稳定训练的**总有效样本/秒或总有效 step/秒**、单作业速度、VRAM 峰值、CPU 解码/IO、温度与 checkpoint 开销。只有总有效吞吐相对串行基线提高至少 10%，且每作业无 OOM、异常减速、数值不稳定，才使用该并行度；否则退至更优的并行度/顺序执行。**99% GPU 使用率不代表并行加速**。
- 同步隔离：可并行训练 `NG-L Global` 与 `NG-F Global`；两 Global 通过同等预算/核查后，再并行训练各自的 Local；**绝不能先启动依赖未定稿 Global 的 Local**。每个 Global 仅用本配方对应训练设置；任何资源中断须保存完整 optimizer、seed、step、normalizer、prototypes、代码版本。

## 3. 轨道 A — 本地真机对照（P4.3-r0 科研语义不变）

### 3.1 候选锁定及预检（BLOCKING）

- `L1`: Legacy P1_OCACT seed1 80k，使用**历史匹配的双重标准化推理路径**，绝不可将 Fixed 代码热补到旧权重。
- `F3`: Fixed P1_OCACT seed3 80k，必须使用 P4.2.1 对应**修复版部署入口**，ACT Top/Wrist 标准化路径不变，另留 `top_raw [0,1]` → DINO 内部**仅一次**标准化。已记录 checkpoint sha256 `92b421820d759b939e04536541d1833eae47d237c1762667eae505a16e75f3f5`；现场复核实际文件。
- `A1`: Original ACT seed1 80k，不带 DINO，实际文件路径/checksum/输入规格/部署入口须检索验证，不可编造。
- 三者均须核验真实 checkpoint SHA256、训练版本、Train90 normalizer、camera ordering、6D 动作、状态映射、ACT chunk/temporal ensembling/控制频率、load/current 单位、follower `nn` 校准、`use_degrees` 和 gripper 0–100 实际设备合同；无法核验则阻塞对应候选并提 PROPOSAL。
- 每模型执行相同 Dev15 **部署封装**离线 forward，检查关键数值符合已有 reference 容差；再做摄像头 RGB/BGR、Top/Wrist 帧率、丢帧、实际单位、推理延迟、输出有界性检查。空杯/无物安全动作方向测试必须现场人工监督、降速、无障碍、可紧急停止。

### 3.2 固定真机试验矩阵

- 在同一 source 区预先标定 **6 个可重复有效位置 P01–P06**（带照片标尺和物理 marker；不捏造 mm），每点每模型 2 次，共 **36 次**。顺序按 `(position, repeat)` 配对分块，在每块内以冻结随机种子随机排列 `L1/F3/A1`，避免将时间和环境漂移混淆为模型差异。
- 每次相同 Home、半满透明杯、光照、Top/Wrist 位姿、目标区条件、杯子放置姿态；记录 source/target 几何、杯液面和测试前后的环境变化。各次最大持续时间依示教时长和可解释安全余量在执行前锁定。
- 计分：`end_to_end_autonomous_success` 需要独立完成接近、抓取、举杯、稳定悬停、向目标移动、目标区放置、释放，且不洒漏/不倾倒。记录首次失败阶段、接触、滑落、抓空、提前闭爪、误提前放开、阶段跳跃、错方向、摆放越界、推理/设备错误、停止原因等。报告每模型 `成功次数/12`、分阶段率、配对胜负、Wilson 95% CI；样本少不声称显著性。
- 人工介入 = `assisted_recovery=true`、`autonomous_success=false`；视频、CSV 可以分别归档，不强迫历史上不存在的逐帧对齐。每次带唯一试次 ID。
- 碰撞风险、液体泄漏、电机异常、关节越界、无法停机、推理链严重异常立刻 STOP，记录 `SAFETY_STOP`，安全事件后暂停所有模型，现场排查且需操作者批准。不可无人值守。
- **不等 B**；B 的完成与否不改变 A 的样本量、随机表、候选身份或评分定义。

## 4. 轨道 B — New-GL Legacy vs Fixed 成对 Global+Local 训练

### 4.1 恢复原生训练协议（BLOCKING，先于训练）

Codex 必须从历史审计/仓库找到并锁定：Strong-G Global 和 New-GL Local 的源代码 hash/commit、状态具体字段与顺序（6D Position + 独立 Load Token）、输入图像管线、Object Tokens 与 Prototype bank 实际调用、Global 输出维度、Local 输入/目标 residual 的计算与裁剪、Global/Local 阶段冻结与训练策略、用于训练的 action/actuator 单位、inference 合成公式、时间对齐、train-only 标准化、optimizer/scheduler/seed/训练 step/batch、原生 checkpoint 结构及恢复方式。若无足够信息实现忠实配方：输出 `SCHEMA_BLOCKED` + 最小证据缺口，**禁止偷偷改造成 P1 或普通端到端 8D ACT**。

为适配 formal6 新任务，明确区分**历史架构恢复**和**新的训练数据适配**：只能对 input adapter / action target 作保持历史语义的兼容映射；所有映射须证明无信息泄露、正确单位、state 原始字段来源。若必须做会改变策略语义的改动，写 PROPOSAL，暂停 B 的正式训练。

### 4.2 唯一主变量与严格配方

两个新训练分支应完全共享：formal6 Train90/Dev15 split、同一数据 revision、同一 Train-only Cup Prototype 数值和构建策略、同一 New-GL 网络拓扑、同一状态/action schema、相同训练/评估视图与增广、同 seed、相同 Global/Local 各阶段 step/batch/优化/clip/scheduler、代码版本与环境（仅预处理开关导致必要的唯一代码不同）。**Legacy/Fixed 区别仅在 Global 侧 DINO online image preprocessing**：

- `NG-L`: 在线 Top 图像已经通过 ACT ImageNet normalizer，DINO 内部再 ImageNet normalize（双重）；prototype 侧仍采用单次标准化。若实际重建配方不同，记录并重新审批；不要用一句“模拟历史”掩盖差异。
- `NG-F`: 从 raw Top RGB `[0,1]` 分出独立分支，DINO 内部执行一次相同 ImageNet normalize，保持 ACT ResNet 的输入与 NG-L **bitwise/numerically unchanged**；prototypes 侧与在线 DINO 相同输入分布。
- **两条轨迹都必须从 step 0 新建随机初始化（含两个阶段各自的起点），同一 Global seed**；不得拿历史 Legacy New-GL 权重给 Fixed 热修；不得拿不同 seed、不同旧任务数据当因果对照。
- 冻结 Global 之后，每个 Global 分别训练自己的 Local residual（一个 Global 对应一个 Local），保存 `Global-only / Local-OFF` 和 `Global + Local-ON` 两套可独立推理的评估配置。**Local OFF 是同一个 Global 的 residual=0，不另训一个 Global**；Local ON 不得反向更新 Global 参数。ON/OFF 需要控制相同观测、执行顺序和部署前处理。

### 4.3 冻结训练预算与模型矩阵

| ID | Global DINO | Global 训练 | Local 训练 | 离线对照作用 |
|---|---|---|---|---|
| NG-L-G | Legacy double norm | formal6 Train90，同 seed/预算 | 无 | Legacy 视觉 Global |
| NG-F-G | Fixed single norm | 同上 | 无 | Fixed 视觉 Global |
| NG-L-GL | Legacy double norm | **复用 NG-L-G 完全相同权重，冻结** | Train90，原生 load residual 训练 | Legacy Local ON |
| NG-F-GL | Fixed single norm | **复用 NG-F-G 完全相同权重，冻结** | 同上 | Fixed Local ON |

离线评估必须形成 2×2：`NG-L-G` vs `NG-L-GL`（同 Global，Local OFF/ON）；`NG-F-G` vs `NG-F-GL`（同 Global，Local OFF/ON）；`NG-L-G` vs `NG-F-G`（Global DINO 预处理）；`NG-L-GL` vs `NG-F-GL`（系统级修复效果，含 Local 的间接影响）。**仅凭最后一项不能归因 Force。**

默认先运行一个匹配 seed 的**完整配对试验**。原生训练阶段预算应首先依据历史代码/可复现配置冻结；如果历史预算不适用 9–10 小时，Codex 根据 smoke 基准提出一个两个分支均可完成的**统一较短 pilot step**，并记录与历史配方差异。优先保证两组 Global 同步完成、再两组 Local 同步完成；只有所有必需阶段达到成对可比 checkpoint 才声明 `PAIRED_COMPLETE`。剩余时段可通过 approved proposal 增加 paired seed，不能单边延长 Fixed 来追求胜出。

### 4.4 服务器并行执行与性能阈值

- 冒烟：各支分别试跑 I/O、loss finite、global forward、grad、保存/恢复优化器；DINO input probe 同固定 Train 图像：检查 online/prototype 的输入范围与差异，NG-F 差异必须在已冻结合理数值容差内，ACT branch NG-L/NG-F 输入相同；检查 Local OFF 等价未加 residual。
- 以同样样本读取负载测 1 worker → 2 workers → 可选 3 workers 的总有效吞吐。优先两条 Global 同时跑；若双模型并行收益不足或影响稳定性，则顺序跑。优化**完成有效实验**而非 nvidia-smi 占用数字。
- 使用两个独立输出路径和 process group；随机数状态、train step、optimizer/scheduler、GPU 显存/吞吐、数据读取瓶颈、异常和重启次数进入 manifest。禁止覆盖 checkpoint。任何异常退出必须让另一支运行继续，并记录恢复公平性是否破坏。
- 预算不足：先保留两个 Global 统一步数的 checkpoint + Dev15；若 Local 不完整，状态 `PAIRED_GLOBAL_ONLY`，**不得将不完整的一支 Local 作为最终 New-GL 完成结果**。保存可恢复状态，明确下一窗口续训计划，不在未授权情况下修改科研问题。

### 4.5 统一 Dev15 离线结果

- 严格保持 Dev15 15 个 episodes，不触碰 Final15（包括 frame metadata 预览及超参选择）；确保评估样本、采样时间、归一化与推理动作模式一致，输出至少 arm/gripper normalized/raw MAE、分位置/阶段代理统计（如果存在可信标签才称真实阶段）、稳定性/动作平滑性、Local residual 分布、DINO 输入一致性/Prototype similarity、object-path ON/OFF 响应、全量失败与异常。
- 必须用**部署级推理 wrapper**同时评估 Global-only 与 GL，而不是用 teacher-forced training-mode 数值伪装执行效果。只报告可支持的诊断含义：Object ON/OFF≠定位正确，离线 MAE≠真实成功率，Local residual 幅度≠Force 成功。
- 预登记评估与 checkpoint 选择规则：阶段完成的候选 checkpoint 使用统一预算并以相同 Dev15 代理指标排序，列出所有被评估候选，不在看到结果后修改指标或仅挑最优 seed；最终保留至少最后 step 与 Dev-best（两者不同须并列解释）。

## 5. 汇合决策门槛 — New-GL 未来是否真机

1. `NG-L/NG-F` 若 schema、训练、预处理探针和推理 wrapper 均 PASS，且配对 checkpoint 可核验，才成为**可供下一次真机部署预检的候选**。
2. 若 `NG-F` 离线指标更好，**仅提示值得真机核验**，不能断言修复解决历史不稳定；若没更好，也不宣布 Force 无效。需重点看 Local OFF/ON 是否在夹爪/接触代理指标上呈现一致变化。
3. 原则上选择 `NG-F-G`（Local OFF）与 `NG-F-GL`（Local ON）作为最小 Force 消融真机组；如要验证 DINO 修复本身，再加入 `NG-L-GL`（必要时 `NG-L-G`）并保持原任务、现场条件和安全预检。**不能自动加入正在进行的 A 轨道 36 次试验，不能把旧实验的随机表临时扩大**。
4. 若两轨结果时间上恰好接近，可以在 A 的 36 次正式试验之后，经新的补充 decision、完整同一部署预检及现场批准，在同一次物理会话中增加预先冻结的 New-GL 试次；它们属于单独实验块，分析时不与 A 的原定 36 次混成同一个预注册三模型实验。
5. 若 New-GL schema/历史依赖无法修复，或训练产物不完整，轨道 A 照常执行，B 报告 `BLOCKED` / `PARTIAL`；不归因为服务器资源不足或模型无效。

## 6. Codex 必交付文件（分开交付、分开审查）

**A 本地 `result/robot/`：**
`model_identity_manifest.json`, `deployment_preflight.json`, `runtime_invariance.csv`, `deployed_runtime_dev15.csv`, `runtime_latency.csv`, `trial_plan_frozen.csv`, `trial_log.csv`, `stage_outcomes.csv`, `paired_comparison.csv`, `failure_taxonomy.csv`, `safety_events.csv`, `video_index.csv`, `summary.md`, `validation.md`, `builder_review.md`, `red_team_review.md`, `delivery_manifest.json`。

**B 服务器 `result/newgl/`：**
`architecture_provenance.md`, `schema_contract.json`, `preprocessing_probe.json`, `prototype_provenance.json`, `paired_training_plan.json`, `parallel_throughput_benchmark.csv`, `training_runtime.json`, `model_identity_manifest.json`, `checkpoints_manifest.csv`（每组本地/远端路径、SHA256、run id、optimizer state），`train_metrics.csv`, `dev15_paired_comparison.csv`, `local_on_off_ablation.csv`, `object_path_diagnostics.csv`, `normalization_contract.json`, `summary.md`, `validation.md`, `builder_review.md`, `red_team_review.md`, `execution_complete.json`, `delivery_manifest.json`。

如输出结果压缩包无法装下权重，至少保留服务端真实 checkpoint 的精确路径、完整哈希、尺寸、可复现加载命令及保留期；不能声称 ZIP 含有模型权重。

**全局 `result/`：** `dual_track_status.md`, `decision_trace.md`, `PROPOSAL.md`（有偏差必须提供），以及本地和服务器各自的最终状态 `PASS/PARTIAL/BLOCKED`。每条轨道单独完成 Builder + Red Team Review；不能因为一条成功而把另一条标 PASS。

## 7. 决策结果分类和硬停止

- A：`ROBOT_COMPLETED_36` / `PREFLIGHT_ONLY` / `ROBOT_PARTIAL` / `ROBOT_SAFETY_STOP`，逐模型列有效尝试与原因。
- B：`NEWGL_PAIRED_COMPLETE` / `NEWGL_PAIRED_GLOBAL_ONLY` / `NEWGL_SCHEMA_BLOCKED` / `NEWGL_RUNTIME_FAILURE`；只有全面证据满足方可做 New-GL 修复收益初判 `FIXED_BETTER`、`NO_CLEAR_GAIN`、`FIXED_WORSE`、`INCONCLUSIVE`。
- 任何伪造 Final15 评估、私改训练语义、未知行动单位、固定模型使用错误 top_raw/DINO 预处理、无法停止机械臂，均为阻断项。
- 禁止在无人现场看护时运动机械臂；如果杯子侧翻、洒漏、危险抓取，安全停止优先于完成实验数量。
- Codex 报告异常、发 PROPOSAL，不自行修改本 decision。两轨任何科研结论必须交 ChatGPT 进行 Builder/Red-Team 实证复核后再更新科研主线。

## 8. 立即执行顺序（唯一的本地 Codex；远端作业自主运行）

1. **同一 Codex 审核工作区**：读取本文件，确认是否已经执行 r0/r1，盘点本地模型包、SSH 连接和远端已存在训练任务。已启动的 r1 任务不得重启或无理由迁移。
2. **远端启动阶段（由本地 Codex 经 SSH 完成）**：核查 New-GL schema/历史依赖 → 构建 `NG-L/NG-F` 配方 → 输入预处理探针 → smoke/吞吐测试 → 冻结同预算训练矩阵 → 创建带日志、checkpoint、状态查询和恢复机制的后台作业队列；验证离开 SSH 后继续运行。若 schema 阻塞，产生 `PROPOSAL` 而非强行启动不兼容模型。
3. **本地真机阶段（仍是同一个 Codex）**：切回本地读取三个模型包、完成部署预检/安全检查、固定 36 次试验矩阵；仅经用户现场批准并监督执行，每次记录独立结果。远程 GPU 作业无需 Codex 持续连接即可运行。
4. **间歇性远端巡检（同一个 Codex）**：真机测试前后或自然停顿时经 SSH 调用 `status.sh` 检查 GPU 作业、错误、保存点和运行时间；不得因巡检干扰正在运动的机器人，不得把日志回传/模型大文件传输放在关键控制路径。
5. **收尾交付**：同一个 Codex 分别汇总 `result/robot/` 和 `result/newgl/`，各自独立写 Builder/Red-Team 记录，最后汇总 `dual_track_status.md`。两条轨道可先后交付，服务器未完成并不阻塞 A 的验收。

## 9. 针对 r1 的修订对照与非目标

- **撤销 r1 中“本地 Codex / 服务器 Codex 分别使用独立会话”要求**；以“**单本地 Codex + SSH 远端可持久运行训练作业 + 本地受监督真机**”取代。
- r1 所有冻结实验语义保持原样：A 的 L1/F3/A1、36 次真机矩阵；B 的 NG-L/NG-F Global 与各自 Local、DINO 唯一主变量、Local OFF/ON、Dev15、Train90、Final15 保留和公平训练预算。
- 本修订**不表示**已实际连接 SSH、已启动任何训练、已下载权重或已运行真机。上述状态必须由 Codex 提供可核验执行记录，不得在汇报中提前写 PASS。
