# P4.8-r5 Decision — LIBERO Prototype Construction Matched to formal6 P1

**状态：冻结的科研决策，等待 Codex 按门槛执行；取代 r4 中不够精确的原型构建描述，不覆盖已存在的 r0–r4 证据。**

**总体原则**：在 LIBERO-Object 单任务中采用与真实 SO-101 `formal6` P1 **相同的原型构建方法论**（固定训练集选帧、执行方看图选目标内部点、固定半径局部 DINO 聚合），而不是光流自动发现；其余 P1 网络保持原义。此方法无需用户人工标注，但包含 **Codex 辅助指定的少量目标坐标先验**；不得称为“完全无监督/零目标先验”。

## 0. 唯一执行者与现有工作保护

- 只有**一个本地 Codex**，通过现有 SSH 连接 RTX 5090 远程服务器。允许服务器进程持续训练，与本地其它任务错时管理；不得假设存在第二个服务器 Codex。
- **P4.7 真机任务完全独立**：不改写、重启或占用其现有作业/模型/结果。
- P4.8 现有 G0/G1 PASS 证据及 r2/r3 自动方法未通过的审计永久保留；先检查是否已有 r4 作业运行，禁止覆盖/重复启动。若已启动，将其身份冻结并汇报现状，不能热改正在运行的训练。
- 目录建议：`experiments/p4_public_benchmark/p4_8_libero_pilot/index5/{decision.md,result/}`；本文件应由 Codex 归档为该目录的 `decision.md`，并记录 source SHA 和 Git commit。

## 1. 冻结事实和实验问题

源文档：`formal6_120集_P1原型标注方法说明.md`（2026-10-30），确认历史 SO-101 P1：Train90 清单 0%、25%、50%、75% 序位各取一 episode；各取约 25% 进度的 Top 帧；Codex 查看拼图并提供四个杯内部坐标；DINOv2-S/14 patch 16×16×384，单次 ImageNet normalize；原图 640×480 中以坐标为中心、x/y 半径 35px 的椭圆选 patch 中心；区域平均→L2 normalize→共享四个原型。无逐帧框/掩码，无用户点击。历史坐标、旧银行绝不可复用于 LIBERO。

实验问题：
1. **原生 ACT vs P1**：Object-Centric 结构在 LIBERO 新任务是否有益？
2. **Legacy vs Fixed P1**：仅在线 DINO 二次 vs 单次标准化，是否影响泛化或成功率？
3. Object ON/OFF 仅诊断使用率，不将关闭已训练残差等同于重新训练的架构消融。

## 2. 模型定义与公平性

| ID | 模型 | 关键差异 | 训练预算（pilot） |
|---|---|---|---|
| ACT-Default-LIBERO | LeRobot 原生 ACTPolicy，无 DINO、无对象残差、无 Load/Current 伪信号 | 官方模型默认配置需导出 effective config；为了配对训练若有 override 则明确写出，不能宣称“所有超参默认” | 10k steps |
| P1-Legacy-LIBERO | P1 主干 + 共享新 LIBERO 原型库，**在线 DINO 双重标准化** | 忠实复现历史缺陷，仅此视觉差异 | 10k steps |
| P1-Fixed-LIBERO | 与 Legacy 相同，**在线 DINO 单次标准化** | 修复版本 | 10k steps |

- P1 两版本相同的 ACT 双相机路径、DINOv2-S Top-only 冻结、4 prototypes、1 global+4 2×2 local tokens、adapter、Q=ACT encoder tokens / K,V=object tokens 的 cross-attention、可正可负的 gated residual、动作 chunk 和控制适配。不要改成 New-GL。
- P1 两版本必须共用相同的 **训练集生成原型库**、split、seed、初始化协议、优化器、batch、步数、评测初始状态、动作接口。不得对每个版本重新挑选原型；对差异记录 preprocessing unit test。
- ACT-Default 是原生架构基准，但为单变量归因，还需比较 effective config：如训练超参与 P1 不同，报告“方法基准”，不能将全部差异归于 Object Path。若必要可在下一份 decision 增设 ACT-Matched，不在本次私自训练第四模型。
- LIBERO 没有对应 SO-101 Load/Current；不伪造数据，P1-LIBERO state/action adapter 记录并验证。SO-101 已训练权重不能直接作为 LIBERO checkpoint。

## 3. 原型构建：严格复用真实项目方法（**不开发新目标发现算法**）

### 3.1 固定、非结果驱动地抽取四张训练图

- 使用已经冻结的 **LIBERO Train35** episode ID 按稳定顺序排序，计算 `train_ids[int(len(train_ids)*q)]`, `q=[0,0.25,0.5,0.75]`。输出真实选中的 ID；**不得按可见度、测试成绩或视觉便利重选**。
- 每条 episode 用 `frame_index=int(episode_length*0.25)`（0 基准），选定与 P1 Top 在语义上最接近的**agentview 外部相机**；记录相机键名、分辨率、坐标系、任何 flip/crop/resize 与索引。禁止将历史杯子坐标用于 LIBERO。
- 导出四帧单图、带 episode/frame 标签的拼图、`sources.json`、原始 RGB SHA256。**只使用 Train35**；不得看 Dev9 或闭环评测帧来确定坐标。

### 3.2 Codex 看图选目标内采样中心

- Codex 必须通过实际可用的图像查看能力打开拼图和必要单帧，依据当前任务说明中的被操作物体（alphabet soup can）为每张参考帧**指定一个目标物体内部中心**，记录四组 `(episode_id, frame, x, y)` 及简短理由和证据图。
- 这是**执行者视觉辅助伪标注**，不是纯自动无监督定位，也不是用户手工标注。不能用文件名猜坐标、用机械臂/篮子/背景代替罐头，不能借助仿真器隐藏目标真值或官方 segmentation。
- 若任一固定帧中目标不可辨识、严重遮挡，或模型无法看图，则**阻断原型构建并提交 PROPOSAL**；不能自行改选帧、降门槛或编造目标坐标。

### 3.3 特征提取、区域规则及分辨率差异

- 冻结 DINOv2-S/14，原始 RGB `[0,1]`→DINO 内 bilinear 224×224→**一次** ImageNet normalize→16×16、384D patches（构建银行的这条通路对于 Legacy/Fixed **均相同**）。
- 原 SO-101 坐标空间 W0=640,H0=480、区域规则：patch 中心回投原图后 `((x-cx)/35)^2+((y-cy)/35)^2 <=1`。
- 若 LIBERO 图像分辨率不为 640×480，先记录像素坐标映射、已存在的 flip/crop。**事先指定跨分辨率相对尺度等价规则**：patch 中心映射到实际图像原坐标；在原图的 x/y 半径分别用 `rx=35*W/640`, `ry=35*H/480`，也等价于先归一到 640×480 逻辑坐标再使用原 35px 规则。不得不加说明地在 128×128 画面硬套 35px 或擅自改 ROI 大小。
- `mask.sum()>0` 为硬门槛；若分辨率/比例导致无 patch 命中或选区主要覆盖机器人或背景，**不得调整半径/帧/视角进行隐式优化**，只提交问题证据及可选方案供下一轮 decision 批准。这是几何映射适配，不宣称同一绝对像素大小。
- 四个原型各由选区 DINO patch 向量均值再 L2 normalize，最终 bank 形状 `[4,384]`；以可读取格式保存 bank、数值校验、Train episode/frame、坐标、图像 hash、视角及几何映射、DINO revision 和 preprocessing hash。

### 3.4 审计门槛（不虚构识别精度）

- 四帧来源确属 Train35；四帧 Codex 可视审查证据与坐标记录存在；图像可视化显示采样区域位于任务目标内部；非零有效 patches；bank 无 NaN/Inf，四个向量单位范数；重复执行数值一致（容差记录）；不得用模型 Dev9 指标优化坐标。
- 若可视证据存在歧义，明确 `PROTOTYPE_SEMANTICS_UNCERTAIN` 并阻断 P1 正式训练。**不需要用户执行标注动作**，但不承诺任何输入图都能无歧义地产生原型。

## 4. 训练与闭环仿真（独立服务器队列）

- **ACT 与原型构建可并行推进**，前提是 ACT G1 训练前向/恢复等检查通过；P1 必须等待共享 bank G2 PASS。
- 统一确认 LIBERO demonstrations 时间戳与仿真控制时间基准（此前观察 10 FPS vs 20 Hz）；先按原生 LIBERO/LeRobot 文档和数据转换约定逐一核验，不能任意插值或重采样。若对闭环结果不可比形成硬阻断，则提交 PROPOSAL；不擅自修复。
- 三组各 10k training steps 为 Pilot，仅验证完整训练 + Dev9 + 同初始状态仿真闭环 10 rollouts/模型。不得把 10k/10 rollouts 直接称为正式论文 leaderboard。Train35 训练、Dev9 仅内部诊断，任何 held-out final 必须封存。
- Codex 先实测 1/2/3 进程有效总吞吐（包括数据加载/推理、总完成工作量、显存和服务器其它负载），只采用实测效率更高且可重复的并发策略。不得为 GPU 利用率牺牲实验完整性。
- 使用相同任务描述、相同仿真初始状态清单及 episode seeds，记录逐 rollout success、first failure、执行异常与控制时间，保存轨迹视频。预测误差与闭环成功率单独报告。
- 统一 10k 的 P1 Legacy/Fixed 与 ACT，不代表不同方法达到相同性能/计算成本；另外记录 wall-clock、GPU-hours、参数量、推理延迟、有效默认配置。
- 先报告 single-task pilot；完整 LIBERO-Object 10 任务、多 seed、正式 rollout 预算**需要新 decision**。不因本轮有一组成功就自动扩展所有任务。

## 5. 交付、停止和验收

交付：`result/status.json`, `prototype_sources.json`, `prototype_contact_sheet.png`, `prototype_bank.pt`, `prototype_provenance.json`, `prototype_geometry_audit.md`, `effective_configs/{act,legacy,fixed}.yaml`, `checkpoint_manifest.csv`, `training_metrics.csv`, `dev9_metrics.csv`, `libero_rollouts.csv`, `performance_summary.md`, `builder_review.md`, `redteam_review.md`, `blocking_proposal.md`（若阻断），并明确每个文件是否实际存在，不能空文件冒充完成。

**必须停止/仅报告**：来源不是 Train35、伪标注无法看图核验、选区不可靠、两个 P1 Bank 不一致、在线预处理不符合冻结差异、时间基准无法解释、存在运行中重复任务或影响 P4.7、checkpoint 不可复现、仿真无法正确加载。

成功分层：
- `G2_PROTOTYPE_PASS`: 四张 Train-only 固定帧、Codex 可视选点、共享银行、审计证据全部齐全。
- `G3_TRAINING_PASS`: 相关三模型完成真实 10k 训练和可加载 checkpoint。
- `G4_CLOSED_LOOP_PASS`: 三模型各完成预设 10 次配对仿真且成功率有记录（不强制大于零）；无结果不能宣称通过。

## 6. 对之前决策的显式修改

- **替代 r4 原型选择文字**：从“Codex 任意选择 4 张可见图”改为 **严格固定 Train35 四分位 episode + 每集 25% 帧**，防止人为优化。
- **替代 r3 自动光流**：不再将全自动无标注作为硬条件。新方法是“Codex 辅助四点目标先验”，**没有用户人工标注**，但属于少量执行者提供的目标位置先验；论文需要披露。
- 保留 r2 原定三模型 ACT、Legacy P1、Fixed P1 及配对预处理对照。
- 不修改 r0–r4 历史结果文件。若 r4 已产生真实 checkpoint，保持其身份，不与新配方混称。

> 科研结论须明确：此实验复用的是 **formal6 P1 的原型生成规则及 Object-Centric 网络机制**，但任务、数据、机器人动作接口和分辨率适配不同，不可称为原 SO-101 checkpoint 在 LIBERO 上零样本执行。
