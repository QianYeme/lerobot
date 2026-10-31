# Decision P4.8-r3 — LIBERO Train-only 自动原型 + ACT / Legacy P1 / Fixed P1 配对并行 Pilot

**状态**：APPROVED FOR EXECUTION（本文件仅冻结后续实施授权；未声称训练已完成）  
**时间**：2026-10-30  
**执行主体**：唯一的本地 Codex，经 SSH 管理远程 RTX 5090；**不启动第二个 Codex**。  
**相对目录**：`experiments/p4_public_benchmark/p4_8_libero_pilot/index3/decision.md`。  
**上游**：保留 P4.8-r0 G0/G1 成果及 r2 Prototype Lineage Gate；以本 decision 取代 r2 中“必须原样复用人工中心原型生成规则”的阻断条款；其余架构/公平性限制仍生效。P4.7 本地真机不受影响。

## 0. 本 decision 做了什么

1. **批准且仅批准**在 LIBERO 上用 **Train35-only、无人工目标中心/ROI/Mask、无仿真器特权分割/物体真值** 的自动原型来源，解决 r2 `G2_PROTOTYPE_PROVENANCE_BLOCKED`。
2. 冻结三组：**Vanilla ACT、P1-LIBERO-Legacy（在线 DINO 双重标准化）、P1-LIBERO-Fixed（单次标准化）**。P1 两组共享同一份自动构建的四原型 bank；唯一允许的在线差别是 DINO 输入预处理。
3. 先实施自动发现的可量化门槛与数据/时基核查，再做单任务配对 Pilot、统一 Dev9 和 LIBERO 闭环 rollout；若通过，再提交新的扩大至 LIBERO-Object 10 任务决策，**本文件不授权直接扩展**。
4. 允许对照吞吐后多训练进程并行；禁止为求跑通修改 P1 核心架构、数据分割或偷偷加入额外人工监督。

## 1. 研究目标与证据边界

- RQ1：在相同 LIBERO 任务和受控训练条件下，P1-LIBERO 的 Object-Centric 路径能否带来闭环性能变化？
- RQ2：相同 P1 架构/原型/训练配方下，Legacy vs Fixed 在线 DINO 标准化差异是否影响离线或闭环表现？
- **不是**声称“历史透明杯 P1 原型构建原本完全无标注”：r2 证实它使用执行者选定的杯中心。**自动原型是 LIBERO 迁移中新引入且须披露的算法变量**，不是对历史原型来源的逐字复现。
- ACT vs P1 属于**方法级比较**（P1 额外有 DINO、自动对象先验与计算成本）；Legacy vs Fixed 才是单一在线预处理受控比较。
- 本 Pilot 的 10k 和少量 rollout 不构成正式论文性能结论；先验证工程可运行与观察趋势。

## 2. 固定数据和模型身份

- 复用 r0/r2 已验证的 LIBERO-Object task index 24（alphabet soup 目标放入篮子），Train35 / Dev9 episode ID 及已有 source/data/环境 manifest；绝不从 Dev9、仿真测试 episode、评估 reset 图像、Final 数据构建原型。**先用实际 manifest 核实 task / split / hash，不能以文字代替证据**。
- 本地已有仿真 EGL、相机/状态/动作协议和控制帧率检查继续继承，保持相同重置序列及评估随机数。
- Vanilla ACT：LeRobot 官方原生 `ACTPolicy`（无 DINO、object path、load/current、force、residual）；仅必要 LIBERO 的双相机/8D state/7D action 接口适配。保存 default config、effective config、每一项 override。若默认训练超参与公平配对配方不同，明确区分 `ACT-default` 与 `ACT-matched`；不得把 matched 冒充完全默认。若确实不同且预算允许，两个版本各自训练，否则提出 PROPOSAL 冻结论文比较范围。
- 两个 P1：保留 ACT 主干及共享 ResNet 实现、Top-only frozen DINOv2-S、4 个 prototype、1 global+4 local tokens、固定 2×2 local 分区、cosine max 与 tau=0.20、387→256→512 adapter、Q=ACT encoder token 序列的 cross-attention、未限幅标量 gate、CVAE/ACT decoder；只因 LIBERO 接口换成合法的 8D observation / 7D action，**不得填造 load/current**。版本所依代码与 config 必须逐项比对。
- Prototype 的 DINO 生成路径统一来自 raw RGB 单次标准化，Legacy 与 Fixed 共享相同 bank 字节及 SHA256。在线 Legacy 才保持历史双重标准化，Fixed 在线单次。
- P1 两组训练 seed、初始化映射、Train35、batch、优化器、lr、scheduler、augmentation、数据顺序、normalizer、updates、eval samples 以及 rollout 状态配对；不确定时标注而不得声称 bitwise identical。

## 3. G2.1 自动原型发现：批准的最小实现范围

**严格禁止：** 人工点击/输入目标中心、人工 ROI/Mask、图像语言模型人工指定的盒、仿真器提供的物体位姿/分割/目标类别标记、Dev/测试图像、训练期人工挑最像目标的四张图。任务文本只可作为数据集任务索引，不得由人工依据某个目标图像额外筛选 patch。

**建议的确定性 Train-only 方案（属于新增迁移算法，须单独命名及审计）：**

1. 使用原始 Train35 演示及合法公开的 RGB、state、action、gripper 观测，统一抽样规则和固定随机种子。只通过**已存在的夹爪运动/状态与连续图像变化**产生候选帧与 patch；不能直接依赖测试任务成败。对于缺乏可靠接触/夹爪事件标志的任务，先做诊断，不可臆测有力觉。
2. 对候选时段的 Top RGB 提取 frozen DINO patch 特征，使用相邻帧视觉变化、与夹爪邻域的几何接近性及跨帧特征一致性构造**可解释的候选分数**。所有阈值、坐标变换、每帧 patch 网格、时段选择和评分权重须在正式 Dev9/rollout 前冻结在 `prototype_algorithm.yaml`，不得按性能回调调节。只使用训练演示，不使用 simulator privileged state。
3. 自动筛除明显夹爪/机械臂主导且背景静态的候选：若在现有可观测量中无法客观区分，不要伪造无监督准确率，转入 BLOCKED/PROPOSAL。可以使用完全由机器人 state/动作和公开视觉几何算出的 gripper proximity proxy，但须披露其不完美性。
4. 在固定种子和 Train-only 候选中，通过确定性多样性选取 **4** 个代表性 patch/小邻域；每个候选用同一 DINO 特征平均 + L2 norm，bank `shape=[4,384]`。不得直接用透明杯的原型或四个中心坐标。输出每个 prototype 的 episode/frame/patch 位置、得分、过滤原因、SHA256、Train membership 证据。
5. 使用相同原型 bank 启动 Legacy/Fixed；不允许为了某个版本独立调最优 bank 或对在线 Legacy 路径使用双重标准化来生成 prototype。

**可行性 gate（不是事后人工标注）：**
- 生成过程可重复：同代码/同 seed/同 Train35 两次得到 bank SHA 一致；`[4,384]`、有限值、近单位范数，bank source 全在 Train35；无任何 Dev9 与仿真 GT 访问。
- 自动候选有效性报告（无标签 proxy，不冒充目标定位精度）：各原型的选择帧、特征相似度分布、时间稳定性、不同源 episode 覆盖、候选与末端运动的空间/时间一致性、静态背景和夹爪邻域负例响应。预先固定静态背景/机械臂否定测试及阈值；若无法通过，出 `G2_AUTO_PROTOTYPE_UNRELIABLE` 和阻断提案，**不能强行宣称找到了 alphabet soup**。
- 允许 Codex 自动生成可视化 montage 作复核材料，但**不允许以人眼挑帧/修图作为构建步骤**；用户无需人工打标。
- G2 gate 通过只代表构建机制一致且无明显塌缩，不代表已经证实物体语义定位正确；明确保存不确定性。

## 4. G2.2 数据时基/模型接口 gate

- r0/r2 指出 **数据时间戳 10 FPS vs 仿真控制 20 Hz**，必须检查官方 dataset/action normalization 与 env step 具体语义和原生 LeRobot LIBERO rollout 行为；按合法公开协议冻结 observation/action 执行窗口、chunk 与时间跨度，**不得凭猜测重复/插值动作**。若无法解释，阻断闭环正式对比并提交 PROPOSAL；可先执行完全可逆的工程 smoke。
- ACT 与两 P1 同样的 train/dev 划分、相机序列、归一化及 action 物理意义；预处理 DINO raw top 和 ACT CNN 的不同输入路径分别核验，确保 Fixed 不改 ResNet。
- 两个 P1 以同一批样本验证在线 DINO 的前向数值确有预期差异；prototype bank bytes 相同；其余 state/action 和 main visual backbone 相同。
- 在不动上游代码的独立工作区运行实际 batch forward/backward、无 NaN、100-step checkpoint restore smoke、离线 Dev9 和 simulator reset/step；不得因成功跑通就声称任务成功。

## 5. G3 训练：相同单任务 Pilot，并行调度

- 计划 `Vanilla ACT`、`P1-Legacy`、`P1-Fixed` 各 **10,000 effective optimizer updates**，同 task、Train35，独立权重、日志及完整 checkpoint；epoch、batch、lr 在 effective config 中记录。10k 为工程 Pilot，严禁套用 SO-101 历史 checkpoint。
- 在预算上先 smoke 100 updates，再实测单任务、双任务、必要时三任务并行的**有效总更新速率**、内存和 I/O；只在总完成吞吐有实际收益且每作业无资源退化/数据竞争时采用多并行。三组任务 checkpoint/缓存/run_id 隔离，所有后台任务可独立恢复，SSH 断开不终止。
- 先保证 Legacy 与 Fixed 两个 P1 **同训练预算完整配对**，同时保证 Vanilla ACT 可用的对照；若资源不允许三者 10k 完成，客观记录完成边界，停止拿不等步数做结论，提交补续 PROPOSAL。
- 参数初始化/seed 的可比性与 DINO 固定参数冻结，记录 model SHA、bank SHA、data revision、trainer code commit，曲线和训练实际时间。

## 6. G4 统一离线与闭环仿真

- 相同 Dev9 记录：Arm/Gripper 分项预测误差、必要时 per-dimension normalized/raw 指标；不要用不兼容 action horizon 的 loss 直接比较。
- 同一初始状态、同一仿真控制频率和最大步数，ACT/Legacy/Fixed 各 **10 次 paired rollout**（Pilot）；记录 init seed、reward/success 事件、实际轨迹、失败阶段、墙钟和推理时延。没有运行不得伪造成功率。
- Legacy/Fixed 补充 Object Path ON/OFF 动作响应、gate、tokens 和 patch score；**ON/OFF 不等价于正式从零训练消融**。
- 输出 `pilot_only=true`，不给出“通用性能领先”的论文结论；扩大 LIBERO-Object 10 任务和多 seed 需单独下一份 decision。

## 7. 并行执行与不干扰

- **唯一 Codex**：本地通过 SSH 配置/启动/查看服务器作业，随后可处理本地 P4.7；无额外 Codex Agent、无服务器交互式长期占用前台。
- 不修改 P4.7、SO-101 部署、HF 原有模型和历史 P4.8 G0/G1 文件；P4.8 使用新的 index3 和隔离缓存，保留祖先结果。
- 原先 r2 的 BLOCKED 记录是合法的历史事实，不删除或改写；本 decision 是之后的明确授权，不得将其伪装为 r2 已经训练成功。
- 任何隐藏人工标注、使用仿真器特权信息、源数据泄漏、模型语义变化、时基无法合法对齐或服务器任务冲突 -> 立即停下对应轨道，产出 `blocking_proposal.md`；其他独立任务可以继续。

## 8. 交付结果

`result/` 至少应包含：`summary.md`、`prototype_algorithm.yaml`、`prototype_bank.pt`、`prototype_provenance.json`、`prototype_proxy_diagnostics.json`、`prototype_visual_montage/`（如生成）、`dataset_split_manifest.json`、`timing_and_control_contract.md`、`code_identity.json`、`effective_config_*.json`、`throughput_benchmark.csv`、`training_metrics.csv`、`dev9_metrics.csv`、`paired_rollouts.csv`、`object_path_diagnostics.csv`、`checkpoint_manifest.csv`、`builder_review.md`、`redteam_review.md`、`blocking_proposal.md`（如有）、`artifact_manifest.json`。checkpoint 可以只提供服务器路径和哈希，但必须可被唯一识别及恢复。

报告四类结论：`G2_AUTO_PROTOTYPE_PASS/BLOCKED`、`TIMEBASE_PASS/BLOCKED`、`TRAIN_PILOT_COMPLETE/PARTIAL/NOT_STARTED`、`SIM_ROLLOUT_COMPLETE/BLOCKED`，并明确下一次研究决策需要验证什么。

## 9. Codex 执行边界

只执行这份已冻结的研究语义；可以实现最小必要代码和工程自检，**不得自行替换成 KMeans 全图、假造目标真值、补人工 ROI、退化成去掉原型的 P1、擅自增加模型或扩大任务套件**。算法无法在 Train-only 数据上可靠产生目标相关候选时，提供最小证据和备选提案，由用户与规划助手下一轮决定，不要无期限调参追求通过。
