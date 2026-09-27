# LeRobot 项目当前状态

> 更新时间：2026-09-21（C0/C1/C2 P3 完成；修正版轨迹方向 Gate 判定 C1 通过）。这里保存当前快照，不保存完整历史。历史过程见 `../docc/日志/工作日志.md`；长期执行规则见 `leorbot_workflow.md`。

## 当前阶段

- 主线：`P6` 首轮探索性真机测试已完成，但没有按杯位配对，也没有判定表，不能作为正式 round 2 成功率对照。
- C0 显式杯框条件支线：`P3 FAILED / DIAGNOSED`。10k检测坐标正确、坐标MLP确有更新，但策略从1k起始终几乎不使用独立box token；完整置零/反转、全horizon与坐标-only反事实均确认，不晋升100k。
- C1/C2 显式杯框条件对照：`P3 C1 PASSED / C0-C2 FAILED`。修正版完整轨迹方向 Gate 已完成：C1 在4k/6k/8k/10k均通过，C0最终span不足，C2仍忽略box。仅C1进入P4候选，100k尚未授权或启动。
- 水面关键点支线：`P2`，代码与真实数据工程联调完成，人工标签不足。
- 完整透明杯 MASK 支线：暂停晋升，现有画质和遮挡下标签语义/时序稳定性不足。

## 已完成

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

1. 为 C1 单独建立 P4/100k 合同：固定 fit32、seed、batch、schema、预处理、评估和 checkpoint 频率，并保留旧 DET-NM/C0 作为对照。
2. 在合同冻结前补 C1 的 Jacobian、box-shuffle 和 normal/zero/reverse 证据包；合同通过审查并获授权后才启动100k。
3. C0/C2 暂停；C2 不再通过增大残差权重推进。
4. 真机前新增夹爪 `pos/load/current` 标定记录；低层状态机只在接触确认后允许抬升，并在固定放置区域才接受释放请求。
5. 正式 round 2 必须恢复杯位配对、判定表和完整阶段记录；探索性31段不混入正式成功率表。

## 当前阻塞与限制

- C0 P3的检测器与坐标MLP已学习，但动作策略基本忽略显式box token；当前C0结构不得进入100k。10k结果只用于工程诊断，不与正式100k模型排名。
- 夹爪接触阈值尚无真机 `load/current` 标定数据；不得复制外部阈值或直接启用自动收紧。
- 水面关键点现有可见人工点数量不足，不启动正式100k训练；遮挡中心不得插值或使用SAM漂移中心代替。
- 固定val10已用于正式比较，后续不应继续把它当开发调参集。
- 服务器大型raw输出受租用实例生命周期影响，关键汇总已同步本地，但仍需保持证据清单。

## 2026-09-27：Formal3 四路线训练准备

- `formal3_kind_merged_nomaster_fit48` 已发布到 Hugging Face 数据集 `QYyyyyyyy/formal3_kind_merged_nomaster_fit48`：60 episodes、35,917 frames，包含双相机视频、NOMASTER 数据、训练48集统计、top检测标注、固定48/12划分和最终人工阶段标签。
- 四路线合同位于 `outputs/formal3_mixed_p4_20260927/experiment.md`，状态为 `PREPARED`：纯 Diffusion、DET基线、C1 state-box显式坐标注入、DET+phase(weight=0.10)。C1沿用已通过Gate的 state token加法，不使用未通过Gate的C2 action residual。
- 四路 batch=8 的真实数据并发2-step预检全部退出0，无OOM/NaN。用户确认后，正式100k训练已在服务器 `screen -S formal3_mixed_p4` 启动；四个主训练器均越过首批步骤，总显存约27.5 GB，预计以最慢ACTDet计约10–11小时。

## 关键入口

- 工作流：`leorbot_workflow.md`
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
