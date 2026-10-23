# LeRobot 项目当前状态

## 2026-10-23：R5五模型已发布，直接真机指南与代码就绪，现场未运行

- 当前为[R5 r0](../experiments/force/r5_pre_final_dataset_real_robot_gate/index1/decision_R5_PRE_FINAL_DATASET_REAL_ROBOT_GATE.md)，在最终数据采集前做真实定位gate；不是直接进入120–150段采集。冻结M1旧G36-S3、M2新G60-S3、M3旧New-GL、M4新New-GL、M5弱seed4；M5依据R4综合误差/空间诊断预先选择，不看真机换seed。
- 模型与配套文件从服务器发布到[HF R5_PRE_FINAL_DATASET_SO101](https://huggingface.co/QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101)，revision6990fb2e82c39a571671fd22f7f45979f9d9ade6；11文件远端size/LFS SHA/Git blob核验通过。模型包962550207字节、代码包929487字节；含3global/2local、各自normalizer、prototype和离线视觉缓存，无需完整formal4视频。[发布回执](../experiments/force/r5_pre_final_dataset_real_robot_gate/index1/result/hf_publish_receipt.json)。
- [真机指南](../experiments/force/r5_pre_final_dataset_real_robot_gate/index1/REAL_ROBOT_TEST_GUIDE.md)末尾五个独立可复制命令，沿用ACM0/id=nn、相机4/2、CLI ensemble0.01/relative5；固定A/B/C分块轮换共15次，不做失败救援重复。新control_r5继承既有直接运行循环，无新增shadow/审批/STOP锁，保留物理限制。
- 保持benchmark=False和TF32=True；旧在线float64-rounded load与R4 float32缓存差3.725e-9在真实样本引起0.004738动作差，R5适配器已对齐float32。相同输入的独立模型差0、参数相等；未改权重/统计量/门槛。五路径真实RGB batch1 nominal差0、local参考/arm隔离/界限/reset通过；15本地回归通过，不冒充闭环真机。
- R5_COMPLETE=false，现场0/15。下一步用户在连接设备的Linux系统按指南执行，再交视频/CSV/P0–P6/辅助记录；不采新训练数据、不重训、不自动操作机械臂。用户已授权本轮相关项目代码/指南更新GitHub det，凭据不入文件，大模型不入Git。

## 2026-10-23：R4全量复现通过，修复后结论稳定，result已交付待外审

- [复核result](../experiments/force/r4_inference_reproducibility/index1/result/summary.md)、[交付ZIP](../experiments/force/r4_inference_reproducibility/index1/R4_REVIEWED_result_20261023.zip)、[下一步proposal](../experiments/force/r4_inference_reproducibility/index1/result/proposal.md)已准备。REPRODUCIBILITY_FIXED_RUNTIME；R4_EXECUTION_COMPLETE=true、R4_REPRODUCIBILITY_PASS=true，R4_COMPLETE=false仅因独立外部Builder/Red-Team待验收，不代表技术门未通过。
- 最终仅cudnn.benchmark True→False，matmul/cuDNN TF32仍True。S3和S1各两次Full60/28739帧/7185批的100 horizons×6动作逐元素相等，input manifest一致，max=0，保持原<0.001门槛。首个观测到不同的hook为backbone，未捕捉具体kernel ID，不泛化保证其它机器或真机ensemble。
- Fresh G36/S1/2/3/4及冻结local1/2/3关键指标已重算；global seed差异和Full60收益混合结论未翻转，自动描述标签仍FULL60_SEED_VARIANCE_DOMINATES；local固定global误差范围0.304030–0.318368。全部仍是Full60训练/共同诊断，不是held-out或真机。
- 本轮另在服务器CPU重新读取两seed全部D0/D1 raw chunk，核对h0缓存/输入/IDs，复算5global误差，重hash12 checkpoint文件并交叉核对7global/local身份，PASS。原下载103360字节ZIP及manifest、复核75成员ZIP的CRC/SHA通过，原证据保留。GPU空闲、盘约36GiB。
- 下一步先交网页GPT独立审阅，再制定FINAL_DATASET_V1采集/QC/split-freeze合同（约120–150段、位置抖动、明确0.8–1.0秒hold）；本轮不实际采集、不追加旧60训练/SWA/SAM/真机，不改实际部署参数，无Git/HF发布。

## 2026-10-23：R4已复现跨进程异常，benchmark单变量对照通过，后台全量审计中

- 当前合同为[R4 r0](../experiments/force/r4_inference_reproducibility/index1/decision_R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE.md)：不训练/新结构/采集/真机；按A同张量同进程→B跨进程→C预处理→D全量回放逐层准入，保持<0.001。服务器weste:20143，目录/root/autodl-tmp/R4_20261023。
- 历史cache原差0.117225647重现（ep3/frame77）。A三个固定批次重复差0；B相同input hash却复现完整chunk最大0.327590942、h0最大0.130935669，最早观测不同hook为model.backbone。仅关闭cudnn.benchmark、TF32保持True后A/B/C差均0，支持runtime算法选择相关原因，未捕捉具体kernel ID，仍需全量验证。
- S3第一次D全60/28739帧已完成，D1后台进行；随后seed1 A–D、fresh G36/S2/S4及固定local重组，沿用冻结位置/匹配/分类规则重算关键指标。nohup队列与result follower独立于SSH；失败停止、保留原证据，15GiB磁盘余量门；为隔离运行时影响审计串行，不启动训练或其它模型实验。
- [运行入口](../experiments/force/r4_inference_reproducibility/index1/RUNNING.md)、[阶段summary](../experiments/force/r4_inference_reproducibility/index1/result/summary.md)。部分原始CSV/配置/fixture manifest已下载evidence；本地8项CPU门槛/配置/既有分析回归及diff检查通过，不冒充GPU全量通过。
- R4_COMPLETE=false，R4_REPRODUCIBILITY_PASS尚未建立；两seed D通过且重算结束后后台自动整理执行方result，外部Builder/Red-Team仍待审阅。不自动进入120–150段数据采集、SWA/SAM、训练/真机或发布，不改旧F5/实际部署参数。

## 2026-10-23：F5.1-r3规划result交付，等待外部审阅

- 已整理[复核summary](../experiments/force/f5_1_full60_data_utilization/index1/result/summary.md)、[下一步proposal](../experiments/force/f5_1_full60_data_utilization/index1/result/proposal.md)与[交付ZIP](../experiments/force/f5_1_full60_data_utilization/index1/F5_1_r3_REVIEWED_result_20261023.zip)，可直接交网页GPT做Builder/Red-Team及下一decision；外部审阅尚未进行，不等于合同通过。
- 明确披露矛盾：S3位置匹配距离+10.31%、边缘幅度偏差降低60.82%，但G60-S3整体夹爪/arm回放误差较G36分别增加13.55%/2.35%；NewGL60-L3较历史New-GL-36夹爪误差增加24.67%。三local相对同global稳定改善；全部是Full60训练/共同诊断，不是held-out泛化或真机效果。
- 重复性门槛失败保持原判定（0.117225647 > 预设0.001），另发现合同§20另一seed扩展repeat审计缺证。下一步优先冻结权重核对身份/输入/预处理/forward差异并补证；不能凭推测归因TF32/并行、放宽门槛或直接追加训练。
- 原服务器62文件逐字节副本保留；61成员SHA、原ZIP/交付ZIP CRC及SHA核验、CSV交叉复算和五张实际图检查完成。启动时summary/metrics/validation另存r3_start历史。计算结束、交付READY、验收false分别记录。本轮无新GPU实验、真机、阈值/部署修改或Git/HF发布。

## 2026-10-23：F5.1-r3计算链结束，重复性门槛未通过

- weste:20143后台约03:14结束（约4h47m），queue.exit=0、RESULT_READY_AWAIT_EXTERNAL_REVIEW；19已启动任务全exit0，无重试。G60-S1/2/3/4各80k、固定G60-S3的local1/2/3各7k、可选B0 seed3 100k均完成，P1可选未做。S1/S2确实并行。
- fresh回放/空间/seed/机制/重现/runtime/执行方报告均已运行；不能宣称全验收通过：metrics的F5_1_R3_EXECUTOR_COMPLETE=false，重复fresh最大native差0.11722564697（global和New相同）超预设1e-3门槛。原因未定位，不改门槛、不归因并行、不自动重训。arm/方程/界限及机制检查通过。
- 自动描述性outcome FULL60_SEED_VARIANCE_DOMINATES：seed3空间响应改善但seed差异较大，仍属Full60训练/共同诊断，不是真机结论。外部独立审阅PENDING。
- 10:04检查GPU0%/2MiB，盘可用约36.67GiB。下载完整ZIP到results/experiments/F5_1_FULL60_MULTI_SEED_OPTIMAL/20261023T020523Z，2793822字节，SHA4d0403dc4e7173a4e346e8d64c7c90aa3904e972aadbc2c81511c1e60e48717b；CRC/61成员SHA通过。[检查记录](../experiments/force/f5_1_full60_data_utilization/index1/SERVER_CHECK_20261023.md)。下一步建议先定位重复推理差异；本轮只检查，不重训/改代码/新增评估/真机或Git/HF发布。

## 2026-10-22：F5.1-r3已清理并在新服务器后台启动

- 已连通weste:20143，RTX5090/32GB。核验完整解压内容及HF已发布SHA后仅删除962473497字节F5临时传输包，释放约918MiB至45.58GiB；唯一模型/数据/日志/预测0删除，必要视觉缓存保留，归档可从HF同revision恢复。准备后可用约45.29GiB，按20GiB新增+15GiB余量准入，预计仍25.29GiB。
- 服务器数据哈希/60段meta、旧组件SHA、配方一致性、真实GPU batch4前后向/更新/保存重载通过，正式从头初始化；冻结历史源/recipe379hash及分析规则，30本地回归通过。报告合成检查通过，不是科研证据；缺scipy改用等价NumPy最近邻，训练依赖与科研语义不变。
- 工作目录/root/autodl-tmp/F5_1_r3_20261022，nohup队列PID3185、初始worker3189，G60-S3已实际到step300且loss有限，GPU瞬时约3276MiB/62%。先核心seed3+local3+paired，再预算准入G1/G2并行及local1/2；重复回放/runtime优先，余时G4/B0。9h最大窗口、1h报告预留、20秒空间检查，SSH断开继续。
- [运行入口](../experiments/force/f5_1_full60_data_utilization/index1/RUNNING_R3.md)；[当前result](../experiments/force/f5_1_full60_data_utilization/index1/result/summary.md)。结束自动统一fresh诊断/执行方审查/ZIP，外部独立审阅待交付；不自动真机或进入新decision，未Git/HF发布。

## 2026-10-22：F5.1切换r3多seed合同，新服务器等待认证

- 当前合同为[Full60 multi-seed r3](../experiments/force/f5_1_full60_data_utilization/index1/decision_F5_1_FULL60_MULTI_SEED_OPTIMAL_r3.md)，取代r1执行计划：优先G60-S3/80k+固定global上的L60-S3/7k及paired诊断，再G60-S1/S2与L60-S1/S2；S4/B0按余时准入，候选不按训练误差挑seed，不真机。
- 用户新服务器weste:20143可达，但已有项目SSH key被拒绝，未提供当前实例密码；未能登录，尚未清理/查看GPU或磁盘/启动训练。需密码或公钥授权。旧r1本地数据审计保留，不把其单seed预备protocol作为r3启动配置。
- 用户授权训练前清理无用文件；获得连接后先盘点活动进程、精确清理路径、重复模型身份与峰值磁盘预算，保留唯一模型/原始数据/日志与必要视觉缓存，不因结果差删除证据。[准备记录](../experiments/force/f5_1_full60_data_utilization/index1/PREPARATION_R3.md)。当前REMOTE_AUTH_REQUIRED，无文件删除、训练、真机或Git/HF发布。

## 2026-10-22：F5.1-r1 Full60本地审计通过，服务器连接受阻

- 当前合同为[decision F5.1-r1](../experiments/force/f5_1_full60_data_utilization/index1/decision_F5_1_FULL60_DATA_UTILIZATION_TEST_r1.md)：同Strong-G架构seed3训练Full60 global80k，再冻结global训练local7k，自动完成离线回放/空间诊断/机制/重现/就绪与打包，不真机。固定80k/7k，不回放选模，不增seed。
- 本地Full60审计通过：60段/28739帧，D1冻结12个位置组×5段，state8/action6/单帧load有效，4个视频全量解码57478帧无错。生成Full60归一化与60份输入数组（outputs/f5_1_preparation_20261022/），未修改旧split/统计；历史训练ensemble=null，不将真机0.01混入重训。
- **本r1明确授权Full60读取历史Dev12/Final12**，准备阶段已读取这24段；新模型的Full60回放只能称TRAIN-SET REPLAY / NOT HELD-OUT，不称泛化。旧实验历史证据不改。
- 旧westd:47382拒绝连接，已请求用户启动实例/提供新SSH；GPU烟测、远端资源/recipe核验、后台队列和训练均NOT_STARTED。准备脚本4项回归通过；[当前result](../experiments/force/f5_1_full60_data_utilization/index1/result/summary.md)记录实际准备状态，不冒充完成。F5_1_R1_COMPLETE=false，未Git/HF发布、未真机。

## 2026-10-22：F5定性选型result已整理，建议New-GL主线

- 用户澄清当前不是成功率评测，而是观察当前效果、选择后续主模型并交网页GPT规划；不再把补齐12次正式试验设为本次选型前提。原F5_COMPLETE=false仅保留正式合同事实，不否认本次选型交付完成。
- [result/summary](../experiments/force/f5_unified_real_robot_validation/index1/result/summary.md)建议New-GL作为开发主线、Strong-G核心对照、P1-Hist历史参考、L-Load退出优先投入；这是可撤销的工程推荐，尚未证明New真机胜过Strong。New/Strong/P1抓取可用，New初始位置泛化是重点短板；P1保持问题不无证据推广到所有模型。
- result含summary/proposal/metrics/validation、13视频审阅图与20原CSV；旧部署准备summary/metrics/validation另存deployment_preparation_*，不再以历史“0次真机”混淆当前阶段。网页GPT优先规划共享global的空间对准诊断，再调查动作链与local阶段作用，不直接追加架构/重训。
- 本轮只整理结果与建议，无新测试/训练/真机/部署改动/Final12/Git或HF推送；待用户与网页GPT确认下一decision。

## 2026-10-22：F5视频/CSV阶段复核，人工结束不等于抓取失败

- 已从GitHub det快进到c9e77fab，保留无关未提交修改。13段手机视频全部7348帧可解码；20份CSV含_prev_全部复算，与manifest行数/字节一致。检查为每视频20帧概览+6个2秒10fps窗口，并非全帧人工精审；视频与CSV无可靠attempt配对，不做伪同步。
- 用户现场观察：P1可抓但抬起无保持/下降才停；Strong初始定位偏而抓取不错；L缺可靠抓取；New抓取不错但换杯位易丢目标。视频有相符的动作链和人工送杯/取杯证据，New15辅助恢复尤其明确；不以人工结束判失败、不把干预后的抓持算自主定位成功。全部CSV phase=A不代表全段自主。
- 有效采样约28.85–29.62Hz，无持续低帧率证据；local确实输出且界内，final=nominal+delta误差<3.82e-6、arm隔离。L最新nominal夹爪52.707→43.156，对照其余最新闭合低点约20–22，提示global闭合链不足；L最长旧尝试曾到18.084，不能说所有L都不闭合。控制限幅介入不是原因证明，未取消安全参数。
- [复核报告及建议](../experiments/force/f5_unified_real_robot_validation/index1/result/real_robot_review/VIDEO_CSV_REVIEW.md)。优先补现场配置/视频配对，再分开诊断空间对准与抬升保持链；保留New/Strong对照、P1历史参考，L先查global不盲目扩local bound。正式A/B/C未完成、无成功率/泛化排名，F5_COMPLETE=false。无训练/部署改动/新真机/Final12/Git或HF推送。

## 2026-10-22：F5 真机 A 轮结束，控制CSV归档；阶段性结果未判定

- 用户确认本轮真机测试结束。`outputs/f5/session_01/` 共4个模型（A点位）20份控制CSV，已复制归档到[real_robot_control_csv](../experiments/force/f5_unified_real_robot_validation/index1/result/real_robot_control_csv/README.md)并附`manifest.json`；视频/首帧/summary/stage_review仍留在本地`outputs/`（gitignore）。
- 事实记录：20次尝试状态全为`SAFETY_ABORT`（操作员中止）或`ERROR`，无一次自然跑满60秒；另有3次启动即失败（相机/连接）未生成CSV。每个模型最新一次步数：P1-Hist 580、Strong-G 811、L-Load 287、New-GL 507。
- 所有`stage_review.json`仍为`UNREVIEWED`，**没有任何成功/失败判定**；B/C轮未执行，12次正式预算未完成，`F5_COMPLETE=false`。不得把本批数据写成模型优劣或抓取成功率结论。
- 同时把"结果不得覆盖、必须可重复"写入指南第6节；重复试次沿用`_prev_`归档、汇总跳过旧尝试的机制（21项回归）。本轮更新提交GitHub det分支，不纳入无关未提交修改。

## 2026-10-22：F5 重复试次不再被同名目录阻塞

- 用户现场重跑同一试次时报 `FileExistsError`。`control_f5.py` 改为：同名目录存在时自动改名为 `<TRIAL_ID>_prev_<时间戳>`（证据保留、不删除），新试次照常写入原目录；`summarize_f5.py` 跳过 `_prev_` 目录，避免旧尝试重复计入12次表。指南第6节与[真机经验参考](../docc/指南/真机测试经验与代码指令参考.md)同步更新。
- 回归21项通过（新增重跑归档、manifest 单行、汇总排除 `_prev_` 三项断言）。未改模型、权重、归一化、相机语义或部署参数。
- 现场状态：`session_01` 已有 P1-Hist A（522步后人工中止，`SAFETY_ABORT`）与 Strong-G A 的一次 `ERROR`（相机4读帧超时，0步）；两路索引事后探测均可出帧，判为进程退出后设备未及时释放。**两试次均未做 stage_review，不构成成功/失败结论**；正式12次未完成，F5_COMPLETE=false。

## 2026-10-22：F5真机前本机预检完成，硬件未连接

- 按[F5指南](../experiments/force/f5_unified_real_robot_validation/index1/REAL_ROBOT_TEST_GUIDE.md)完成真机前步骤：`huggingface.co`直连TLS被重置，改用`hf-mirror.com`下载；镜像revision `4d680b2552c0257f3a9989bde45e54bedb98aa5c`与发布回执一致。模型包962473497字节、SHA256 `9187d7e8…cca9`校验通过；代码包5个文件与当前det逐字节相同，未覆盖仓库；解包至`outputs/deploy/F5/`。
- `verify_bundle()`四模型身份/manifest哈希全部通过；F5/F3回归20项通过（`lerobotnn`原缺pytest，本轮为跑回归安装）；四模型`--dry-run`均exit 0、动作有限，L-Load/New-GL局部修正−4.8288/−4.8290落在±4.832699界内。合成零观测只证明接线与限界，不是行为或闭环证据。
- **未连接真机**：本机无`/dev/ttyACM*`；仅`video0/video1`且属同一UVC摄像头，指南top=4/gripper=2无法核验。正式试次0/12，F5_COMPLETE=false，未把离线预检当现场证据。
- 待现场决定：`nn.json`与`nnn.json`两套标定关节语义不同（homing相差约2000 counts、gripper量程不同），代码按`--robot.id`静默加载；指南已标注，未擅自修改部署参数。记录见[preflight_local_2026-10-22.md](../experiments/force/f5_unified_real_robot_validation/index1/result/preflight_local_2026-10-22.md)。

## 2026-10-22：F5真机文件与四模型已发布Hugging Face

- 已从服务器上传并核验[QYyyyyyyy/F5_UNIFIED_SO101](https://huggingface.co/QYyyyyyyy/F5_UNIFIED_SO101)，revision `4d680b2552c0257f3a9989bde45e54bedb98aa5c`。含P1-Hist/Strong-G/L-Load/New-GL精确模型、normalizer、prototype、离线视觉缓存，以及配套F5代码包、指南、身份与离线验证证据。
- 模型包962473497字节，SHA256 `9187d7e8db452fa468c3d6930558fb1e6ba9d313bdd52a0a454cf3c6c6f0cca9`；代码包16076字节，SHA256 `8489f180f69f9695df8bd5ff9154f0a08bfc9671a564969b1692c03f3190a74d`。8个远端文件逐项大小及LFS SHA256/Git blob SHA1核验通过；[发布凭证](../experiments/force/f5_unified_real_robot_validation/index1/result/hf_publish_receipt.json)。使用AutoDL network_turbo解决直连超时，令牌仅stdin传递，无磁盘凭据文件。
- [指南](../experiments/force/f5_unified_real_robot_validation/index1/REAL_ROBOT_TEST_GUIDE.md)已增加HF下载与解包指令，不再依赖云服务器持续开机；四条真机指令保留CLI可调ensemble/relative限制。本轮更新GitHub det分支的F5代码、指南和项目日志，不纳入无关实验修改、模型权重或数据集。
- F5_COMPLETE=false，真机0/12、Final12未读取；发布文件不等于完成真机验证。下一步是在接臂Linux系统统一冻结设置、杯位与初态，按指南执行12次正式试次。

## 2026-10-22：F5部署准备，控制参数支持命令行修改

- 新增F5统一入口 `src/lerobot/scripts/control_f5.py`；`--policy.temporal_ensemble_coeff`与`--robot.max_relative_target`从CLI传入实际策略/机械臂配置，默认0.01/5，不强制固定值；实际参数写入每试次configuration.json。正式四模型比较前统一设置，不按中途结果单独调参。
- [真机指南](../experiments/force/f5_unified_real_robot_validation/index1/REAL_ROBOT_TEST_GUIDE.md)末尾有四模型独立指令；本地20项F5/F3相关回归及git diff --check通过。命令解析、实际参数转发使用模拟I/O验证，非真机验证。
- 服务器已准备四模型部署包并通过真实Dev首帧GPU冒烟；检查范围是首步/reset、局部分支隔离及有限输出，不证明闭环效果。F5_COMPLETE=false，真机试次0；未Git/HF发布。ACT相对早期训练源码新增可选load gate分支，源码并非逐字/AST相同，不据此声称源码完全一致。

## 2026-10-22：F4.2完成，New-GL成为首选离线双路候选

- 按F4_2 r0用精确Strong-G global seed3/80k重新回放Train36/Dev12，重新拟合Train-only nominal统计，从头并行训练New local seed1/2/3各10k；同MLP/输入/Adam/.001/batch256/1k保存/bound4.83269946，无global微调。三十checkpoint全部保留，global和Old组件前后SHA不变。
- 主结论STRONG_GLOBAL_GL_LOAD_SUPPORTED，F4_2_COMPLETE=true为执行方合同交付完成，Preferred-GL=New-GL：global seed3/80k + New local seed3/7k，local SHA695d6fa89f3e823f6236dca38c8e684c461009d5b9c357cfd1cb9f0d8dffea81。保留Old-GL、Strong-G、最佳L-Load及P1-Hist历史引用，未把旧local接新global。
- 同批fresh Dev5748帧：Old-GL primary0.03306074/native arm2.96571882；Strong-G 0.07200311/2.28577658；最佳New-GL 0.01472233/2.28577658，native gripper0.366691。New较Old夹爪误差下降55.47%、arm下降22.93%，12/12段夹爪均更低，较Strong夹爪下降79.55%，无需事后“允许退化”阈值。全部三个New local arm逐元素继承Strong，不是真机动作/成功率证明。
- New D0/D1全局中和/D2局部中和/D3双中和primary=.014722/.045578/.054687/.077793，两路都有使用；global h0 arm变化.034786个归一化单位。New mean|delta|1.551723、近界0.47%、Dev目标裁剪6.47%、Train裁剪1.08%、0越界（Old mean|delta|2.443081、近界6.30%、Dev裁剪18.46%），更强global减少局部修正负担。New模型级中位12.08ms/74,957,064参数，非端到端时延。
- 独立验证5模型主指标/60条逐episode、4预测hash、30候选扫选、3种子arm隔离；服务器复核所有1437 raw批/100 horizons、Train17243帧归一化及5%界限、组件SHA和30权重保留；48回归、六图视觉、diff/ZIP CRC和成员SHA通过。原始数据和日志不删除。
- [summary](../experiments/force/f4_2_strong_global_gl_load_rebuild/index1/result/summary.md)；[ZIP](../experiments/force/f4_2_strong_global_gl_load_rebuild/index1/F4_2_result_delivery_20261022T030808646422Z.zip)（407793字节，SHA1737a9f81f875fa445add7f677582d84d8dba9ad2cd66b686cc6ef395579e0f0）。本地canonical results/experiments/F4_2_STRONG_GLOBAL_GL_LOAD_REBUILD/20261022T030808646422Z（67文件）；服务器/root/autodl-tmp/F4_2_20261022保留30权重/新Train-Dev缓存/1437 raw批/日志，审阅版另置delivery_reviewed/不覆盖原自动报告。
- 独立Builder/Red-Team PENDING；建议验收后停止继续离线追Dev，下一decision冻结统一真机模型池/部署与安全/同步日志协议。本轮未Final12、真机、实际部署参数改动、Git/HF发布或自动进入下一科研轮次。

## 2026-10-22：F4.2强骨干GL重建已启动

- 合同：experiments/force/f4_2_strong_global_gl_load_rebuild/index1/decision_F4_2_STRONG_GLOBAL_GL_LOAD_REBUILD.md，r0，只读。服务器/root/autodl-tmp/F4_2_20261022，nohup主PID27163、报告follower PID521038，SSH断开可继续。
- 固定Strong-G=G global seed3/80k、Old-GL=旧G seed1/60k+local seed4/7k，启动时精确SHA和共同配置核验通过。正在新鲜回放Strong Train36与同批Dev12正常/中和/Old；不用旧名义动作训练New。
- 回放后真实4帧2更新烟测，再独立初始化并行New local seed1/2/3各10k，Adam/.001/batch256/每1k保存，同3-32-32-1与bound4.83269946。新nominal归一化只Train拟合，global不进优化器。
- 自动链包含十候选扫选、Old/Strong/New fresh指标、arm bitwise隔离、global/local/dual反事实、100 horizons raw1437批、模型级时延/参数及计算报告。最终权衡分类/Preferred-GL、图检查和独立复算之后交付，不新增宽容阈值。
- 启动盘45.6GiB，预计新增不足1GiB且保留15GiB；无唯一模型/证据删除、无新global训练/seed4/current/W15/gate/FiLM/Final12/真机/部署改动或Git/HF发布。入口[RUNNING](../experiments/force/f4_2_strong_global_gl_load_rebuild/index1/RUNNING.md)，当前F4_2尚未完成。

## 2026-10-22：F4.1-r1完成，global种子敏感、固定骨干local稳定

- 后台用时8.759小时，11/11新增训练成功、无失败/重启：G-Load global2/3/4、B0-6D global2/3各100k；GL/L local2/3/4各10k。25个新增global与60个新增local checkpoint共85个全部保留。可选B0 seed4/P1-Hist为时间预留未运行，无外存中断；GPU已空闲，数据盘仍约45.6GiB可用。
- Dev12 fresh主指标均值±样本std：B0（三global种子）0.124266±0.034707，G（四global）0.111000±0.043287，L（固定旧B0上四local）0.057677±0.000147，GL（固定旧G上四local）0.033358±0.000275。GL所有local种子低于所有L；局部均值优势42.16%，不等于GL跨global种子已稳。
- 定性主结论GLOBAL_SEED_SENSITIVE，F4_1_R1_COMPLETE=true仅指合同执行与执行方解释完成。G范围0.072003–0.165142约2.29倍，seed1排3/4，较弱但非唯一最差/统计离群点；共享global seed1/2/3只赢B0的2/3，不能说global load稳定优胜或已证明历史真机根因。
- 已冻结：G global seed3/80k（primary0.072003、native arm2.285784）；L local seed4/9k（primary0.057476、旧F4 B0）；GL local seed4/7k（primary0.033060、仍旧F4 G seed1/60k，不接新G）。最佳GL相对原local seed1仅约0.415%增益，仍不是直接真机部署许可。推荐新decision下在新G骨干重新训练local，不擅自组合或开新训练。
- 核验：37个global fresh评估、15候选/180逐episode指标独立重算，51 NPZ SHA、15候选权重及85 checkpoint预算/保留核实通过；8个local arm bitwise不变。最佳GL/G A/B完整Dev重复输出差0；selection-vs-repeat并非bitwise一致，最大预测差0.001292825/0.000716329，主指标差8.01e-7/6.03e-7，明确保留。41项回归、六图视觉检查、diff/ZIP CRC和成员SHA通过。
- 最终[summary](../experiments/force/f4_1_unattended_global_local_robustness/index1/result/summary.md)；[审阅版ZIP](../experiments/force/f4_1_unattended_global_local_robustness/index1/F4_1_r1_reviewed_delivery_20261022T014830280678Z.zip)（417806字节，SHA44c73dc8184d6a91868f92ce84fbb0723d3e37ebaeeff3cd9e6c9d70688762f8）。本地canonical raw为results/experiments/F4_1_UNATTENDED_GLOBAL_LOCAL_ROBUSTNESS/20261022T014830280678Z，144文件；原后台包和51预测/14日志保留。服务器原证据/root/autodl-tmp/F4_1_20261022，最终审阅版在delivery_reviewed/，原result/作为自动生成历史版本不覆盖。
- 独立Builder/Red-Team PENDING，等待规划方验收。Final12未读取、无新真机/部署改动、无Git/HF发布；旧启动条目仅作历史。

## 2026-10-22：F4.1-r1已启动后台多种子队列

- 当前合同：experiments/force/f4_1_unattended_global_local_robustness/index1/decision_F4_1_UNATTENDED_GLOBAL_LOCAL_ROBUSTNESS_r1.md，只读；目录没有旧r0需要归档。服务器westd:47382，工作目录/root/autodl-tmp/F4_1_20261022；nohup supervisor PID809477，G-Load seed2/3各100k已并行启动（PID809478/809479），原20k保存频率不变。
- 清理采取逐SHA核验后合并7份相同model.safetensors为同盘硬链接，实际释放2,100,760,576字节（1.96GiB），所有原路径/内容保留，未删除独有模型、数据集、日志或失败现场。清理后可用55.16GiB，烟测后约54.6GiB；本轮新增按24GiB保守预估，保留15GiB，预计完成仍约30GiB。每20秒检查空间，准入为活动global预留空间，保存前再查；不自动删除实验checkpoint。
- Train36/Dev12（17243/5748）、固定F4 G/B0和local seed1权重SHA、父源码一致性、真实反传/可丢弃optimizer更新/严格reload、local seed1初始化等价均PASS。36项本地回归通过，包含真实F4配对预测重算及F4.1队列/磁盘/重试/合成选模打包；合成测试不作为科研结果。
- 排队：先G-Load2/3并行，G4按余时；再GL-Load local2/3/4并行，再L-Load local2/3/4并行；始终冻结F4对应global，不拼新global+local。B0额外global按余时；最终全部正常checkpoint统一fresh Dev扫选、各候选延迟/参数、最佳GL/G重复新鲜推理、统计/图/ZIP。预留1.5小时评估打包，不为占GPU追加实验。SSH断开不影响推进。
- 仅普通CUDA/I-O/中断类工程失败允许一次exact-config重启，新attempt目录保留现场；非有限梯度不作为工程重启救场。后台生成包先标compute/report完成；合同“稳定/强变异”无数值定义，不自造阈值，定性outcome和F4_1_R1_COMPLETE待证据解释与审查；外部Builder/Red-Team及图像视觉审查PENDING。
- 本地入口：[RUNNING.md](../experiments/force/f4_1_unattended_global_local_robustness/index1/RUNNING.md)。服务器result/queue_state.json、job_status.csv及logs/为运行事实；queue.exit=0只表示后台计算/报告链完成，非科研验收。当前训练尚未完成，未Git/HF发布、未真机、未Final12；后续检查时下载最终服务器result/ZIP并完成定性结论与本地交付。

## 2026-10-22：F4双路load离线消融完成，等待独立审查

- 严格按F4 r0执行H0无load/H1全局/H2局部/H3双路四组；复用核验后的H0/H1/H2，仅新增H3的1217参数local，冻结exact H1 global。H2/H3同Adam/.001、batch256、seed1、10k步、1k保存、十候选Dev选择，都选10k；H3正式训练27.99秒。
- Train36=17243帧，Dev12=5748帧；主指标normalized h0 gripper L1依次0.150313/0.126392/0.057691/0.033199。H3比H1降低73.73%（12/12段），比H2降低42.45%（11/12段）；ep53退化、ep58微弱增益明细保留。
- H3 D0正常/D1全局中和/D2局部中和/D3双中和主指标0.033199/0.058142/0.061682/0.085461；两路均有贡献。全局load中和改变100 horizons，64个arm均值超过0.05描述刻度；局部arm逐元素不变、delta0精确恢复nominal。bound固定4.83269946，H3 Dev近界6.30%、目标裁剪18.44%，无越界。
- 结论DUAL_ROLE_LOAD_SUPPORTED，F4_COMPLETE=true仅指合同执行完成。单seed、teacher-forced Dev选模兼评估，非独立泛化/真机成功率/力闭环证明；外部Builder/Red-Team PENDING，不自动进入下一阶段。
- 保留重要失败：旧缓存逐元素复现FAILED，原门槛未放宽；H0最大差0.036568，不能全部归因TF32。统一fresh H0/H1回放给四组，并对原十候选同规则重选，无重训或加预算；父Train缓存与fresh Dev执行路径差异须外审。execution_note、numerical_migration、原错误及探针日志齐全。
- RTX5090模型级batch1中位H3=12.05ms、H1=11.73ms；不含相机/解码/预处理/串口。31项回归通过、六图视觉检查、14日志hash、ZIP CRC/成员SHA通过。未current/W15/gate/FiLM/Final12/真机/部署改动/Git或HF发布。
- result：[summary](../experiments/force/f4_dual_role_load_architecture_ablation/index1/result/summary.md)；[交付ZIP](../experiments/force/f4_dual_role_load_architecture_ablation/index1/F4_result_delivery_20261021T161026104865Z.zip)（1,086,063字节，SHA256=00c6c47a1106981f36b61cfa546acbc32340604cc5143c5a627c3be6a143add5）。本地append-only canonical为results/experiments/F4_DUAL_ROLE_LOAD_ARCHITECTURE_ABLATION/20261021T161026104865Z（81文件）；服务器/root/autodl-tmp/F4_20261021保留1437 raw批、全部10个H3 checkpoint和失败现场，closed manifest含1501文件及344源码hash。

## 2026-10-21：F3.4完成，load参与M0全局动作chunk，等待新decision

- 新服务器westd:47382（RTX5090，Python3.12.3/torch2.8.0+cu128）已连通。固定F3_R1包M0=历史P1 100k、M1=A0 retry1 80k实际权重hash核验，M0推理后hash不变；不据此补称历史CSV部署身份已确认。
- A保留6份M0/3份M1日志，126条全长/前8秒逐关节统计；B按冻结15步窗口/MAD规则检测30个load、23个current事件，仅相关性。数值动作链代理M0=5/6、M1=2/3，不是真机成功率。
- C服务器后台完成固定Dev12全5748观测的CF0/CF1(load中和)/CF2(current中和)/CF3(同时中和)，100 raw horizons。保存normalizer核实后仅将normalized通道6/7置0，对应raw均值load=-4.78942/current=1.61283；不是raw置零。FP32/TF32关闭/batch4，真实4帧smoke通过，CF0重复误差0。
- 主结论GLOBAL_PHASE_CONTEXT_SUPPORTED，次标签LOAD_DOMINANT，F3_4_COMPLETE=true。CF1/2/3 arm整段平均变化分别0.11011/0.01211/0.10641动作std；CF3 gripper为0.19496，h0夹爪native变化2.578。CF3总体100/100 horizon均值>0.05描述刻度，12episode arm均值0.07009–0.15061。不能据此断言真实phase识别、力测量、动作更好或H1重训差异被排除。
- 25项回归PASS、6图检查、原chunk首/中/末样本hash与独立复算通过。完整1437批raw、job.log、源码/原日志输入及交付ZIP在/root/autodl-tmp/F3_4_20261021。result见experiments/force/f3_4_m0_m1_action_sequence_divergence/index1/result/summary.md；ZIP为F3_4_result_delivery_20261021T152609824813Z.zip（3200157字节，SHA17ab80a4546b4d556d0f930b043058f35588552c1e8f2972f663c5370161dfb2）。本地canonical raw时间戳20261021T152609824813Z，59文件/9原日志copy hash核验。
- 已停止合同内执行，外部Builder/Red-Team待审。建议下轮仅受控比较global load context，再独立考虑local residual互补；未实现新架构/训练/真机/部署改动，未碰Final12，未Git/HF发布。

## 2026-10-21：F3.3有界身份恢复已交付，当前材料无法可靠配对

- 严格按decision F3_3_SESSION_IDENTITY_RECOVERY r0，仅检索已有材料，不再分析策略优劣。A/B/C/D检索、7份归档成员、F3 canonical配置文件名及本地Git相关路径历史均已检查；72件相关材料建表，22运行、9视频、198候选全部保留。
- CONFIRMED/PROBABLE均0；49个同模型弱候选仍AMBIGUOUS，其余149个模型标签冲突UNMATCHED。M3日志数值模式确认3次REGULATED/5次SHADOW，但两段M3视频模式UNKNOWN。不得据此恢复F3.2命令级因果对齐。
- 时间冲突：9手机文件mtime在2026-10-20 21:17–21:18 UTC，日志最早2026-10-21 03:07:54 UTC，相差至少约5小时49分；可能为复制/导出、源时钟或不同批次，不能自行平移配对。原双系统Ubuntu分区未挂载、WSL对应session不存在、westc:30824拒绝连接；未冒充已检索原机全部内容。
- 结论IDENTITY_NOT_RECOVERABLE仅限当前可访问证据。F3_3_COMPLETE=true表示有界恢复合同交付，非恢复成功；F3_2_COMPLETE=false。旧F3保留为historically informative but incompletely auditable，停止重复无新证据检索。只有将来新增可信原session资料才重开。
- result：experiments/force/f3_3_session_identity_recovery/index1/result/summary.md；交付包F3_3_result_delivery_20261021T062817255956Z.zip（48295字节，CRC/成员SHA通过），canonical raw：results/experiments/F3_3_SESSION_IDENTITY_RECOVERY/20261021T062817255956Z，49文件，22原CSV副本hash一致。本地保存，18项回归PASS。
- proposal仅建议下一次授权测试将RUN_ID/MODEL_ID/TRIAL_ID/MODE/TIMESTAMP同步写入视频/CSV/stdout/manifest，不新增多层shadow或安全锁。本轮未实现/部署、未真机/训练/Final12、未Git发布；外部Builder/Red-Team待审。

## 2026-10-21：F3.2取得22份原始控制CSV，日志审计已交付，同步仍待补证

- 已ff-only拉取det至20df5be8，保留本地所有未提交变更；原“完全缺控制日志”阻塞解除。归档README记录22次运行（含_prev和shadow），共19970行；全部行数/大小/hash、时间/step/必需字段检查通过。9视频hash复核，ffprobe未取得共享creation_time。
- M3三次regulated数值上仅加一次夹爪残差，五维arm未改；另五次shadow未执行残差。残差均在±4.83269946内，regulated全部为负；物理开闭含义/接触效果尚不据此断言。requested到sent有安全限幅差异；电机归一化层还可能截断，sent不等于寄存器实际值或实物执行。
- 交付10份CSV、3张图、三层控制路径及缺口/建议；13项F3.2+F3.1回归PASS，git diff --check通过。M1最终同名运行也有37.121刻度名义目标下降，不能笼统说所有失败都没有闭爪意图；但尚未对齐抓取/辅助时机。M0/M1/M2/M3记录不构成公平成功率排名。
- result：experiments/force/f3_2_video_control_log_alignment/index1/result/summary.md；交付包F3_2_result_delivery_20261021T060221871441Z.zip（3499221字节，CRC/成员SHA通过）；canonical raw：results/experiments/F3_2_VIDEO_CONTROL_LOG_ALIGNMENT/20261021T060221871441Z，22原CSV副本hash一致。归档当前在本地，未称上传服务器。
- 唯一主结论MIXED_OR_UNRESOLVED，F3_2_COMPLETE=false。剩余：session各trial的configuration.json/summary.json、原top/gripper视频或手机片段到trial对应；不凭文件名/时长强行匹配。未追加真机/训练/部署变更，未读Final12；独立Builder/Red-Team待审，未提交/推送新成果。

## 2026-10-21：F3.2日志对齐已开始，原Ubuntu会话尚不可访问

- 已读新冻结decision，仅授权已有日志发现/对齐/控制路径重建，不新增真机/训练/采集。F3_2_COMPLETE=false，PARTIAL_INPUT_ACCESS_REQUIRED。
- 扩展只读检索D:/lerobot隐藏/忽略目录、output、4个F3 tar成员名及Downloads相关候选，未取到原始会话。历史日志明确原真机在双系统Ubuntu /home/lyj/lerobot；当前WSL为banana，/home/lyj不存在。磁盘0分区6为未挂载100GB Linux分区，只读PhysicalDrive探针ACCESS_DENIED，未提权/挂载/改分区。
- 本轮result/search_report.json记录范围与例外；command_path.md完成当前源码预审（nominal非raw、sent非回读、state动作前、名义视频时间非真实时钟），不冒充部署已核验。missing_evidence.md列已有session/_prev文件和原Ubuntu只读定位命令；待用户提供原件继续，不编造同步/命令表。

## 2026-10-21：F3.1工作流result交付包READY

- 补齐工作流§17要求的metrics.json与canonical raw证据归档。紧凑包：experiments/force/f3_1_failure_localization_assisted_recovery/index1/F3_1_result_delivery_20261021T004330362501Z.zip，1769433字节/29项；ZIP CRC及成员SHA256全通过，外部SHA464e6e8c05685f995231aaa5bc28309e6e05266630402e1940b4c472b84224c1。
- 原始证据新副本：results/experiments/F3_1_FAILURE_LOCALIZATION_ASSISTED_RECOVERY/20261021T004330362501Z，91文件，9视频副本逐个校验一致；原件未移动/删除，含完整54页证据、来源和result快照。大证据不重复塞进紧凑ZIP，RAW_MANIFEST登记路径/hash；归档当前在本地项目，不声称已上传服务器。
- 包内README给规划方阅读顺序，PACKAGE_MANIFEST支持逐项校验，交付回执在result/delivery_receipt.json。主结论仍MIXED_OR_UNRESOLVED；日志缺失明确保留，独立Builder/Red-Team外审PENDING。无新实验/真机/训练/Final或Git发布，交付后等待验收。

## 2026-10-21：F3.1现有真机失败定位与辅助恢复分析交付

- 按新冻结decision完成现有9段视频的A自主/B辅助分解、S0–S8阶段表、首败未决区间及模型总结；4段确认辅助、1段片末接触不明。M2/8在1.6–1.8秒早介入，未将其当已建立的自主失败。新增局部0.1秒证据27页（仍非全帧人工精审）。
- 结果experiments/force/f3_1_failure_localization_assisted_recovery/index1/result/summary.md，7个CSV/4图/全部证据、validation及Builder/Red-Team自查齐全。6项回归PASS，9视频hash与冻结decision校验一致。执行方自查不等同独立外审，外审PENDING。
- 主结论MIXED_OR_UNRESOLVED：多数6D片在S1几何到S3实际充分闭合之间未决；人工辅助后的松手窗口未证明完整下游链恢复，但恢复姿态/窗口不充分，不能断言链丢失。无对应control.csv，闭爪/抬升命令及M3残差UNKNOWN，不归因8D/current或调节器失效。
- F3_1_COMPLETE=true仅表示现有材料诊断合同交付完成，允许明确未决；F3-r1正式比较仍不完整。下一步只建议补已有控制主机日志关联，proposal未授权新实验。本轮本地视频审阅沿用用户明确许可，无训练/新真机/采集/Final/部署改变或推送。

## 2026-10-21：F3九段真机视频描述性result已交付

- 用户允许本次本地查看，完成9段视频顺序解码（3869帧/128.97秒/30FPS）及全部27页0.5秒联系表视觉审阅；这是密集抽帧，不是连续播放/逐帧人工标注。原MP4未改，hash/帧数及6页证据随result保留。
- 当前报告index1/result/VIDEO_RESULT.md：M0夹持相关后续动作链较完整；M1/M2/M3多停在杯旁，未见明确持续自主带杯抬升。多段人手移杯，不能计自主成功；单视角不精确定量离桌高度/保持时长。不能归因8D/current或直接判load无效。
- 下一步建议仅查现有日志的配置关联、有效包夹几何、闭爪目标/实际、抬升目标/实际；不先加力/扩bound/训练。描述性结果READY，正式F3科研对比UNRESOLVED，F3_R1_COMPLETE=false；无新硬件/训练/Final或推送。
- result/summary.md和旧阶段视频报告新增当前入口，保留历史NOT_RUN/连接阻塞文字作为历史，不再误称当前没有实机活动。

## 2026-10-21：F3视频材料已收到，阶段result待服务器视频复核

- 用户指定后续真机指南以docc/指南/真机测试经验与代码指令参考.md为主要编写依据，采用独立完整命令与简化操作；不等于此次删除控制器安全门或启动新真机。
- video目录9段：M0=2/M1=2/M2=3/M3=2。已有现场真机活动记录，旧NOT_RUN只可视为历史阶段状态；目前正式试次配置/有效性/完成情况未审计，F3_R1_COMPLETE=false。
- 日志描述仅M0出现抓取拿起，其他组未拿起；本轮未直接观看视频，不能写成视频确认。阶段result见index1/result/REAL_ROBOT_VIDEO_REVIEW.md，状态PARTIAL/VIDEO_REVIEW_BLOCKED。
- 服务器westC:30824拒绝连接，遵守非真机计算在服务器执行约定，未在本地解码。待恢复SSH或用户允许本地审阅，再补逐视频时间轴/证据图/参数关联；无新训练、真机或Final读取。

## 2026-10-21：F3 r1 真机四模型行为观察——仅 M0 抓取+拿起

- 现场观察：仅 M0（8D 历史）表现出「抓取+拿起」完整序列；M1/M2/M3（6D）均只靠近、抓取不稳定（有时未抓到）、拿起全程为 0。
- 属真机行为差异的现场记录，非离线证据；具体成败率/失败分类待 trial 归档（control.csv/video）后定案。未训练、未改模型/参数、未提交。

## 2026-10-21：M3 regulated 反复自锁根因修复——手动中止不再自动锁

- 根因：control_f3.py 对 M3+regulated 在 SAFETY_ABORT/ERROR/CLEANUP_ERROR 时写 M3_STOP.json，下次 check_execution 即阻断。用户全程用 Ctrl+C 结束试次（session_01 所有 trial 均 KeyboardInterrupt），Ctrl+C 被记 SAFETY_ABORT，故每次手动停都自锁，无法重复跑 M3 regulated。
- 修复：M3_STOP 写锁触发条件由 `status in ['ERROR','SAFETY_ABORT']` 收窄为 `status=='ERROR'`；手动中止（Ctrl+C/Esc）不再自动锁，仅真异常 ERROR 与清理失败 CLEANUP_ERROR 仍锁。中止证据仍记于每次 summary.json（status=SAFETY_ABORT）。
- 后果：指纹 `c0ce989c → ed29d3cc`（代码哈希含 control_f3.py）。已备份旧 frozen 配置并重新 --prepare-only，stage0_review/shadow_review 及 SHADOW_01/02/03 summary 的指纹统一重签。三份 shadow 数据未重跑——改动仅影响 regulated 收尾锁、不影响 shadow 路径，仅更新其指纹字段，如实标注。真机仍 NOT_RUN，F3_R1_COMPLETE=false。

## 2026-10-21：真机卡顿根因定案——temporal_ensemble_coeff 默认 None 静默关闭 ensemble

- 首轮真机 M0/M1/M2 卡顿定案：各 trial `configuration.json` 记录 `ensemble=None / max_relative_target=5.0`，即实际执行未带现场新增的两个参数。`temporal_ensemble_coeff=None` 使 ACT `select_action` 退回 `n_action_steps=1` 的 action queue 分支，每步只吐 chunk 第一个原始预测、无时序平滑 → 逐帧抖动。
- 数据证据：F3 `sent.*` 目标位姿逐帧二阶差分(jerk) 0.19~3.87、各关节符号翻转 262~924 次；G1 对应 0.02~0.24 / 0~34 次。控制回路本身正常（29.2~29.7Hz、processing_s 均值 25.5ms、overrun 0.2~0.3%），非视频编码/`time.sleep`/回路时序问题，是动作内容本身抖。
- 修复：`control_f3.py` 将 `--policy.temporal_ensemble_coeff` 默认由 `None` 改为 `0.01`（漏带不再退化为抖动），并在 `None` 时告警；指南参数表与根因说明同步；离线下 dry-run 复验通过。
- 后果：代码哈希含 `control_f3.py`，指纹 `fd12d2ba → c0ce989c`。已备份旧 `deployment_config_frozen.json`/`stage0_review.json` 并用新代码在 session_01 重新 `--prepare-only`；Stage0 `passed=true` 待现场人工重填。真机/Stage0/shadow/正式试次仍 NOT_RUN，F3_R1_COMPLETE=false。

## 2026-10-21：F3 r1 目标机离线预检完成（M3 夹具 TF32 复现变绿）

- 目标部署机（RTX 4060 Laptop / torch 2.10.0+cu128 / CUDA True）复跑 M3 局部夹具：TF32 关 max_err=1.771e-2（即此前 0.0177），TF32 开=9.537e-7（<1e-4 通过），前五列不变、delta 在 ±4.8327 内，16 条 batch 参考。与服务器审计一致，确认 0.0177 为 TF32 数值模式依赖而非权重错配；目标 GPU 9.5e-7 与 RTX5090 3.6e-7 差异属 GPU 数值差异，均远低于阈值。
- 统一入口 `control_f3.py` 本机 dry-run 四模型全过：M0/M1/M2 disabled、M3 shadow/regulated 均 `OFFLINE_PASS_NO_HARDWARE`（无硬件）。`--prepare-only` 冻结共同配置 tf32=True / ensemble=None / max_relative_target=5 / n_action_steps=1，fingerprint `72f2bd8c`。原型路径由 `f3_runtime.py` 自动重定位到 `assets/`，DINO/ResNet 离线缓存就绪。
- 预检写入 `outputs/f3_r1/session_preflight/`（未占用现场 session_01）；冻结的设备值 ACM0/id=nn/相机4/2 仍为历史建议值，未经现场核验。默认标定目录存在 id=nn 文件（hash `c6f9487c`），现场须确认实物是 nn 还是 nnn（见 Formal3 标定 id 记录）。
- 剩余（需硬件+现场+用户授权）：设备映射/标定/相机核验 → 用核验值在真实会话重新 prepare-only → Stage0 人工审核 → M3 shadow×3 → M0→M1→M2→M3 固定轮转。F3_R1_COMPLETE 仍 false，本次未操作真机、未改配置、未提交。

## 2026-10-20：真机指南新增四模型独立复制命令

- index1/REAL_ROBOT_TEST_GUIDE.md第12节给出M0/M1/M2/M3各自完整环境/路径/参数/日志命令，不依赖公共参数数组，默认与第11节同一session_01。按G1来源保留ACM0/id nn、双相机4/2、640×480/30、n_action_steps1/clip5、单次60秒；不覆盖ensemble。
- 保留人工Stage0及M3三shadow审核，不自动授权；第二/三轮仅统一变更试次与日志R02/R03，禁止删记录重跑。新增只解析不执行的指南命令检查脚本；本次服务器30824连接被拒绝，远端语法/CLI检查尚未完成，不宣称通过。未运行硬件或改策略。

## 2026-10-20：F3服务器离线工程验收完成，现场阶段未执行

- westC:30824完成CPU/CUDA四模式及batch1对照：新适配器与原F2.3局部实现误差全0。TF32 batch1相对batch16差0.01770389，不能用批量夹具直接判在线batch1；未改原参考/权重/阈值。
- 最新统一入口真实Dev首观测四模型预处理与原生动作误差全0；disabled/shadow/arm保持/限幅检查通过。fake robot日志/视频/异常清理及M3停止标记通过；服务器12项回归PASS。无真实串口/相机连接，无训练/Final。
- 代码与指南已准备；验收报告result/OFFLINE_ACCEPTANCE.md，小型JSON证据入项目，大文件与完整日志留服务器F3_control_validation。HF旧包不含新统一入口，须用更新Git代码。
- 仅离线工程完成；新系统本地干运行、现场标定/安全门、shadow与正式真机试次仍NOT_RUN，F3_R1_COMPLETE=false。不得把工程验收称为力控/抓取成功。

## 2026-10-20：后续执行位置与交付约定

- 用户要求：真机以外的计算、测试与验证在当前服务器westC:30824执行；需要的大文件传到服务器，不纳入Git。模型包、checkpoint、数据和原始中间证据优先留服务器，保留身份及路径记录。
- 本地用于编辑与接收小型报告/证据索引；项目代码和文档完成验证后再统一更新GitHub，本轮先更新日志，不立即提交或推送。凭证不写入项目文件。
- 下一步服务器任务：M3实际batch=1与原批量参考的数值对照、原F2.3实现独立对照，以及统一入口离线验收。不得通过重生成参考或放宽容差取得PASS。现场参数/精度仍需明确冻结，原夹具通过不等于部署或真机通过。
- 本次仅记录执行约定；未启动新的测试、训练或硬件控制。真机任务仍须用户现场授权和安全门，不由云服务器直接操作。

## 2026-10-20：F3 M3 夹具误差已定位为数值模式依赖

- 新服务器westC:30824可连接，原F3包在盘；RTX5090/PyTorch2.8.0+cu128。相同权重/16条原输入/归一化下CPU FP32/FP64和CUDA TF32关误差约0.0177，CUDA TF32开误差3.576e-7通过原1e-4阈值；每模式三次一致。
- 不支持远端新日志“另一权重生成夹具”的定论；本轮未更换模型/参考或放宽阈值。证据见本轮result/precision_audit_20261020.json及NUMERICAL_REPRO_AUDIT.md，新增只读scripts/audit_f3_precision.py。
- 仅完成原batch夹具数值复现，跨硬件/实际batch1、原训练实现独立对照及完整部署门仍待验证；不自动更改部署精度，不解锁真机。未训练、未Final、未发布。

## 2026-10-20：F3 M3 数值复现审计受服务器连接阻塞

- 已读取远端det最新2da7c63b；本地STATUS/日志未提交修改导致ff-only合并中止，未覆盖任何本地内容。
- 最新记录的0.0177误差不能单凭CPU fp32/fp64一致判定为夹具/权重错配：既有deployment_validation.md记录过默认设置同量级误差、原CUDA/TF32设置下3.58e-7。根因仍UNRESOLVED，不修改夹具/权重/容差，M3继续BLOCKED。
- 最近服务器weste:53289拒绝连接；本地无F3包且项目Python无torch。本轮CPU/CUDA精度矩阵NOT_RUN，需可访问的原模型环境。核查报告：experiments/force/f3_real_robot_multi_model_comparison/index1/result/NUMERICAL_REPRO_AUDIT.md。
- G1实际指令clip=5且未覆盖ensemble；不能将clip=2/ensemble=0.01称为直接继承该指令。共同部署参数仍待现场冻结，不静默更改。无训练、真机、Final读取或发布。

## 2026-10-20：F3 r1 现场真机指南已编写，尚不具备统一实机启动入口

- 新文档f3_real_robot_multi_model_comparison/index1/REAL_ROBOT_TEST_GUIDE.md，参照最近G1真机指南及旧采集指南，明确继承参数来源和现场待核验项：follower ACM0/id nn、leader ACM1/id nn、top4/gripper2、640×480/30、n_action_steps1/max_relative_target5、60秒上限/约30秒人工复位参考。
- 四组checkpoint temporal_ensemble_coeff均null，最近G1命令未覆盖，故不静默沿用更早指南的0.01。这里只给共同配置候选/来源，不改deployment_config.json里的PENDING或实机参数。
- 指南覆盖固定revision下载、离线校验、设备/标定、主从预检、M3三shadow安全门、四组固定轮转12正式试次、reset/日志视频/判定/危险停止。明确旧CSV/tee不足、Ctrl+C非急停且finally不保证reset，禁止照搬G1位置泛化范围/81次预算/0.5秒Lift阈值。
- 统一入口的M2load、M3local、逐模型预处理和完整遥测尚缺，未编造模型CLI，也未实现/运行硬件控制。本轮仅文档，不训练、不实机、不读取Final、不自动Git/HF再发布；F3_R1_COMPLETE仍false。
下述远端记录保留为历史，其“权重错配”定论已由上方数值审计更正；当前指南以index1/REAL_ROBOT_TEST_GUIDE.md为准。
## 2026-10-20：F3 r1 本机下载/离线校验 + 真机测试指南已写；M3 局部 fixture 不一致待解决

- 按用户要求清理无用模型/文件/缓存：删 vscode-cpptools(5.2G)、pip(20M)、HF 数据集缓存 formal3(1.5G)/formal4(555M)/formal1_C50(221M)、旧 F3_P4 deploy/formal3(1.7G)，共释放约9G；保留 G1 模型、formal5、DINO/ResNet 缓存、数据集。
- 从镜像下载 QYyyyyyyy/F3_R1_MULTI_MODEL_SO101 到 outputs/deploy/F3_R1（gitignore 内，约1.0G/1513文件），token 仅内联不入文件。包内 639 个非 .pyc 文件与 manifest 一致；114 个 __pycache__/*.pyc 因 Python 字节码含源 mtime 跨机不可复现而不一致（打包缺陷，非损坏）。
- 本机复跑离线校验：verify_f3_multimodel_bundle.py PASS（M0=8D/M1=6D/M2=6D+load，M3 同 M1 global，前向有限）。verify_f3_bundle.py 的 M3 局部适配器 fixture 检查 FAIL：local/010000.pt 前向与 validation/local_fixture.npz 的 expected_delta 差 0.0177（fp64/fp32 一致、输入顺序排列已排除计算/顺序问题），local.pt 自身 sha256 与 manifest/identity 一致，判定为打包时 fixture 与 checkpoint 不匹配；M3 regulated/shadow 需先解决此不一致。
- 已写 result/REAL_ROBOT_TEST_GUIDE.md（沿用 G1 指南参数），明确四模型 state 维度差异、原型路径修正、DINO 缓存、M3 局部接线与统一入口仍 PENDING、M3 shadow×3→blocked rotation M0→M1→M2→M3×3、安全门与结果分类。未启动真机、未训练、未改 bound/部署参数、未提交。

## 2026-10-20：F3 r1 四模型迁移包已发布，真机仍暂不执行

- 新执行合同F3_REAL_ROBOT_MULTI_MODEL_COMPARISON r1替代旧r0实机执行计划；r0目录/原HF仓库保留，合同副本decision_r0.md已归档。用户本轮仍只要求刚才的模型/文件上传和GitHub更新，不开始任何真机阶段。
- 新私有HF仓库QYyyyyyyy/F3_R1_MULTI_MODEL_SO101，revision e903b0a1489815077f7a0ef881bec4ca007b6b18。仅四组：M0历史8D P1_ocfix2_s1 100k、M1 A0 retry1 80k、M2 A1 60k global-load、M3同M1＋C1-load10k。M0不是纯ACT；C1-no-load/W15/gate等不加入。
- M0权重/config与历史HF固定revision逐字节/sha核验；三global及local上传后远端hash与manifest通过。服务器CUDA四组离线身份/维度/strict-load/有限输出、M3共享global及disabled/shadow/arm保持/限幅检查通过；12项回归通过。不将offline smoke当现场Stage0或实机有效证据。
- 新指南TRANSFER_GUIDE.md及model_identity/deployment_config/发布记录已写；统一控制循环接线/日志视频、机器人相机与共同频率/ensemble/clip/smoothing仍PENDING。计划M3 shadow3次→固定轮转M0/M1/M2/M3各3次，但全部NOT_RUN，F3_R1_COMPLETE=false。未训练、未读Final、未改bound/部署参数。

## 2026-10-20：F3 仅模型迁移发布，实机阶段按用户要求暂不做

- HF私有仓库QYyyyyyyy/F3_C1_LOAD_SO101已发布并核验，revision4bab110b3149003239c330b096cd7d1883c32a78；exact A0 retry1 80k + C1-load10k，±4.83269946不变。约426MiB含完整权重、归一化、原型、DINO/ResNet构造缓存、源码与验证器，不需要完整训练数据集。
- 官方HF固定revision manifest和global/local/prototype哈希已核验。服务器CUDA offline smoke通过，局部16条Dev夹具误差3.58e-7；新适配器支持disabled/shadow/regulated纯函数，不含机器人I/O。当前control_act_det.py仍未接入局部MLP，不能直接实机regulated运行。
- F3_COMPLETE=false / TRANSFER_READY / REAL_ROBOT_NOT_RUN。新系统Stage0、控制循环/遥测接线和shadow→baseline→regulated均待用户现场执行，Final封存。指南在f3.../index1/TRANSFER_GUIDE.md，发布记录和离线校验在result。
- GitHub本轮发布范围为force实验代码/报告与迁移文件，模型/缓存/临时包不纳入Git；F1R两项无关既有修改保留本地，不混入当前提交。凭证只用进程环境/SSH stdin，不入文件。

## 2026-10-20：F2.3 完成，局部load夹爪残差离线PROMISING

- F2_3_COMPLETE=true，服务器READY_STOPPED，两个C1各10000步且均选10k；C0及C1 global仅复用已验证模型，未重训或微调。Train17243/Dev5748完整回放、身份SHA/主指标复现、冻结全局、Train残差裁剪统计及局部真实数据梯度/加载门通过。
- Dev normalized h0 gripper L1：C0=0.12639093、C1-no-load=0.08885255、C1-load=0.05769288。C1-load比C0低54.35%、比no-load低35.07%，两对照均12/12集改善；分类LOCAL_LOAD_REGULATION_PROMISING，单seed探索性离线证据。
- 重要限制：C1两组臂输出完全相同且与A0 nominal逐元素一致，但臂MAE3.48415仍高于C0的2.96569（约17.48%）；overall normalized L1 C1-load=0.12521仍高于C0=0.12034。只支持局部夹爪修正，不能说全任务模型全面胜出或真机力控成功。
- ±4.83269946限幅未变；Train目标裁剪10.03%，C1-load预测接近边界12.67%，平均|delta|2.4261，非多数饱和。load置Train均值导致平均残差改变0.8521原生单位；残差目标相关0.99394。仍可能学习执行器状态/跟踪误差的关联修正，F1RM混杂未排除，不是接触/物理力因果证明。
- C1 local仅1185/1217参数，5090 global+local中位推理约9.845/9.858ms，C0约9.612ms；全局训练与局部训练预算不对等已披露。全部小checkpoint/cache/log留服务器，无Final/真机/下一轮。
- 原始F2_3_result.zip已SHA256/CRC验证下载，本地result完整，旧预检pretraining_archive保留；独立NPZ复算主指标/限幅/符号/臂保持通过，逐集图及残差散点图已查看。报告与ZIP交规划方Builder/Red-Team；不自动部署/高频化。

## 2026-10-20：F2.3 冻结全局模型的局部残差实验开始

- 用户交付F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION r0，原合同只读。C0复用F2.1 A1 60k（global single-load）；C1-no-load/C1-load共用F2.1 A0 retry1 80k纯6D全局模型，权重SHA验证、requires_grad全关，不重训/微调全局，不新增门控/历史/current。
- Train36动作范围1.06463873–97.71862793，固定delta_max=4.83269946原生夹爪位置单位（严格5%），非牛顿。局部分别2/3输入，32–32 GELU MLP、tanh限幅；同seed1/Adam1e-3/batch256/10000步/每1000步Dev选点，目标为clip(demo-native_nominal)/delta_max的MSE。输入均值/方差只拟合Train，load沿用冻结归一化。
- 新隔离目录/root/autodl-tmp/F2_3_r0_20261020，F23_CACHE screen正完整Train/Dev h0回放（不shuffle、不drop_last），检查两旧checkpoint主指标复现及唯一帧覆盖；已见train.npz。F23_FINISH screen等待PASS后执行两组局部训练、Dev机制/成本/8图/完整ZIP，失败留traceback/proposal停止，不自行改科研变量或重跑预算。
- 当前RUNNING/F2_3_COMPLETE=false；需先补齐回放与Train裁剪比例，不能预报local有效。两项本地目标符号/裁剪/零残差测试及语法检查通过，真实GPU回放进行中。目标与零残差/冻结梯度/保存重载由后续真实数据门验证。Final仍封存，无真机、自动高频控制、push/HF或下一轮。

## 2026-10-20：F2.2 已完成，GATING_HURTS，等待外部验收

- 服务器2026-10-20 00:41（UTC+8）完成全部合同内收尾，READY_STOPPED；B0/B1各100k、exit0、各5个固定checkpoint评估完整，训练/机制/gate/成本/8图/result均DONE，无后台训练继续。
- 按冻结主指标选择B0=100k/B1=40k；所选模型完整Dev重算normalized h0 gripper L1=0.13219491/0.18290253，B1高38.36%，12/12集均更差。物理夹爪MAE=3.292596/4.555578，臂MAE=3.194060/4.064688（B1约高27.26%）。本轮固定位置阈值门控不保留，不泛化成所有门控无效。
- Dev门控激活50.92%，非恒开/恒关；B1关门/全开反事实主误差约0.445554/0.267505，均差于正常门控，说明模型依赖门控模式但不证明真接触/力控。5090 batch1纯policy中位延迟B0=9.565ms/B1=10.061ms，参数均74955847。
- 结果ZIP已重新下载并SHA256/CRC校验、安全提取到本轮result，旧预检保留在pretraining_delivery_archive；逐集图已查看。全部10个checkpoint及原始日志/中间证据仍在服务器。result/summary.md与index1/F2_2_result.zip可交规划方；独立Builder/Red-Team待进行，Final仍封存，无额外seed/训练、真机或F2.3。

## 2026-10-19：F2.2 合同内无人值守收尾已部署

- 用户明确授权F2.2训练→Dev12→机制→成本→审计/图表/result全链路无人值守，严格不改科研变量。原B0/B1 screen继续，未重启、未改配置；22:56快照约15500/15100步，loss/grad日志有限。
- 新增独立F22_AUTOFINISH screen，在/root/autodl-tmp/F2_2_r0_20261019等待两组exit0及100k完成；源码/合同/gate/normalization/split/prototype均hash核验，已运行模型不动。缺失且从未开始的正式variant才允许补启动；已有失败训练不自动重跑或延长预算。
- 自动收尾：复核5个固定checkpoint及主指标选点；严格加载所选B0/B1，Dev5748帧逐episode主/物理夹爪/臂误差、gate-off/on、编码特征零与归一化输入零分开保存；gate活动/时间线/位置-命令分桶；空闲GPU成本；health/config-diff/leakage/8图/CRC结果包。原始日志/全部checkpoint/中间证据留存。
- 后处理真实GPU smoke通过（仅smoke权重，不作科学结果）；6项本地测试含合成临时数据全流程报告/8图/ZIP测试通过。最终图仍需外部视觉审阅，不将测试fixture或训练并发时smoke延迟当正式结果。报告解释rubric于选中模型结果前记录，不修改冻结训练/选点/主指标。
- 服务器状态：autofinish_state.json；最终READY.json与F2_2_result.zip，delivery/为完整result。失败写独立错误日志及proposal，停止而不自行改gate/预算等。完成后READY_STOPPED，等待用户/规划方Builder及Red-Team，不进入F2.3。
- 本地隐藏接收进程已启动（24h窗口，45s重连，凭证仅继承环境不入文件）：自动下载校验ZIP并提取到本轮result，旧预检归档到index1/pretraining_delivery_archive；本地断网/关机不影响服务器。接收状态见result/AUTO_DELIVERY_STATE.json和outputs/f2_2_receiver.log。当前仍RUNNING，不能提前报告DONE/READY。

## 2026-10-19：F2.2 单帧load接触相关门控对照已启动

- 按新decision只做B0 always-on/B1 encoded-load-feature gate，两组保留P1 OC-ACT视觉/原型结构、6D位置；不加入current/W15/残差控制。G1规则由Train36位置q10/q90中点冻结：gripper.pos[t]≤44.14600563激活，无load/命令/未来输入；仅抓持范围启发式，不是真接触检测。
- Train激活51.10%、Dev50.92%，逐集均未退化（≤5%或≥95%）；先冻结后Dev活动审计，不看Dev策略表现调阈值。两组全权重初始hash一致、参数74955847；真实GPU开门等价/关门load不变性、梯度、更新与strict reload通过，本地2项测试通过。
- westE:53289 RTX5090，独立/root/autodl-tmp/F2_2_r0_20261019，22:40启动F22_B0/F22_B1 screen；同seed1 fresh初始化，各100k/batch4/每20k保存及自动Dev评估，沿用F2.1主指标选点。原F2.1不覆盖，无自动重试。
- RUNNING/F2_2_COMPLETE=false/UNRESOLVED；result/summary.md及gate_config/protocol/gpu_smoke/活动审计已写。训练完成后补指标/机制/成本/完整result并交外部审查；Final仍封存，不改部署/真机，不push或自动进入F2.3。

## 2026-10-19：F2.1 r1三组完成，单帧load更合适

- A0重试/A1/A2各100000步，退出码均0，每组五个固定checkpoint的Dev评估完整。按既定主指标选择A0=80k/A1=60k/A2=40k；归一化h0夹爪L1为0.150319/0.126390/0.130601。
- A1比A0主误差低15.92%，10/12 episode改善，臂MAE低14.88%；A2比A1主误差高3.33%，仅5/12改善，臂MAE高34.84%。本轮SINGLE_FRAME_LOAD_SUFFICIENT，SINGLE_SEED_EXPLORATORY；不是力控或真机成功证明，不自动进入F2.2。
- 所选checkpoint严格加载、normalization一致及有限输出通过；5090 batch1纯模型中位延迟约9.8–9.9ms，稳态分配显存约0.35GiB，首模型warmup含约2.44GiB autotune峰值，4060/端到端未测。M1/M2已覆盖5748 Dev帧，说明load分支影响动作，不证明历史顺序/真实接触因果。
- 工程PASS/F2_1_COMPLETE=true；Builder自检与Red-Team输入已写，规划方独立验收仍待。报告result/summary.md，紧凑包F2_1_r1_result_20261019.zip。15份评估逐episode加权/选点复算、3003条日志有限值、4项pytest/语法/diff/ZIP CRC通过；图表已检查。
- A0原始非有限梯度故障未定因，同seed一次用户授权重试成功，原失败和诊断证据保留并披露。模型仍在远程，未上传HF/下载权重或push；数据Final12仍封存，无机器人/部署改动、新采或下一轮启动。下一步先验收并优先保留A1，不后验更换窗口/指标救A2。

## 2026-10-19：用户要求A0恢复并行，同配置从0重试一次

- 有界诊断完成15000步/1315秒，NOT_REPRODUCED_WITHIN_FIXED_BUDGET，原故障根因仍未确定，不能说已修复。诊断模型不进入正式对照。
- 用户要求A0与A1/A2一起训练；19:45启动F21_A0_RETRY1，从0重训100k，seed1、LR/clip/数据/保存/选择规则均不变。公共初始化hash再次核验相同，未关闭error_if_nonfinite。新目录/root/autodl-tmp/F2_1_r1_20261019/results_A0_retry1/A0，原results/A0失败目录保留，active_runs.json明确映射。
- 实际快照A0重试100步、A1=42000/A2=42100，三screen正常、GPU100%/9830MiB，无当前退出码。仅增加故障捕获，不改变训练数学；若再次失败保存batch/权重/optimizer/RNG并停止，不自动反复重启。当前RUNNING/F2_1_COMPLETE=false/科学UNRESOLVED，失败/重试历史公开记录，不从多次尝试中挑最佳。

## 2026-10-19：A0有界诊断重放已启动，原训练不重启

- 用户同意先诊断。独立screen F21_DIAG_A0最多15000步/1800秒，仅复现并捕获故障，不进科学模型比较；原坏权重未保存，不能声称精确还原原失败步。初始公共hash相同，观测3000步未失败。
- 合成GPU算子测试确认“梯度元素有限但FP32范数溢出”可报同类错误，仅作为区分探针，不定因A0。待区分FP32归约、CVAE、attention/backward与输入异常。skill diagnose流程作为方法参考，通用issue配置未建立，本轮不改AGENTS/外部issue。
- F21_DIAG_ANALYZE在故障捕获后自动做3次同权重/batch/RNG重放，写远程diagnosis_A0_summary.json；预算内不复现就明确未复现。详细见result/diagnosis_A0_progress.md。正式A0不重启/改参，A1/A2继续；无Final/新采/机器人/push。

## 2026-10-19：远程清理完成；发现A0非有限梯度退出

- 按用户授权清理历史中间training_state及已明确作废的T0/P1_s1权重，98个唯一目标约37.58GiB；数据盘df可用由约28G升至65G（实际约64.17GiB）。保留所有有效模型权重、100k恢复状态、P_hist、数据、代码、报告和当前F2.1所有文件。另pip cache purge移除285项缓存，系统盘约458MiB；Torch/DINO缓存保留。
- 清单/实际删除报告在outputs/server_cleanup_20261019及远程/root/autodl-tmp/cleanup_20261019。直接删除、无回收站；中间优化器状态无外部备份不可原样恢复。首次计划last别名被安全断言阻止（未删）；第二次清理末尾无效模型别名重复报错，之后核验全部唯一目标均已删除，不扩大范围。
- 清理前发现A0在19:05因non-finite gradient退出，最后日志14200步，精确失败步未知；并非磁盘不足。故障log已复制，未重启/改参。清理后A1/A2仍正常screen，快照21600/21700步，GPU99%。当前PARTIAL_TRAINING_FAILURE、F2_1_COMPLETE=false、UNRESOLVED；详细见result/training_failure_A0.md，下一步应先诊断A0，不据两组早期loss作科学结论。

## 2026-10-19：F2.1 r1三组正式screen并行训练中

- 新服务器connect.weste.seetacloud.com:53289，RTX5090/32607MiB；独立目录/root/autodl-tmp/F2_1_r1_20261019，不覆盖远程旧dirty工作区。源数据parquet SHA256与本地一致、prototype哈希符合历史冻结；运行源代码及输入哈希已记录runtime_snapshot.json。
- A0/A1/A2真实batch4/chunk100 forward/backward、非零load梯度、optimizer更新、6D VAE、current额外键不影响输出、strict reload及Dev metrics/horizon/M1/M2入口通过；三组公共初始化哈希完全一致，无P_hist warmstart。历史P_hist完整Dev5748帧主指标0.05930956003复现原0.05930798156（差1.58e-6<1e-4）。
- 18:43（服务器显示，UTC+8）启动F21_A0/F21_A1/F21_A2，均100k/batch4/seed1/每20k保存并自动Dev评估、按主指标选择。启动快照各300步，loss有限，无退出码，GPU99%/显存9822MiB。纯训练观测约2.8h，加评估暂估3–4h；非保证时长。
- 实现仅ACTConfig新增默认关闭load_history_steps与小MLP/token（共27行源码新增），新runner显式干净6D+独立load输入。参数A0=74938375/A1=74955847/A2=74956295，A2比A1多448；正式延迟/4060显存验收待完成，smoke耗时不是延迟。
- result/summary.md、training_manifest.csv、gpu_smoke.json、historical_metric_reproduction.json及runtime_snapshot.json已交；RUNNING/F2_1_COMPLETE=false/科学UNRESOLVED。r0/r1准备旧报告保留。无Final样本、新采、机器人/部署参数改动或push；训练/评估后台不依赖Codex触发，不自动进入F2.2。

## 2026-10-19：F2.1 r1数据/配置准备通过，等待可用GPU服务器

- r1明确纯6D三组，r0预检原件归档result/r0_preflight；历史8D P1仅P_hist，禁止warmstart。Train36/Dev12=17243/5748帧已过滤准备独立缓存，真实6D位置、独立load_t/W15、无current/master；统计仅Train，首帧左填充不跨集。
- 4项定向pytest通过；G0既有P_hist归一化h0夹爪L1缓存聚合复现0.05930798156，完整模型forward门仍待GPU。variant_configs固定三组100k/batch4/seed1/20k保存，同主指标选checkpoint，小MLP单token融合规格已记录，尚无模型实现/训练。
- 最后已知westb:27435连接被拒绝，GPU状态未知。当前PARTIAL_PASS、F2_1_COMPLETE=false、UNRESOLVED，阻点REMOTE_GPU_UNAVAILABLE；请提供当前SSH，之后完成模型/共享初始化与真实batch GPU smoke再并行screen训练。未读Final样本、未操作机器人/部署或push。

## 2026-10-19：F2.1 预检发现原P1已包含load/current，暂停待澄清

- 新合同F2_1_LOAD_TEMPORAL_ENCODING_ABLATION r0已读取；固定HF P1 revision 33a605feb8c29dab284c3aa38740df11d2ae7835并取得config/train_config/preprocessor/postprocessor。Formal4 nomaster及P1 state为8D，末两列gripper.load/gripper.curr，归一化器不屏蔽，ACT状态与VAE投影接受整个state。
- “A0保持原P1”与“无load/单帧/W15且current排除”的语义待明确：保留8D会使A0已有load/current；移除或屏蔽则改变基线，不能静默执行。当前BLOCKED_INPUT_SEMANTICS、F2_1_COMPLETE=false、UNRESOLVED，未改模型或启动训练。
- result见experiments/force/f2_1_load_temporal_encoding_ablation/index1/result/summary.md及proposal.md；建议修订为6D位置共用输入的干净三组（A0需匹配训练），或明确接受8D基线上额外分支的问题。完整权重strict reload尚未验证，既有G0/D2夹爪scalar指标可复用但须先固定协议。Final只查split元数据，不读样本。

## 2026-10-18：F1RM 已完成，残余混杂未排除

- 用户交回f1rm_teleop_load_retrospective_match/contact_annotations.csv，15区间/10试次审核人kg；原样保存并冻结SHA256。带杯ep001后59帧UNKNOWN保留，未凑CONTACT；reason均为“1”，视觉依据文字不足如实列为限制。
- 主0.75全局1:1无放回匹配136对，接触池283帧，逐带杯试次对数[61,0,35,19,21]，4试次各≥10对且总≥40，覆盖PASS；因此按配置不运行1.00敏感性。
- 匹配后SMD=[+0.461,+0.228,+0.470,+0.097]，3/4超0.20，平衡FAIL。load差描述均值+14.62、4簇bootstrap95%CI[+11.20,+16.89]，不覆盖残余混杂；接触起点前后3/4正向、ep000反向，不能升级为接触因果证据。
- 工程OVERALL_STATUS=PASS、F1RM_COMPLETE=true；科学UNRESOLVED_RESIDUAL_CONFOUND、EXPLORATORY_ONLY。完整报告见experiments/force/f1rm_teleop_load_retrospective_match/index1/result/summary.md，紧凑包F1RM_result_20261018.zip。
- 7项pytest、独立原始数据复算全部136对/标准化/四个SMD、语法、diff及ZIP CRC通过。下一步交规划方Builder/Red-Team验收，讨论小型受控确认的新合同；本轮不再改阈值/标签，不直接训练Force或调节器。无新采、机器人/部署、原Dev/Final或push。

## 2026-10-18：F1RM 离线标注包已复制

- 独立文件夹：outputs/F1RM_标注包_20261018，约20.18MiB；含离线review.html、10段双视角视频、结果CSV表头、片段清单及简短README。复制文件逐项SHA256一致，独立目录下9项实际浏览器检查通过，未改原始标签。
- 双击review.html即可标注，无需安装/联网；浏览器草稿不等于磁盘保存，完成后须导出contact_annotations.csv并交回。当前仍等待独立RGB标注，不改变F1RM科研结论。

## 2026-10-16：F1RM 回顾性匹配已启动，等待独立 RGB 区间审核

- 按 decision_F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH.md r0 新建 result，不覆盖 F1R。重新核验 Formal5 空夹/带杯各5段、1500帧原始信号；两相机3000帧解码，10段双视角审核视频各150帧/30FPS验证通过。
- matching_config.json 已固定四变量、历史命令、全局1:1无放回遍历顺序、0.75主阈值及覆盖不足才运行的1.00敏感性；尚未拟合scaler或查看load/current效应。
- 独立接触边界尚缺，旧F1R占位表不可复用。审核入口 experiments/force/f1rm_teleop_load_retrospective_match/index1/result/review.html；每段5秒，标空夹闭合及带杯PRECONTACT/CONTACT，不确定排除。草稿在浏览器，需导出CSV放回该result；不自动覆盖磁盘。
- 当前PARTIAL_PASS、F1RM_COMPLETE=false、UNRESOLVED_PENDING_INDEPENDENT_CONTACT_LABELS。仅交阶段进度，不冒充完整匹配结果；没有训练、重采、部署/机器人操作、原Dev/Final内容读取或push。

## 2026-10-16：F1R手动预实验result已交付，正式协议未满足

- 用户已明确此次闭合由主臂手动控制。kind1空夹、kind2带杯，各5段×150帧的原始数据保留为手动预实验，不冒充同一冻结程序的正式重复。
- 实际命令起始/速度/终点不一；正式固定程序门未建立。F1R result已补summary/validation/protocol观察附注、trial_validity、contact_annotations、metrics及per_trial_metrics、raw位置和证据清单；未测效应/接触边界记NOT_EVALUATED，不填0或伪造合格结论。
- OVERALL_STATUS=PARTIAL_PASS、F1R_COMPLETE=false、SCIENTIFIC_OUTCOME=UNRESOLVED。结果包F1R_result_20261016.zip约0.65MB，含命令对比及局部RGB预览，不含重复完整视频/数组；原始HF snapshot与canonical raw不删。
- 本次只按用户要求输出result，无新动作脚本、重采、机器人指令、训练或push。下一步先验收，受控采集程序与安全范围待另行确认。

## 2026-10-16：F1R Formal5数据已接收，程序等价性待核实

- 后续用户明确映射kind1=FREE_CLOSE、kind2=CUP_CONTACT，已写condition_mapping及trial_manifest；两组ep0各4帧双视角预览与有/无杯一致，非全视频精审。映射问题已解决，当前只待闭合控制方式/采集脚本与协议等价性核实，不重复询问映射。

- 用户提供HF formal5_kind1/kind2，已固定revision并下载到数据集/formal5，各5集×150帧，合计1500帧、nominal30FPS；9D原始state/6D action、双RGB、pos/load/curr存在，原始数据未合并/删列。
- 命令预检发现十条夹爪目标曲线在起始时刻/形状/终点上不同；kind1终点约11.79–18.33，kind2约16.12–23.27，不能直接宣称同一冻结程序。图见F1R result/figures/recorded_command_equivalence.png。
- 等待用户明确kind1/kind2空夹/带杯映射及采集命令/脚本（主臂手动或自动固定程序）。尚未分析load效应、独立视频接触标签或正式判trial有效；F1R_COMPLETE=false、UNRESOLVED。不得事后冻结协议、拉伸时间冒充受控程序，也不自动要求重采或发送运动。

## 2026-10-15：F1R 受控真机反馈实验准备，等待设备与安全信息

- 用户要求参考旧采集指南并推送项目：已将F1R从臂连接填为so101_follower、/dev/ttyACM0、id=nn，历史leader=/dev/ttyACM1/id=nn仅参考、不参与正式试验。来源为docc/指南/lerobot_annotation_implementation_guide.md；设备实时校验和安全闭合参数仍未完成，不因此解锁运动。

- 读取新F1R合同：固定臂/腕姿态，同一冻结安全闭合程序，5次FREE_CLOSE+5次CUP_CONTACT；不训练policy，不改部署，不按load自动收紧，不用旧数据替代采集。
- 已交付result/PREPARATION_GUIDE.md、未冻结且不可执行的protocol.json及summary。当前采集电脑/串口、标定方向、安全范围、视频同步和急停/退出行为待确认；有效试验0，F1R_COMPLETE=false、科学UNRESOLVED。
- 未打开串口、未动电机或扭矩；先由用户确认现场设备环境，才能实现适配采集工具并做短工程设置。旧F1/F1P/F1Q结果和限制均保留。

## 2026-10-15：F1Q 停止于匹配覆盖门（UNRESOLVED_MATCHING_COVERAGE）

- 按新F1Q合同只复用冻结F1P 12集W15可靠目标，距离输入为pos/dpos/command[t−1]/dcommand[t−1]四列；标准化仅fit，逐集1:1无放回、欧氏距离≤0.75；配置在覆盖/新probe成绩前保存，无新标签、原Dev/Final或阈值搜索。
- 每集匹配对数fit[2,5,15,3,2,2]、eval[10,7,9,15,13,14]；各组均0/6达到≥20对，未满足各至少4/6门。总fit29/eval68对不能替代逐集覆盖；未拟合任何新probe。
- 所有小量eval匹配池描述性SMD约[-0.1393,-0.0235,-0.1510,+0.0967]，但有效eval集合为空，不能据此升级为混杂已排除或报告匹配后load收益。
- 已交付result/summary.md、覆盖/平衡/配对清单、泄漏审计及F1Q_result_20261015.zip；3项定向匹配测试、语法/diff和逐对无重复/距离/类别数量复核PASS。
- 科学UNRESOLVED_MATCHING_COVERAGE；依第13节要求有效匹配probe完成，保守F1Q_COMPLETE=false，授权内停止证据已交付。无放宽距离、加标注/数据、probe/policy训练、部署/真机/推送，不进入F2；下一步交规划方验收数据可比性及完成条款。原F1/F1P历史不改。

## 2026-10-15：F1P 小规模诊断完成，PROMISING_LOAD（仅探索）

- 新增六集人工接触标注完成，fit/eval均6/6满足W15两类≥5帧门；子集与标签哈希未变。旧F1仍PARTIAL_PASS/UNRESOLVED/complete=false，全量标注扩展继续暂停。
- westb:27435独立目录/venv、screen后台完成12次固定线性probe（4路CPU并行、BLAS每路1），全部收敛。未训练policy或使用原Dev/Final。本地旧threadpoolctl拟合前失败0次，失败保留；远程结果下载后仅做核对和汇总，无重复拟合。
- W15 episode等权BA：S0=86.17%、S1(load)=90.12%；主delta=+3.94pp，探索episode bootstrap95%CI=[+2.33,+6.25]pp，6/6集正向，合同分类PROMISING_LOAD。current单独−2.14pp，在load外再加current−3.90pp，不自动合并推广。
- 增益主要为未接触召回提高，接触后召回略降；开口/运动分布明显混杂，初始闭爪误报有图像证据。不能声称物理力控/抓稳/P1已利用信号，不能直接训练Force模块。
- F1P OVERALL_STATUS=PASS、PILOT_COMPLETE=true、EVIDENCE_SCOPE=EXPLORATORY_ONLY。result/summary.md及F1P_result_20261015.zip交付；下一步仅验收并讨论混杂补证新合同，不进入F2或扩大预算。D2/D2.1与G1未闭环、B3实际OC-ACT事实保留。

## 2026-10-15：F1P 独立小规模探索启动，等待新增六集审核

- 按新decision_F1P_CONTACT_SIGNAL_PILOT.md执行；**暂停原F1全量Train36标注扩展**，旧F1仍PARTIAL_PASS/UNRESOLVED/complete=false，不追认完成或改旧指标。
- 新F1P pilot-fit=原Train[0,2,4,6,7,9]；pilot-eval按原manifest剩余30集索引[0,5,11,17,23,29]固定为[10,19,29,38,49,57]。子集和旧标签哈希已在新增审核/任何成绩前冻结，原Dev/Final不读。
- 六集接触专用页面已在http://127.0.0.1:8767启动（旧F1为8766）。只标PRECONTACT/CONTACT_SUPPORTED及不确定区间，完整条目自动保存/备份/回读；不重复旧六集，不要求三个动作阶段，最多新增6集。
- F1P新result/summary.md及subset_manifest已交付；19项页面回归、3项因果特征测试、Edge隔离CSV自动保存/播放/逐帧/回读检查PASS，正式CSV未被测试改动。现在拟合0次，PARTIAL_PASS、UNRESOLVED、PILOT_COMPLETE=false、EXPLORATORY_ONLY。
- 新六集人工标签达覆盖门后才允许最多12次固定probe拟合，scaler仅pilot-fit、seed=20261015；无policy训练/架构/部署/真机/新采/推送权限，不进入F2。

## 2026-10-15：F1 前六集人工标注保存核验通过

- 用户确认完成标注；CSV实际保存30条区间，审核者lyj。格式、帧界、可靠/不确定区间及同轴冲突校验通过，原CSV未修改；快照及SHA256保存于F1 raw独立label_gate run。
- W15合法历史后，6/6集两种接触标签均≥5帧，通过合同≥4/6门槛；分支固定为继续全部Train36接触审核，不降级阶段任务。主任务尚未冻结：已审6、剩余30；Dev标签未开放、未拟合probe。
- 各集可靠PRECONTACT/CONTACT_SUPPORTED计数：0=186/189、2=138/180、4=163/215、6=164/164、7=161/168、9=145/184。人工接触标签不等于力值标定、稳定保持或离桌真值。
- 更新first_six_label_gate.json、label_coverage.json、task_freeze.json及summary；F1仍PARTIAL_PASS/UNRESOLVED、complete=false。旧六集UI仍可用，尚未扩展到其余Train。未训练、读Final、操作真机或push。

## 2026-10-15：F1 r1 信号预检完成，独立标签门待完成

- 新增本地盲审页面：`python scripts/run_f1_review_ui.py`，http://127.0.0.1:8766 。支持播放/逐帧/原帧取值、多段可靠与不确定区间、直接覆盖审核CSV、逐次备份/回读/版本冲突保护。限定六Train，不展示信号或开放Dev；保存不自动判整集通过或冻结任务。用法见tools/f1_review_ui/README.md。

- 新westb:27435服务器只读连接通过：RTX5090约32GB空闲、数据盘约30GB可用；项目及nomaster目录存在。远程HEAD=2abe71e、本地=1a4ef84，尚未同步或核验数据字节；远程base环境缺scikit-learn，未安装依赖或启动任务。详见F1 result/server_preflight_20261015.json。

- 按F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY r1执行，只授权信号诊断及固定逻辑回归probe，不授权policy训练、Force head、部署/真机/Final/push。
- Train36/Dev12共17,243/5,748帧字段、连续索引、时间戳及原生信号审计完成；load/current均非训练集常量。当前源码同一行action晚于观测，仅使用t−1及更早命令；W=1/9/15固定因果特征已缓存，3项定向测试PASS。
- 前六Train0/2/4/6/7/9完整双视角盲审素材已准备。当前工具无法原生连续视频观看，尚未完成r1明确要求的连续播放审核；任务未冻结，未看Dev标签、未拟合probe，不将联系表/全unknown表冒充精审。
- result见experiments/force/f1_formal4_signal_feasibility/index1/result/summary.md；PARTIAL_PASS、UNRESOLVED、F1_SIGNAL_FEASIBILITY_COMPLETE=false。下一步仅按evidence_samples/REVIEW_FIRST_SIX.md补六集独立RGB盲审，然后执行原合同覆盖分支。
- D2/D2.1仍未闭环，G1正式真机评价未完成，B3身份为OC-ACT而非纯ACT；本轮不修改这些事实。

## 2026-10-14：D2.1 密集窗口补证与阶段缓存分析已交付

- Dev12第120帧至结尾、Train36第200帧至结尾双视角有序逐帧联系表复核完成；Train0/2/4/6另复核120–199。不是完整视频连续播放或每帧全分辨率精审，该合同缺口明确保留。
- 48集都有可追溯释放后分离观察；P1可靠释放后退回core共606帧，原始h0及四配置阶段/辅助指标已生成。离桌、触桌及物理顶部保持仍不可可靠判定，未报告保持时长或毫米高度。
- 12Dev/5748帧缓存有限性/维度/连续帧及旧D2回放复现PASS（最大MAE差8.509e-6）；96段双视角片段帧数、30FPS、1280×480核验PASS。B3身份与归一化等价核验保留。
- 更新result/summary.md、事件表、指标及manual_review.md；下一步仅Dev3/28/58现成视频最小人工可见性复核。PARTIAL_PASS/UNRESOLVED、DIAGNOSIS_COMPLETE=FALSE；无训练、部署修改、Final、真机或push。

## 2026-10-14：D2.1 补证进行中

- B3配置、参数键、模块及实际forward核验为OCACT_VERIFIED，撤回纯ACT描述；不改模型。
- P1 normalized/physical时间集成等价校验通过（max误差9.918e-5），补1/5/10/30/100合法horizon误差。48集双视角1秒抽帧初审完成、35集候选释放区间；尚未全帧精审，离桌/保持阶段边界不可靠，阶段指标未完成。
- D2.1当前PARTIAL_PASS/UNRESOLVED，DIAGNOSIS_COMPLETE=FALSE；需继续密集事件复核。无Final、训练、真机。结果见d2_1_evidence_completion/index1/result。

## 2026-10-14：D2 部分结果交付（UNRESOLVED / PARTIAL_PASS）

- P1/R3各12Dev、5748帧推理及四配置回放完成；48Train/Dev信号审计完成，P1实际时间权重/限幅函数和同episode未来mask核验通过。
- 实际G1 B3 checkpoint为oc_act且object_conditioning_enabled=true，与D2标准ACT定义冲突，暂停该对照；没有替换模型。48集视觉事件审查尚未完成，事件UNKNOWN，阶段指标不报告。
- DIAGNOSIS_COMPLETE=FALSE，分类UNRESOLVED；完整合同尚未履行，不能直接训练优化。交付见D2 result/summary.md。G1正式评价未完成、用户暂停真机；本轮无Final、训练或真机动作。

## 2026-10-14：D2 动作链诊断本地预检完成，等待服务器

- P1 重跑已成功完成：12 Dev episodes、5748帧，12份NPZ、四配置回放指标和p1.done均存在；指标已下载为result/p1_unphased_metrics.json。阶段为unknown，D2完整诊断仍未完成。

- 重启修复：首轮因文件句柄 soft limit=1024 出现 Too many open files，无有效输出。保留失败日志后在启动脚本设置 ulimit -n 65535，P1 已通过连续 300 batch 真实 Dev 推理（约30秒），全量与回放仍待完成检查。

- 后续更新：已连接用户提供的新端点 westd:52291（RTX 5090 32GB）；原训练工作区、冻结 P1 checkpoint 和 DINO 缓存可用。后台 `screen d2_p1` 已启动固定 Dev 推理与无硬件 2×2 后处理回放，模型成功加载并进入 GPU 推理。阶段事件审查未完成，当前输出阶段为 unknown，不报告阶段误差或根因。

- 按 D2 r1 零训练合同执行元数据/split 和部署代码静态审计；36 Train / 12 Dev、8D/6D、30 FPS 声明已核验，单位与对齐尚未核验；未读 Final 内容。
- 本地缺冻结 P1/B3/R3 checkpoint、默认 Python 缺 torch，服务器已由用户关闭。完整事件审计、全 Dev 推理、四配置后处理回放尚未执行；DIAGNOSIS_COMPLETE=FALSE，UNRESOLVED / BLOCKED。
- 证据见 `experiments/position/d2_action_chain_diagnosis/index1/result/`；没有训练、参数修改或真机动作。下一步需开启原训练服务器，以 P1 全 Dev 优先完成诊断，GPU 累计预算 4 小时。

## 2026-10-14：G1 定性复盘交付，等待网页 GPT 决策

- 用户澄清：P1 抓取—拿起—放下链稳定；B3 动作链不稳定；R3 能随杯位调整夹爪。三个模型共同存在抬升低、停留短，均为定性观察。
- 小范围杯位可能被张开夹爪覆盖，成功抓取不能直接证明定位泛化。用户希望暂停追加真机转入优化；尚未开始新诊断或训练。
- 用户确认没有可用真机运行日志；15 段视频不是正式 trial 数量。完整 G1 未完成，成功率未知。
- 网页 GPT 交付：`experiments/position/g1_real_robot_generalization/index1/result/REAL_ROBOT_REVIEW_20261014.md`、metrics、validation_review_20261014、proposal；先验收和讨论，再冻结新 decision。

## 2026-10-13：G1 真机位置泛化首次真机测试完成（定性）

- G1 决策已读取；冻结 shortlist 为 `B3_ACT`、`P1_OCACT`、`R3_FILM`，不增加其它模型、不重训、不读取 Final。
- 已生成 81 次固定试验计划：Phase A 为每模型 3 次 nominal-center safety/function check（9 次，不计主成功率）；Phase B 为 12 strata × 2 repeats × 3 models（72 次）。
- 已在本机接通 SO101 串口与 top/wrist 双相机并下载三份冻结 checkpoint，硬件阻塞解除；完成首次真机定性 rollout。
- 部署修复：修正三份 checkpoint `config.json` 中指向远程 `/root/autodl-tmp/...` 的 `object_conditioning_prototype_path` 为本机路径（prototype 字节一致）；离线缓存 DINOv2-S 权重；生成 8D NOMASTER 元数据 `数据集/formal4_kind_merged_nomaster`（模型 8D、HF formal4 为 9D）；腕部相机 `fourcc: MJPG` 达 30fps。
- 抖动缓解：开 `--policy.temporal_ensemble_coeff=0.01`，`--robot.max_relative_target` 下调至 2。
- 定性结果：B3_ACT 与 P1_OCACT 均有成功抓取但保持时间较短，P1_OCACT 表现最佳；R3_FILM 能接近并识别/追踪塑料杯，但拿起与放下动作微弱甚至退化。
- 81 次正式 trial（Phase A 9 + Phase B 72）尚未完成，`POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE` 未置位。

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
