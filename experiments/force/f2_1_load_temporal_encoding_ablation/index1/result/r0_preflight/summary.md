# F2.1 预检：输入语义需要澄清，未启动训练

- OVERALL_STATUS: BLOCKED
- F2_1_COMPLETE: false
- SCIENTIFIC_OUTCOME: UNRESOLVED
- Blocker: BLOCKED_INPUT_SEMANTICS

## 已核实的事实

现有 P1 名称对应 `P1_ocfix2_s1/checkpoints/100000/pretrained_model`，不是 T0 最初被排除的 P1_s1。公开模型为 `QYyyyyyyy/G1_P1_OCACT_100k`，本次API取得 revision `33a605feb8c29dab284c3aa38740df11d2ae7835`，随后按该revision下载config、train_config、preprocessor、postprocessor四份小型JSON，保存原文和SHA256。不下载完整权重、不声称strict reload/forward已通过。

公开config的robot state为8维；原始Formal4 nomaster元数据的顺序为：

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos,
gripper.load, gripper.curr
```

公开preprocessor只是重命名、加batch、设备转换和8维归一化，没有去除或屏蔽最后两列。源码ACT的机器人状态投影接收整个state，训练VAE状态投影也接收整个state。因此从配置/数据/schema/代码可知，原P1的既有输入路径已经包含单帧load和current；这不等于已经证明模型有效利用它们。

## 与合同的歧义

合同§6要求A0完全保留P1输入，§2/§8又以无load/单帧load/W15为研究对照，§6/§20要求current排除。§2的“No new load branch”可以理解成只禁止新分支，并不自动说明原state是否允许保留load/current。

两种解释会产生不同实验，不能由执行器静默选择：

1. 保留P1输入：A0已经有load/current，A1是重复输入当前load，A2是额外历史编码；可以研究“额外分支表示”的增益，但不是干净的无load对照，也不能称current已排除。
2. 去掉/屏蔽load/current：可构造干净对照，但改变了P1输入语义，A0不再是原checkpoint的原样输入/输出。无论删列还是置零都需要明确授权，不能当纯工程修复。

## 其他预检

- 冻结split元数据仍为Train36/Dev12/Final12，公开P1训练episode列表与T0训练清单一致。只读split成员信息，不读Final样本/视频。
- P1配置：OC-ACT、双相机640×480、6维action、chunk100、n_action_steps1、temporal_ensemble_coeff=null、gripper_loss_weight3、MEAN_STD。
- 历史训练：100000步、batch4、seed1、AdamW、无scheduler、每20000步保存；新实验预算尚未启动或冻结为正式训练配置。
- 已有scalar夹爪指标存在：G0的归一化h0 gripper_l1_h0及D2的反归一化h0 gripper MAE，不能声称“没有指标”。后续需选定一个已有协议并明确聚合/后处理，不能训练后挑指标。
- 完整权重、processor state和prototype的strict reload/实际forward尚未验证，基线完整身份门未通过。未尝试用其它checkpoint替代。

## 交付边界

当前仅预检证据：baseline_identity.json、baseline_metadata/、metrics.json、validation.md、implementation_diff.md、leakage_audit.md及proposal.md。不是合同要求的完整训练结果包。没有A1/A2实现、训练、延迟测试或新离线性能结果，不能填写虚构分数/曲线。

下一步由用户/规划方澄清输入语义并修订合同，再准备匹配初始化、W15启动填充、load Train-only归一化及相同训练预算。不会因为本轮允许架构可行性实验而把F1RM的混杂未决改写为已排除。
