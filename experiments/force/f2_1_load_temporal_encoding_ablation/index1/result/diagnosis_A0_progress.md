# A0 诊断进行中（不是正式训练重启）

> 最新：诊断已完成15000步、耗时1315秒，NOT_REPRODUCED_WITHIN_FIXED_BUDGET（见diagnosis_A0_summary.json），没有捕获新故障，根因仍未确认。用户随后明确要求A0并行训练，已启动一次同配置/seed1从0重试，独立目录results_A0_retry1；诊断权重未沿用，原失败记录保留。以下是诊断启动时的记录，不是仍在运行的状态。

当前尚未确定原始故障根因，没有应用修复。

## 已确认

原A0在首个20k保存点前退出，只有日志，没有失败时权重/batch。clip_grad_norm_报告non-finite total norm不等于已证明每个参数梯度NaN。算子级GPU样例证实：两个有限的1e20梯度在FP32范数计算中就能触发同类报错，FP64范数仍有限；这个合成样例不能归因给原A0。

原坏权重无法精确恢复。为建立实际故障反馈环，独立screen F21_DIAG_A0从相同seed1/公共初始化/数据顺序开始，最多15000步或1800秒，仅Train数据，无Dev模型选择；所有诊断权重都禁止进入A0/A1/A2科学比较。观测到3000步时尚无故障，初始公共hash与正式A0一致。

## 四个待证伪分支

1. FP32范数归约溢出：全部梯度元素有限，但norm32非有限、norm64有限。
2. CVAE数值异常：方差相关输出或梯度先坏，需捕获mu/logvar/预测和异常反向算子。
3. 注意力/其他算子反向异常：输入及loss有限，backward首先产生NaN，需同batch anomaly trace定位。
4. 输入异常：失败batch含非有限值；此前数据审计有限，使其优先级较低。

## 后台执行与证据

运行根目录 `/root/autodl-tmp/F2_1_r1_20261019`：

- `diagnosis_A0.log`、`diagnosis_A0/progress.json`：有界重放进度。
- 若故障出现：`diagnosis_A0/failure_capture.pt`保存当时权重、optimizer、实际batch和forward前CPU/CUDA随机状态；`failure_report.json`记录输入/输出/梯度与试次帧号。
- `F21_DIAG_ANALYZE`等待捕获后自动运行3次同权重/batch/RNG反向重放，记录`diagnosis_A0_exact_1.log`等及对应`capture_replay.json`。
- 最终后台写`diagnosis_A0_summary.json`。若达到预算未复现，会如实记NOT_REPRODUCED_WITHIN_FIXED_BUDGET，而非挑选一个猜测原因。

当前正式A0不重启，不调LR/seed、不开关异常检查、不跳坏batch，A1/A2不停止或改动。不训练Force新结构，不读Final，不push。此轮仅诊断，后续修复或重跑需根据证据与原合同公平性决定。
