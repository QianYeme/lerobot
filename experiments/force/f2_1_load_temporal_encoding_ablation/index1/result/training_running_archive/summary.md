# F2.1 r1：训练状态快照

OVERALL_STATUS=RUNNING；F2_1_COMPLETE=false；SCIENTIFIC_OUTCOME=UNRESOLVED；SINGLE_SEED_EXPLORATORY。

截至UTC 2026-10-19T11:45:34.510940+00:00：A0: step=100, exit=None; A1: step=42000, exit=None; A2: step=42100, exit=None。失败组：[]。screen实际状态见runtime_snapshot.json，不把已退出的组说成运行中；未自动重启或调整参数。实际GPU：NVIDIA GeForce RTX 5090, 32607 MiB, 9830 MiB, 100 %。服务器connect.weste.seetacloud.com:53289，运行目录/root/autodl-tmp/F2_1_r1_20261019；旧远程dirty工作区未覆盖。

当前尝试记录：{"A0": {"attempt": 2, "output": "/root/autodl-tmp/F2_1_r1_20261019/results_A0_retry1/A0", "previous": {"output_dir": "/root/autodl-tmp/F2_1_r1_20261019/results/A0", "exit_code": 1, "last_logged_step": 14200, "reason": "non-finite gradient norm; root cause not confirmed"}}, "A1": {"attempt": 1, "output": "/root/autodl-tmp/F2_1_r1_20261019/results/A1", "previous": null}, "A2": {"attempt": 1, "output": "/root/autodl-tmp/F2_1_r1_20261019/results/A2", "previous": null}}。A0若为retry1，是用户明确要求的同seed/同参数从0重试，不覆盖原失败，也不把诊断模型纳入比较；本轮不自动反复重启。

## 已通过的门

- 原始parquet SHA256与本地一致，prototype SHA256=114de07e6cf6c0a2eba7fbede910212e0be6b0740751ab42855780c63b5fb876。只用Train36/Dev12；Final只查split元数据，未读取样本。
- 三组共同6D位置、无current/master；A0不接收load，A1仅当前归一化load，A2因果W15首帧左填充。load/state/action统计仅Train。VAE的状态投影也为6D。
- 真实batch4、chunk100训练forward/backward、有限梯度/裁剪、非零load层梯度、optimizer参数更新、strict checkpoint重载和current额外键不影响输出检查通过。Dev指标、合法horizon与M1/M2入口做实际batch smoke。
- 三组公共初始化SHA256完全一致：a7105e4b47051dc74e2afe5247a4a20c127f4147a159e82a0cc86b18e163baf5。不使用P_hist策略权重warmstart；共享视觉/object预训练来源相同。
- 完整历史P_hist的5748帧Dev主指标重跑为0.05930956003，原G0为0.05930798156，差约1.58e-6，低于预设1e-4容差；这次不是仅缓存聚合。

## 正式运行协议

三组均100000 optimizer steps、batch4、seed1、AdamW、lr1e-5/backbone1e-4、weight_decay1e-4、clip10、无scheduler、无AMP。每20000步保存并完整评估Dev12，按归一化h0夹爪L1最低选checkpoint，平局较早。

A1/A2使用相同位置的额外load token，MLP1或15→32→512/GELU，新增512维位置embedding，always-on无gate。总参数A0=74938375，A1=74955847，A2=74956295；A2比A1仅多448。确切计数见latency_and_params.json。延迟与RTX4060实测尚未完成，smoke秒数不能冒充推理延迟。

每次固定checkpoint评估还输出物理单位夹爪/臂误差、逐Dev episode主指标、合法1/5/10/30/100目标horizon，及相同观测上的load置零与±0.1标准差离线扰动（不是物理力扰动）。主指标不因次指标改变。

当前screen和输出路径以training_manifest.csv/active_runs.json为准。A0原尝试日志为A0.log，重试为A0_retry1.log/results_A0_retry1/A0；A1/A2仍为原日志与results目录。结束写对应exit文件，SSH断开不依赖Codex触发。每20k checkpoint保存在各自output_dir/checkpoints/020000/pretrained_model等；latest_optimizer.pt只保留最近一次优化器状态，不宣称实现完整可复现断点恢复。

## 当前边界

现在只有早期训练健康证据，没有完成的模型比较，不可根据训练loss宣布A2更好。所有科学结论等待固定checkpoint评估和机制分析。预计时长应按后续稳定吞吐更新；约0.1秒/组/步的启动观测对应纯训练约2.8小时，另加5轮完整Dev/机制评估，暂估总3–4小时，不是保证。

原r0见r0_preflight；r1数据准备报告见r1_pretraining。没有机器人操作、重采、Final样本、部署参数修改或push，也不进入F2.2。
