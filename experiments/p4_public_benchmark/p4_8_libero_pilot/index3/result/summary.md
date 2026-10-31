# P4.8-r3 result — automatic prototype gate executed, formal pilot not started

| Gate | Status | Evidence |
|---|---|---|
| G2 automatic prototype | **BLOCKED / G2_AUTO_PROTOTYPE_UNRELIABLE** | frozen discovery rule yielded12 candidates vs pre-registered16; qualitative robot/background contamination |
| Timebase | **TIMEBASE_BLOCKED** (collection-time provenance); native interface PASS | measured one step=.05s at20Hz; dataset timestamps=.1s; no resampling |
| Training | **TRAIN_PILOT_NOT_STARTED** | three formal jobs0/10000; native ACT disposable100-step smoke PASS |
| Simulator policy rollout | **SIM_ROLLOUT_BLOCKED** | no trained policy rollout, no success-rate estimate |

实际完成了Train35全部35集315张事件窗口图像的DINO提取与光流跟踪。发现池23集获得12候选/7个episode，Train-only保留proxy检查12集获得7候选/5个episode。自动选出ep1171/1187/1151/850四个不同来源的4×384 bank，重复DINO提取/确定性选取后两次SHA一致：`1d78709651259d73edc5f1d00a687f30ce0c251b3e79b5045d73478673953d66`。这份库只用于诊断，**不批准训练使用**。

proxy正例Q75对静态背景Q95间隔0.02684、对premover Q95间隔0.05823，均大于预设.02；范数/多样性通过。这些不是目标定位精度，不能抵消候选覆盖不足与可视化污染。候选门槛16由本轮实现预注册而非decision直接指定，未事后调低。静态桌面被前景遮挡后光流跟随机械臂、局部机械臂先静后动，是目前算法的实质不足。详见qualitative_failure_evidence.md和figures/auto_selected_temporal_tracks.png。

原生官方ACT实际工程smoke：Train35统计5476帧，Train ep813真实batch8，state8/action100×7，两路RGB；100步loss/梯度有限，完整optimizer/RNG checkpoint恢复后下一步权重差0，参数51,600,263。仅接口特征有override，训练超参原生默认；无DINO/Force/Load/OC。所有更新均为丢弃的工程测试，不计入正式10k、不用于模型排名。

另用原始FrozenDINOv2Small实现执行单/双在线归一化数值probe：实际输入最大差9.24849，patch feature平均绝对差1.65468，Fixed与prototype单次路径差0。该probe不是完整P1策略forward，尚未验证CNN主干不变或配对初始化。

18个Train35双相机共享视频分片校验完成。网络API/Hub接口失败及重试日志保留；仅在进程内使用服务器已有网络加速设置，无凭据写入交付。nohup gate队列已完成退出，没有正式训练自动队列。服务器2026-10-30T23:00:04+08:00：GPU2MiB/0%，约26.9GiB数据盘可用；不需要清除历史证据以启动未过gate的模型。

原r2 decision/result已完整归档revisions/r2，原ZIP保持不变。当前r3规范在index3/decision.md，r3结果位于result/。源码、配置、原型、图、CSV、日志、脚本SHA与大文件服务器指针均交付。大缓存和ACT工程checkpoint留服务器，不放进紧凑ZIP。

下一步仅审核blocking_proposal.md：决定保留全自动方向另立改进轮次，或明确允许与历史P1一样的少量执行者目标先验。没有擅自使用旧四中心、降低阈值、训练其它模型、扩任务、读Final15、做真机或发布HF/Git。独立Builder/Red-Team均PENDING。

验证：5个脚本编译、JSON解析、100行实际smoke记录与空正式指标表校验通过；本轮STATUS/工作日志定向diff检查通过。全工作树diff检查被预先存在的无关指南尾随空格阻断，原输出已保存，未清理用户改动。
