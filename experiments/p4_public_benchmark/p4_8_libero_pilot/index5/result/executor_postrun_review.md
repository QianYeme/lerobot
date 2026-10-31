# P4.8-r5 执行后核对与关键发现

## 本轮已经实际完成

- 三组各10000个正式optimizer updates；checkpoint5000/10000及完整optimizer/RNG保留服务器，10k模型均被真实加载用于评估。
- 每模型完整Dev9共1391帧；27条逐episode指标；模型预测缓存与原生训练日志留存。
- 三模型各10次同init10–19、同seed1仿真，共30次。所有10组起始公开8D状态跨模型逐元素相同（最大差0），见paired_reset_verification.json。
- P1各27个固定Dev锚点做对象ON/OFF诊断，共54条；后续只读CPU审核optimizer，没有追加训练更新。
- 延迟在所有训练/仿真工作结束后，按单模型、batch1、chunk100、预热10次/测量50次独立测量。ACT约8.56ms，Legacy约17.02ms，Fixed约17.25ms；这是整块预测调用，非每动作控制周期。
- Dev视频/依赖提前缓存齐全；评估启用HF离线标志，实际使用服务器本地模型、数据、DINO/ResNet和MuJoCo/EGL资源。没有本地推理或真机操作。

## 成功率和配对结果

| 模型 | 步数 | 本任务仿真成功 | 成功init |
|---|---:|---:|---|
| ACT-Default-LIBERO | 10000 | 4/10 | 10、11、12、13 |
| P1-Legacy-LIBERO | 10000 | 4/10 | 10、11、12、13 |
| P1-Fixed-LIBERO | 10000 | 4/10 | 10、11、12、13 |

三者在init14–19均未于280step预算内触发成功。配对结果完全相同，不支持P1优于ACT、或Fixed优于Legacy的结论，也不证明模型能力等价。仅1任务/1seed/10次的10k pilot，非完整LIBERO-Object leaderboard。

成功依据是官方env success predicate，不是人工看视频评分。Native wrapper在terminal后自动reset，视频保留每次执行前帧；不要把最后一帧直接当成终止后最终物体姿态。没有阶段oracle时，first_failure_stage保留UNKNOWN；不凭超时编造漏抓/掉落等阶段结论。

## 关键机制证据：末期对象路径影响极小

| 指标，27锚点/模型 | Legacy | Fixed |
|---|---:|---:|
| 10k signed gate | -6.75154e-6 | -4.14562e-7 |
| 5k signed gate | -3.01057e-5 | 8.57610e-6 |
| h0机械臂raw ON/OFF平均绝对差 | 7.46486e-8 | 1.90307e-8 |
| h0夹爪raw ON/OFF平均绝对差 | 2.38706e-7 | 8.36021e-8 |
| 残差/query范数比均值 | 1.56750e-6 | 8.41063e-8 |

raw单位是OSC控制器输入，不是SO101关节角。受检Dev锚点上的最终策略对对象残差几乎不敏感。该结果与三者配对表现近似一致，但不能推出训练全程从未使用该分支、其它视觉信息失效，或该现象造成了所有失败。

只读optimizer审核确认：两组gate都requires_grad=True、各在一个AdamW参数组中，LR1e-5、weight_decay1e-4，optimizer step=10000，Adam一/二阶矩有限且非零。因此不是简单的gate漏登记到optimizer；矩值是历史裁剪后梯度统计，不是本轮新测的单批梯度。

目前更应关注“对象路径为何只产生极小作用”，而不是用当前同分宣称单/双标准化没有差别。零门控初始化、随机adapter与CNN路径形成捷径是待检验假设，当前没有因果归因证据，不自动修改gate/init/LR或增加训练预算。

## 完整性与边界

服务器原始artifact manifest256项全部SHA复核通过；自动ZIP本地CRC及所有打包文件SHA通过。DINO不带可靠git revision的现有源码缓存已另行快照，权重/源码树哈希见原型验证文件。大缓存与checkpoint不下载到本地，不删除服务器证据。

原始自动交付ZIP及manifest保留。REVIEWED交付包在原包基础上增加本核对、配对状态验证、optimizer审核、源码依赖快照和最终日志说明，不覆盖旧r0–r3结果、不重新执行benchmark。独立Builder/Red-Team仍PENDING。无Final15、第四模型、新seed、10任务扩展、HF或Git发布。
