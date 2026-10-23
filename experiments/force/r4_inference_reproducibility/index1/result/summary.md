# R4结果交付：重复性修复，原有seed差异结论未翻转

日期：2026-10-23。合同：[R4 r0](../decision_R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE.md)。服务器原始报告见[server_original/result](server_original/result/summary.md)，本文件为执行方复核，不冒充独立外部审阅。

## 1. 当前结论与状态

**`REPRODUCIBILITY_FIXED_RUNTIME`；`R4_REPRODUCIBILITY_PASS=true`。**

seed3和seed1在最终评估设置下分别完成A–D；各自两次Full60回放的完整100步、6维动作逐元素相等，最大差为0，严格通过原`<0.001 native`门槛。此次没有修改门槛。

执行与交付已完成；`R4_COMPLETE=false`保留至独立外部Builder/Red-Team审阅完成，不把执行方检查当成网页GPT验收。**技术重复性gate通过，不等于新数据采集已开始或已获另一个执行合同。**

## 2. 异常是如何定位的

| 检查 | 原historical设置 | 仅关闭benchmark后 |
| --- | ---: | ---: |
| A：同张量、同进程完整chunk最大差 | 0 | 0 |
| B：同张量、跨进程完整chunk最大差 | 0.327590942 | 0 |
| B：同张量、跨进程h0最大差 | 0.130935669 | 0 |
| C：固定IDs重新解码/预处理两次 | 原设置B失败，未越级执行 | 输入一致，chunk差0 |
| D：完整回放两次 | 原F5.1 h0最大差0.117225647 | S3、S1均0 |

历史F5.1差异最大的样本为ep3/frame77。B层输入hash相同，最早观测到不同的模块hook为`model.backbone`；A层稳定，支持跨进程runtime算法选择相关原因，而非把现象归因训练并行或坏checkpoint。

唯一最终控制改变：**`torch.backends.cudnn.benchmark=True → False`**。matmul TF32及cuDNN TF32均保持True；没有关闭全部TF32、加新训练seed、改权重/归一化/模型结构。实际真机控制器和旧F5训练/评估脚本未被覆盖；本轮使用隔离的R4评估入口。

此处没有捕捉具体cuDNN kernel ID或做backbone内部逐算子定位，结论限于受控运行时开关与跨进程差异关系。也不保证另一GPU/PyTorch版本自动同样可复现，迁移时须重新验证。

## 3. 全量验证及独立复算

| checkpoint | 每次帧数 | 每次batch数 | 全部输入hash | 100 horizons×6动作最大差 | h0及关节MAE差 |
| --- | ---: | ---: | --- | ---: | ---: |
| G60-S3/80k | 28,739 | 7,185 | 一致 | 0 | 0 |
| G60-S1/80k | 28,739 | 7,185 | 一致 | 0 | 0 |

来源：[full_replay_repeat.csv](full_replay_repeat.csv)、[seed1补证](seed1_repeat_verification.csv)。每次覆盖全部60 episodes，batch4并保留尾批3，所有100步均检查，未只挑h0或少量帧。

此外，本轮在服务器用**独立CPU验证脚本**重新读取两seed全部D0/D1 chunk文件和input manifest，确认每一chunk逐元素相等、缓存h0与chunk相符、IDs唯一完整；重算5个fresh global的native h0误差；重新散列12个checkpoint配置/权重/归一化文件，并交叉核对4个global与3个local的历史SHA。结果见[independent_cpu_validation.json](independent_cpu_validation.json)。该脚本不是新GPU推理，也未重新解码视频或独立重推全部local。

## 4. 修复后F5.1关键指标

以下均来自最终`benchmark=False`设置下的fresh global回放和冻结local重组，不复用旧global预测。全部28,739帧仍为**Full60 TRAIN-SET / COMMON-DIAGNOSTIC ONLY，NOT HELD-OUT GENERALIZATION**；历史Dev/Final已被本轮Full60训练消耗，不能据此声称泛化提升。

| 模型 | 夹爪MAE native | 机械臂MAE native |
| --- | ---: | ---: |
| G36-S3 | 1.232587 | 1.789554 |
| G60-S3 | 1.399581 | 1.831726 |
| G60-S1 | 2.445643 | 2.690024 |
| G60-S2 | 1.971966 | 2.263356 |
| G60-S4 | 2.614576 | 3.310617 |
| New-GL-36 | 0.251422 | 1.789554 |
| NewGL60-L3 | 0.313384 | 1.831726 |
| NewGL60-L1 | 0.318368 | 1.831726 |
| NewGL60-L2 | 0.304030 | 1.831726 |

来源：[postfix_f5_1_metrics.csv](postfix_f5_1_metrics.csv)。native不是末端毫米或力单位，arm为5关节均值。R4没有重新运行B0，因为合同§8要求的关键组不含B0；不将旧B0值混入新表。

冻结位置规则重算后：

- S3匹配状态动作距离：G36的0.394871 → G60的0.435617；边缘幅度偏差0.420250 → 0.164667，方向余弦0.846209 → 0.916230，仍支持该seed的描述性改善。
- S2仍标`POSITION_SEPARATION_IMPROVED`，S1/S4仍`UNRESOLVED`。该多条件标签不要求between/within同时提升；原始冲突保留，不能只看标签。
- 必要三seed匹配距离std=0.102269，大于S3新旧差0.040746，自动描述标签仍`FULL60_SEED_VARIANCE_DOMINATES`。这不是显著性检验、因果方差分解，亦未证明新旧方差降低，因为缺少同口径G36多seed对照。
- 固定G60-S3上的local1/2/3夹爪MAE范围0.304030–0.318368，std=0.005943，全部改善同一global且逐元素保持arm；但仍不是接触力控制、真机成功或位置泛化证据。

详表位于[postfix_f5_1](server_original/result/postfix_f5_1/interpretation.json)。沿用原冻结12组/4,054匹配对，不新增位置分组、挑seed/step或改变分类。固定候选仍S3/80k+local3/7k。

**研究解释：修复评估重复性没有消除已观察到的global seed差异；不能再把这些差异主要归咎于本次已定位的评估噪声。** 但这也没有证明数据规模、归一化或优化覆盖中的哪一项是根因，不能直接推导出某个新架构一定有效。

## 5. 下一步建议与授权边界

先将本包交网页GPT完成独立Builder/Red-Team，随后制定`FINAL_DATASET_V1`的采集/QC/split冻结decision。按R4预规划，目标约120–150段，覆盖杯位和小幅抖动，并统一抓取—抬升—保持约0.8–1.0秒—放下—释放链条。

本轮停止继续在旧60段上追加训练或结构搜索；不提前引入SWA/SAM，不直接启动36/60/90/120规模实验。下一合同还需确认物理位置覆盖、采集安全、样本QC和未见测试保留规则。**技术gate通过只是进入下一阶段规划的前提，不替代独立审阅和新合同。**

运行结束时GPU空闲，盘仍约36GiB；权重、全部chunk/输入manifest/日志保留服务器`/root/autodl-tmp/R4_20261023`。本轮没有新训练、采集、真机、Git/HF发布。原报告和中途失败证据均保留，轻量ZIP不含完整大数组或权重。
