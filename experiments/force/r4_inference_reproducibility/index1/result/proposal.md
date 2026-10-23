# 下一步建议：Final Dataset V1合同规划（未执行）

## 已解决与未解决

R4两seed A–D通过，完整chunk重复最大差0；只将cudnn.benchmark关闭，TF32保持不变。评估数值可复现后，global seed差异、Full60空间收益混合、整体回放误差未改善等结论仍在；local固定骨干上的离线拟合较稳定。没有新增真机证据。

因此不建议继续在旧60段上追加架构、训练seed或SWA/SAM来追逐回放分数。先验收R4，再按已有预规划提升最终数据质量和覆盖。

## 请网页GPT先审阅

1. Builder：完整核对R4合同§17/18输出、两seed A–D、身份不变、fresh指标重算、原门槛保留。执行方已有检查不等于独立审阅。
2. Red-Team：重点检查benchmark单变量对照是否足以支持runtime归因、固定fixture与全量D的区别、跨机器可移植性、Full60训练/诊断污染边界、seed描述性标签是否过度解释，以及local拟合是否被错误宣传为力控。
3. 通过后生成新的FINAL_DATASET_V1采集/QC/split-freeze decision；不在R4内启动采集或训练。

## 新合同应冻结的内容

- 目标约120–150条完整、自然完成的演示，杯型/水量/相机/任务与动作风格稳定；主变量以位置及小幅初态抖动为主。
- 抓准后稳定闭爪，明确抬升和约0.8–1.0秒保持，再放下与释放；用少量先导演示做视频+关节/load可观测性QC后，再批量采集。
- 物理位置区和抖动范围先校准。R4中的5×3/mm坐标是预规划条件示例，不是对旧D1视觉12组的物理标定结论；正式方案需按安全可达区确认。
- 必须在新数据用于模型选择前冻结QC排除规则、位置分层Train/Dev/Final及未见测试保留。未来36/60/90/120规模子集必须覆盖全部位置区，且不可取first-N。
- 新训练仍需后续授权；全60旧数据可保留作历史/工程对照，但不继续称未见测试。后续离线评估沿用明确记录的最终runtime设置，并在目标机器复验。

## 可直接提交网页GPT

> 请按项目工作流独立完成R4 Builder Review与Red-Team Review。读取summary/validation/metrics、完整R4 decision、两seed full_replay_repeat、原始audit_config以及independent_cpu_validation。两seed的全60两遍、100步6维chunk最大差均0，原<0.001门槛未改；唯一评估开关改变是cudnn.benchmark True→False，TF32保持True。原始失败和服务器报告未改写。Fresh关键指标重算后仍是global seed差异明显、Full60空间收益混合、local在固定global上相对稳定，全部为训练/共同诊断，不是泛化或力控真机证据。请先裁决R4验收；通过后规划新的FINAL_DATASET_V1（约120–150段，位置覆盖与抖动、统一抬升/0.8–1.0秒保持/放下、QC和split冻结），不要自动追加旧数据训练、SWA/SAM或真机。
