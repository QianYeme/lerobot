# Executor Red-Team precheck

最主要风险是无标签motion proxy与目标语义混淆：保留Train审计中proxy响应差值通过，并不意味着prototype对应被抓物。预注册16候选门槛是工程启发式，不是decision强制数字，也不能将12/16误说成监督目标发现准确率。没有事后改阈值或人眼替换原型。

其它限制：Train保留proxy正例很少；光流可能随遮挡前景漂移；两P1尚无全链路输入/梯度/配对训练验证；100步ACT使用同一真实batch，仅验证工程可训练/恢复，不证明数据集收敛或任务效果；10FPS编码时钟不等于20Hz物理时钟。空指标表不是零成功率。

没有新增seed/模型、没有追加10tasks、没有Final/真机、没有把Object OFF当训练消融。当前正式训练全部0步。独立Red-Team Review PENDING。
