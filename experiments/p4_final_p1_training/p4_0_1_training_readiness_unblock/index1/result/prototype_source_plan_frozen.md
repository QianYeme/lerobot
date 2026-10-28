# Train-only 原型来源规则（生产执行未解阻）

先读取并校验冻结Train90 ID hash；只从Train90中按episode升序取0%、25%、50%、75%序位的四集，分别取原生总长25%处top帧作为四个候选来源。仅Train候选可审核杯区域；若候选不适用，停止并proposal，不借Dev/Final替补。具体裁剪/特征流程保持P1既有实现，由后续decision冻结。
本轮没有读取图像构建原型，没有创建bank，没有验证生产入口白名单强约束，因此PROTOTYPE_TRAIN_ONLY=BLOCKED。不能把计划当成已构建资产。
