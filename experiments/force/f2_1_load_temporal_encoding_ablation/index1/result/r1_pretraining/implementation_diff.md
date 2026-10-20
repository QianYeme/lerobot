# 当前实现差异

## r1 当前实现

新增scripts/prepare_f2_1_r1.py及tests/scripts/test_f2_1_preparation.py。真实数据投影6D，load为独立输入字段，共用Train统计；没有改ACT/processor/部署。variant_configs为训练前规格，不是已实例化模型。公共ACT先初始化、反馈模块使用独立固定seed初始化，以避免新增层消耗RNG改变公共权重；这项模型hash验证尚待GPU。

P_hist仅历史，不warmstart。A0未来loader不得消费共用缓存中的load字段；A1只消费单帧，A2只消费因果W15。

## r0历史说明（输入语义已由r1解决）

未改模型、processor、数据列、归一化、训练器或部署代码。仅增加scripts/preflight_f2_1.py及本轮预检报告。

不能通过偷偷把8D state末两列置零、删列或只在一个分支屏蔽current来“修复”本实验：这会改变A0身份或导致额外变量。A1/A2尚未实现，不声称它们已经只有时间表征差异。
