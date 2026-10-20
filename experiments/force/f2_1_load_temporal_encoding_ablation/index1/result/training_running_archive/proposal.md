# 当前合同内的后续

等待三组100k及每20k自动Dev评估结束，检查exit/complete、每组5个checkpoint、per-episode主指标及M1/M2；按冻结规则选择checkpoint，测推理延迟/显存并生成最终result交Builder和Red-Team验收。

不因训练loss较低提前选赢家，不延长某组、不增seed或FiLM/contact gate/current，不直接改真机部署。若任一训练失败，先保留日志诊断工程原因，不把失败组替换为历史P1。F2.2需要新decision。
