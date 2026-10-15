# 验证

- PASS：原子集/标签SHA256对照冻结F1P版本；仅打开其12份W15特征缓存，未加载F1P模型预测或新probe成绩。
- PASS：配置在匹配前保存；仅4个S0当前通道，fit-only标准化；同集、无放回、距离≤0.75逐对复核。
- PASS：可靠目标展开沿用F1P，不重新生成/外推标签。
- NOT_MET：逐集≥20对、每组≥4有效集；实际fit0/eval0。
- NA：合格eval集合的平衡门；全部小样本的SMD单独描述，不升级结论。
- NOT_RUN：probe/scaler拟合、增量和bootstrap。匹配变量scaler与probe scaler是两回事。
- F1Q_COMPLETE=false：第13节probe完成条件无法达成；覆盖停止证据已交付。
