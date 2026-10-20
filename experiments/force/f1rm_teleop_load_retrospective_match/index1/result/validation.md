# 完成验证

1. 准备阶段10段1500帧结构/有限性/连续帧与时间验证、两相机3000帧解码、10段30FPS审核视频验证通过；本次重验源文件SHA256一致及state/action名字与顺序。
2. 用户15条区间覆盖10试次：整数端点、范围、无重叠、条件兼容、非空审核人/依据检查通过。原CSV与result快照字节相同，label_freeze.json记录SHA256；UNKNOWN保持原样。reason均为1，语法通过不等于视觉标签已独立验真。
3. 7项定向pytest通过：历史命令无未来泄漏、力列不入特征、greedy阈值/不放回/并列顺序、SMD退化、UNKNOWN/条件与重叠校验、cluster bootstrap确定性。
4. report_f1rm.py独立读取原始parquet复算：scaler拟合集合、均值/标准差、136对标签、唯一性、全部距离≤0.75、load/current差值、池均值和四个SMD一致；analysis_validation.json为PASS。不重新匹配、不另选阈值。
5. 1.00敏感性未运行是冻结条件（仅覆盖不足）未触发；其CSV表头不代表已有结果。136对来源唯一，5个CUP试次均列出；ep001缺可靠CONTACT不填0效应。
6. 8张图生成，平衡图已人工视觉检查；原页面9项浏览器测试保持为准备阶段证据，与科学门分开。

工程完成PASS；覆盖PASS；平衡FAIL；科学UNRESOLVED_RESIDUAL_CONFOUND。F1RM_COMPLETE=true只表示授权分析交付完整，绝不等于有用信号确认。
