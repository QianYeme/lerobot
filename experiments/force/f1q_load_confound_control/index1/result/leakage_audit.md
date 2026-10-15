# 泄漏审计

仅冻结fit[0,2,4,6,7,9]和eval[10,19,29,38,49,57]。未读原Dev/Final。

匹配距离输入仅F1P S0列[0,4,8,12]（pos、dpos、command[t−1]、dcommand[t−1]）；没有load/current/预测/时间/进度/episode特征。标签只定义可靠候选类；帧号仅确定性遍历/同距解歧，不进入欧氏距离。

匹配标准化只拟合全部可靠pilot-fit W15目标；未用eval统计。候选按episode分别处理，逐对确认两类帧无重复。匹配后probe scaler未拟合，因覆盖不达标。不删除不合格episode，所有12集均列出。

沿用因果模板：obs≤t，命令≤t−1，未跨episode或填补历史；历史采集源码/固件/硬件曝光同步的原局限保留。
