# 泄漏审计

用户通过仅显示RGB的页面产生15条标注，在任何本轮load/current效应之前提交。先验证区间/标签/试次，保存原CSV及SHA256、固定匹配配置和脚本快照；未修改用户UNKNOWN或边界。来源声明为USER_SUBMITTED_RGB_REVIEW，不声称代码可证明人的盲法遵守或边界准确。

仅读取Formal5 kind1/kind2内容和父F1R清单。未读取原Formal4 Dev12/Final12样本内容。四变量由pos[t]、pos[t]-pos[t-1]、action[t-1]、action[t-1]-action[t-2]组成，不含force/current、绝对进度、ID特征或未来命令。ID只作确定性遍历。

标准化拟合719帧，仅FREE闭合和PRECONTACT；CONTACT不参与。UNKNOWN、未标注、前两帧排除。每一pass一次固定greedy，不放回；匹配结束后才读取被配对load/current并计算残差。主覆盖通过因此未执行1.00，不用敏感性替代主结论。

接触起点诊断固定15帧前后，不调标签、不搜索窗口；current在完全相同配对上次要报告。bootstrap按试次重采样而非帧。两阶段源码/标签哈希和验证文件可追溯；未训练、部署、重采、操作机器人或推送。
