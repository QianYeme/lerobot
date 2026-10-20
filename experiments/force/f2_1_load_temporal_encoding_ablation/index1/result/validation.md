# 最终实际验证

准备阶段4项本地pytest及真实GPU batch训练/反向/优化更新、strict reload、6D VAE、current额外键不影响输出、公共初始化hash相同的证据保留。历史P_hist完整5748帧主指标复现与原差1.58e-6<容差1e-4；原配置解析错误及后续通过分别留档。

完成后审核三个有效尝试的exit=0/complete=100k/CSV100k一致；每组五个20k保存点完整。15份评估均5748帧/固定12 Dev episode；逐episode按帧加权复算总主指标一致；selected.json确为五点最小、平局较早。3003条训练日志四项数值有限。

每个选中A1/A2的zero/plus/minus机制记录均覆盖5748帧；导出CSV的汇总按样本数而非batch等权，现batch4恰好整除。合法horizon原评估保留并屏蔽padding。selected checkpoint重新strict加载，state6和load_steps=0/1/15核验，normalization.json与Train-only冻结文件逐字节相同；batch1预测1×100×6全部有限。

5090成本实际测量warmup10+50重复；首次版本显存peak包含cuDNN warmup，保留旧JSON后将统计窗口改为warmup后的稳态，不改模型或主指标。未实测4060、摄像头或真机。新脚本语法/diff及原4项单元测试在本轮重新执行；报告不将工程完成等同真实力控有效。
