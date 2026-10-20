# 实现差异

ACTConfig仅新增load_history_steps=0/1/15（默认0关闭），启用时要求6D state；ACT在所有公共参数初始化之后，以隔离CPU RNG/固定seed1建立load MLP和独立零初始化pos embedding。在state之后、image之前追加单token。A1/A2只有输入宽度不同，不进入VAE状态路径，无current/gate/FiLM/残差控制。

新训练runner显式从干净缓存读取6D位置，独立选择single/W15，使用共用Train统计和原图像处理。原录制/部署程序未修改。每组训练模式、预算、seed、数据加载生成器和checkpoint选择一致；公共state_dict hash已实际相同。原P_hist不warmstart。

模型默认关闭回归通过旧8D checkpoint strict加载和完整Dev指标复现。训练入口同时处理原action chunk mask，合法horizon排除padding，h0为主而不使用训练VAE随机latent作为评估预测。

checkpoint保存标准policy/config和f2_1_normalization.json；带load的checkpoint需要本轮runner相同输入适配与历史缓存，不可直接拿旧control_act_det命令上真机。部署适配/真机不在本轮授权。
