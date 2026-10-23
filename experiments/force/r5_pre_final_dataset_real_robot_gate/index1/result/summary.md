# R5部署准备与发布完成，真机尚未运行

日期：2026-10-23。本轮只完成模型/配套文件发布、直接运行入口、指南和离线工程验证，**R5_COMPLETE=false，真机0/15**；不是R5科学结果。

Hugging Face：[QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101](https://huggingface.co/QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101)，固定发布revision `6990fb2e82c39a571671fd22f7f45979f9d9ade6`。远端11文件的大小及LFS SHA256/Git blob SHA1核验通过，详见[hf_publish_receipt.json](hf_publish_receipt.json)。上传从训练服务器发起，凭据不写入文件或日志。

- 模型包`R5_deploy_bundle.tar.gz`：962,550,207字节，含3global、2local、Train36/Full60各自统计、prototype及离线DINO/ResNet缓存。
- 配套代码包`R5_project_files.tar.gz`：929,487字节。也可直接使用GitHub det更新代码。真机不需要完整formal4数据集。
- 冻结模型M1 Strong-G-36-S3、M2 Strong-G-60-S3、M3 New-GL-36、M4 New-GL-60、M5 G60 seed4。M5根据R4综合误差/空间诊断在真机前固定，理由见[bad_seed_selection](../bad_seed_selection.md)。所有精确身份见[model_identity.json](model_identity.json)。

新增`control_r5.py`和R5协议/适配器，复用旧F5直接运行循环。保持R4的benchmark=False、TF32=True；保留CLI可调ensemble/relative limit，不增加Stage0/shadow/审批文件或STOP锁。旧F5默认行为由回归保护，未把它的benchmark开关改掉。

工程中发现并修正一个真实输入一致性问题：旧在线load归一化用Python float64中间计算再转float32，与R4缓存的float32中间计算差约3.725e-9；同样权重对这个差异产生约0.004738 native动作差。相同输入、两模型实例输出差0，参数逐元素相等，因此没有改权重或放宽0.001门槛。只在R5覆盖prepare以精确匹配R4 float32运算。相关失败日志/差分probe保留服务器。

最终五路径真实RGB batch1检查全部通过：图像/state与参考逐位一致，load使用相同float32公式，nominal动作最大native差0，reset重现首步，current不参与策略；M3/M4与冻结local参考组成一致，arm隔离且残差在原界限内。`runtime_smoke.json`的`load_normalization_max_difference`记录的是**被拒绝的旧float64路径与R4的差值**，不是当前R5仍存在的输入误差，其reason字段有明确说明。

本地15项F5/R5 CLI、指南五指令解析、参数透传、R5 trial/P阶段、fake-I/O中止/清理/归档、load float32及汇总缺项回归通过。离线首步/重置不是目标主机或闭环真机验收。

下一步由用户在连接机械臂的Linux系统按[真机指南](../REAL_ROBOT_TEST_GUIDE.md)执行A/B/C固定顺序15次；完成后交回session、视频、P0/P1等人工判断与辅助记录。先不开始120–150段最终采集、不重训、不调整模型或在失败后追加救援重复。所有物理急停/相对限幅/残差界限保留。
