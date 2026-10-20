# F3 — TRANSFER_READY / REAL_ROBOT_NOT_RUN

按用户要求只准备和发布迁移文件，不执行任何真机阶段。F3_COMPLETE=false。

- HF私有仓库：https://huggingface.co/QYyyyyyyy/F3_C1_LOAD_SO101
- 固定revision：`4bab110b3149003239c330b096cd7d1883c32a78`
- 内容：A0 retry1 80k全局6D模型、C1-load10k局部MLP、归一化、
  原型bank、DINO/ResNet初始化缓存、源码快照、数值夹具及校验器。
- Global SHA：`65be80d1a70c178cffd29b8efed8ab08d923a79013eb56bdaa3475e69679fdcf`。
- 官方HF固定revision的manifest及关键三项模型/资产hash核验通过。
- 服务器CUDA离线strict-load、disabled/shadow/机械臂保持/残差限幅通过，
  与F2.3局部16条Dev夹具最大差3.5763e-7。只是离线工程校验，不是部署硬件Stage0。

真实串口/相机、实时日志、shadow3次、baseline3次、regulated3次均NOT_RUN。
现有control_act_det.py没有局部MLP接线，不能直接据此宣称regulated可运行。
适配器不含机器人I/O；迁移后必须先完成F3硬件接口、安全和遥测验证。

详见../TRANSFER_GUIDE.md；publication.json记录实际发布信息。
