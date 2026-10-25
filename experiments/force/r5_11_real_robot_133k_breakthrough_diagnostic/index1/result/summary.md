# R5.11 发布准备交付（真机未执行）

- HF：[仅原R5.3 133k策略及必需资产](https://huggingface.co/QYyyyyyyy/R5_11_FULL60_133K_SO101/tree/7a4c47a6331538426fc315d52d049acb810f200c)。本次未上传任何R0/R1/R2或clip5/SWA权重，原参考模型包保持不动。
- 精确原始checkpoint：`/root/autodl-tmp/R5_3_20261024/runs/G60-S3-133k/attempt1/checkpoints/133000/pretrained_model`；SHA256 `d094de659f58df3e55cd6284a83729cfd8658fb1d4b19f73d6a8cb634a61fc0f`。与R5.3历史训练完成回执、评估身份记录和实际文件三方一致，无重训/改权重。
- 原Full60 normalizer、prototype及backbone离线缓存一并发布。源码归档和指南随HF包发布，尚未执行本轮Git提交/推送，直接使用该匹配源码归档。
- 36个真实训练输入：normalized state/load/RGB与原tensor逐位一致；normalized chunk及原单位第0步最大预测差0，reset重复差0。不是held-out验证或真机效果。
- R0/R1/R2/R3四个新入口路由在服务器全部完成CUDA合成观测dry-run，无设备连接；P1保留8D和原预处理器，New-GL保留Train36 global/local，R2复用R5.10 M1，R3为原133k。四模型身份hash见model_identity_manifest.csv；本次发布仓库只包含R3。
- 本地24项控制/协议/CLI回归通过，Python编译与git diff --check通过。初次读取R5.3缓存时因旧字段名`normalized_chunk/native_chunk`与后续缓存命名不同报错；只修正读键后重跑，完整初始错误日志保留服务器release_initial_cache_key_error.log。未改变模型/数据/指标。
- 发布后按精确commit验证243文件的大小及LFS SHA256/Git blob SHA1；仅一份策略model.safetensors，backbone预训练缓存不是额外实验策略。
- [指南](../REAL_ROBOT_TEST_GUIDE.md)按指定参考编写，末尾四条独立指令，CLI显式可调ensemble=.01和relative_target=5；人工复位、不要求Home文件、不加Stage0/shadow/持久锁。
- 后续由用户在本地连接真机：A位R0→R1→R2→R3各一次；需要时仅配对确认R2/R3；只有确认133k真实改善才做B/C，建议上限12次有效尝试。视频与CSV独立，不按小样本计算总体成功率。
- PREPARATION_COMPLETE=true，ROBOT_TRIALS=0，R5_11_COMPLETE=false；Builder/Red-Team仍PENDING。未新训、未增加数据、未操控真机。
