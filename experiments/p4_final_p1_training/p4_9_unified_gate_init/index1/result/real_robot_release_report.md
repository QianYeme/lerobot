# P4.9 Gate+ 真机测试报告补充

本次新增formal6 Legacy seed1 Gate+与Fixed seed3 Gate+，均训练至80k，仅Gate初值0→+.001。训练后实际gate随权重导出，不在部署时覆盖。

新模型真机结果：**未测试**。历史L1/F3结果保留，不能移作新模型结果。LIBERO模拟结果不是SO101真机成功率。

部署工程校验：两个导出包分别在真实Dev15首集的0/1/2帧，经过流式预处理、时序聚合和后处理，与源checkpoint输出逐项差异为0；重置重放差异为0。CLI帮助解析通过，无机器人连接、无Final读取。HF独立下载验证与revision见release/result/upload/upload_receipts.json，以实际receipt为准。

测试方法和每模型独立完整指令见../REAL_ROBOT_TEST_GUIDE.md。沿用nn、ACM0、gripper2/top4、640x480/30Hz、use_degrees=true；n_action_steps=1、temporal_ensemble_coeff=.01、max_relative_target=5在命令行人工可调。没有Home数值审核/额外shadow/文件锁门。

按Legacy seed1 Gate0 vs Gate+、Fixed seed3 Gate0 vs Gate+分别配对；跨seed不判断DINO修复因果。记录Reach/Grasp/Lift/Hold/Move/Release与人工中断原因。是否真正改善抓取、保持和搬运，留待有人监督真机验证。当前不填成功率、不宣布主模型替换。

发布最终状态：两仓库各613文件独立下载SHA及运行校验PASS；精确revision见release/result/upload/upload_receipts.json。真机仍未测试。
