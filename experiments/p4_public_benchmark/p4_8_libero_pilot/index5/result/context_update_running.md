## 2026-10-31 P4.8-r5 formal6方法迁移：三路正式训练运行中

- r5独立归档p4_public_benchmark/p4_8_libero_pilot/index5/decision.md，r0–r3原证据和原ZIP不改；服务器未发现index4/r4作业，无P4.7运行任务被更改。
- 固定Train35四分位episode813/908/1074/1166，25%帧38/48/37/37，Codex实际查看拼图及四张放大图后，各指定目标罐内点(192,120)。按256×256相对尺度rx14/ry18.6667命中每帧2patch；4×384 bank重复提取差0，SHA15dbf9323f1b363b5ad2dac1c358069ad8e936f7c1a21150e9d5552fdca4f3b4。G2_PROTOTYPE_PASS，明确包含执行者四点先验，不称无监督。
- 新核查确认数据RGB已是转换后方向，训练不再flip；仿真raw只经官方processor旋转一次。旧r3额外flip仅涉及已丢弃smoke，未热改历史。5476帧Train缓存18个视频样本与TorchCodec像素差0。
- 找到固定commit的OpenPI转换示例（fps10但逐条RLDS写入）与OpenVLA逐动作回放/180度转换说明；冻结原生序号→20Hz单step、chunk100/执行100、无ensemble、不插值不重复。TIMEBASE_PASS_NATIVE_STEP_PROTOCOL，仍披露精确发布转换commit未知，不把编码秒数等同物理秒数。
- ACT/Legacy/Fixed真实100步及完整optimizer/RNG下一步恢复差0；三者主干初始SHA与CNN输入/特征相同，两P1完整初始化SHA相同；在线DINO双/单标准化实际张量核验通过。三模型都从新初始化训练，不使用SO101策略权重。
- 两次实测1/2/3进程：三路总完成速度约为串行2.37/2.09倍，100步最终权重跨并发配置逐字节相同；选择triple。nohup控制器132229，训练worker133321/133322/133323，三组各10k。00:07快照ACT4400、两P1各3800，GPU7230MiB/99%，数据盘约23GiB空闲。
- 后台队列已包含三组10k→统一全部1391帧Dev9→每模型init10–19/seed1的10次原生仿真→54点对象ON/OFF诊断→孤立推理延迟/参数量→校验与自动ZIP。当前不是训练/评估完成结论，不填造成功率。Dev9额外分片网络超时保留日志，修复只涉及下载，不改数据。
- 工作区workspace/scripts与ObjectConditioning只读快照隔离；独立Builder/Red-Team仍待最终交付。无Final15、真机、额外seed/task/第四模型或HF/Git发布。
