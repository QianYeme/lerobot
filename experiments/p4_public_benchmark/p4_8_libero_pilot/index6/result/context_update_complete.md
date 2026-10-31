## 2026-10-31 P4.8-r6对象路径机制诊断完成：GATE_SUPPRESSION_SUPPORTED

- r6独立归档index6/decision.md；真实核验r5 ACT/Legacy/Fixed六个5k/10k checkpoint及config/split/bank/source SHA。锁定Dev27+Train9+原型参考4共40帧，未Final；无optimizer更新、模型权重保存或新rollout，r5资产SHA前后保持一致。
- D0全160条gate0与物理绕过全chunk差0，重复fp32差0；54个r5重合锚点h0 ON/OFF精确复现误差0。当前训练模式本来就是fp32，未伪称多精度对照或发现旧测量错误。
- 10k Dev27中位query L2约234.05/238.33，raw residual53.29/46.14，gated/query比1.56e-6/8.06e-8。临时alpha=.01使h0 arm raw差从7.46e-8/1.90e-8到1.02e-4/7.72e-5，说明通路存在并被正常gate抑制；未证明动作更好。所有强制alpha为离线诊断/OOD，没有物理执行。
- 等价initial完整hash通过；alpha0使adapter/attention/slot梯度为0，alpha/CNN仍非零。末期强制.01显著放大内部梯度；5k/10k参数和Adam确有变化，不能称完全没训练，也没有完整gate历史曲线。
- Fixed参考帧ROI两格权重约14.8–18.0%，Legacy约.85–1.14%，仅建库图自一致性。已查看8张参考图及Train/Dev不同阶段热图，存在目标拿起后响应仍留原物体群的语义疑点，无GT定位准确率。只换对象Top几乎不变，换主干Top反应明显，不等于主干正确跟踪目标。bank槽位打乱本来就是max对称不变。
- D5复用原30视频/轨迹，固定六时刻抽样并查看三组总览；失败组部分画面疑似未持稳目标而继续搬运，准确接触/失效帧UNKNOWN，不由success flag推断。原r5三者4/10保持原样。
- 合同960条dose、168条梯度、160条信号/热图数值索引、192条依赖探针、24条参数delta等完成；补全离散索引normalized/raw及机械臂/夹爪全chunk分项max。短测并行诊断约1.95倍；无正在训练任务受干扰。
- 唯一主状态GATE_SUPPRESSION_SUPPORTED是模型内直接机制，不是小gate历史成因的唯一归因；初始化约束和语义/主干捷径均为并存解释。r7仅提出单变量非零gate初值候选，未执行，不改LR/bank/结构/预算。独立Builder/Red-Team PENDING。
- [r6交付摘要](../experiments/p4_public_benchmark/p4_8_libero_pilot/index6/result/summary.md)。后续等待用户与规划助手审阅，不自动进入r7、发布HF/Git或真机。
