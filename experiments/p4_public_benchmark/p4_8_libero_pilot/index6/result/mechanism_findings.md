# P4.8-r6 mechanism findings

**唯一主状态：GATE_SUPPRESSION_SUPPORTED。** 这是对10k末对象路径被抑制的直接数值机制判断，不等于已经找到了训练为何形成小gate的唯一根因，也不证明增大gate能提高成功率。

## H1：门控前信号存在，乘积被压到很小

10k固定Dev27的中位数：

| 指标 | Legacy | Fixed |
|---|---:|---:|
| query L2 | 234.051 | 238.334 |
| raw CrossAttn residual L2 | 53.294 | 46.145 |
| gated residual/query L2比 | 1.556e-6 | 8.063e-8 |
| 正常h0 arm raw ON/OFF均差 | 7.465e-8 | 1.903e-8 |
| 临时alpha=.01的h0 arm raw差 | 1.015e-4 | 7.717e-5 |
| 临时alpha=.1的h0 arm raw差 | 1.014e-3 | 7.633e-4 |

完整p50/p90/max、机械臂/夹爪、全chunk及0/1/5/10/25/50/99索引均在CSV。raw单位是LIBERO OSC输入，不是关节角或实测物体位移。剂量响应显示通路存在且能改变动作；这些是离线OOD输出探针，未执行物理动作、不比较任务成功率。超ActionBox的输出另有标记。

因果链的模型内部分可写为：R=CrossAttn(Q,T,T)，E=Q+alpha*R，A=ACT(E)。此处R并不消失；实际alpha很小使E接近Q，输出接近OFF。do(alpha)改变了A，排除正常后续前向完全覆盖掉该分支这一解释。没有证明较大的A变化更接近正确动作。

## H2：初始梯度约束已验证，但不能说内部完全没训练

重建初始模型的完整hash与r5 initial hash相同。alpha=0时，两个版本adapter/cross-attention/slot梯度均为0，而alpha梯度分别约.02725/.01575，CNN梯度非零。这符合代数关系dL/dtheta_object=alpha*(dR/dtheta)^T*dL/dE；dL/dalpha不受这个前乘alpha抑制。

10k固定Train batch8：Legacy adapter/attention梯度L2约1.14e-6/2.13e-6；Fixed约7.28e-8/1.23e-7。临时alpha=.01后分别约1.69e-3/3.16e-3及1.76e-3/2.98e-3，CNN仍约1.5。比较同组实际/强制状态支持门控限制上游梯度；不同组参数数量不同，不能只凭跨组L2推断学习贡献。

D2使用原生CVAE posterior训练损失结构，保留梯度，但关闭Dropout（包括MHA内部dropout）、固定seed1；model.training只用于激活原生posterior分支。它不是推理prior误差，也不是原训练每批历史梯度。没有optimizer更新，没有在Dev调参。

已有Adam状态和5k/10k参数delta表明内部参数并非完全未动；Adam会自适应缩放小梯度。只拥有initial/5k/10k稀疏证据，不能画成完整gate训练曲线或推断全过程关闭。

## H3：预处理修复改善参考图匹配，但跟踪语义仍有风险

四张建库参考帧中，原ROI两格的global权重总质量：Legacy约.85–1.14%，Fixed约14.84–18.00%；两格均匀基准为.78125%。这是**同一建库图的自一致性sanity check**，不是独立定位准确率。ROI之外也包含目标其它部分，不能全部当背景。

实际查看全部8张参考热图，以及各版本首个Train/Dev episode的start/middle/end。Legacy整体分散；Fixed未抓取阶段在右侧罐头/邻近盒状物区域更强，但抓持/末尾仍有响应停留右侧原物体群，未清楚跟随已拿起、移至篮子侧的目标。遮挡和相邻物的上下文混淆是竞争性解释。没有像素GT，不报告盲测定位准确率，也不据此重选bank。

5k/10k同输入热图SHA一致，符合DINO和bank冻结。四个2x2区域的条件权重、全局权重、tokens xy/confidence及top5 patch位置均保存；固定区域产生背景token不等于分割负标签。

## H4/H5：主干替代路径敏感；对象语义与是否有益仍未证明

10k六个固定输入探针，正常gate下只换对象分支Top的h0 arm raw均差约3.08e-8/2.38e-8；同时换主干Top和对象Top、保持Wrist/State不变时约.07138/.07057。因此不能说整个策略视觉失效；主干Top输入能明显影响输出。图像与state混合本身是OOD，这不证明主干在正确追踪目标或足够泛化。

随机bank来自固定Train参考帧的随机非ROI patches，事后查看源patch在机器人/墙/桌面/边缘区域；DINO带全局上下文，不能保证特征完全无目标语义。没有重选随机样本。当前gate下random-bank影响也极小，强制门控下可见变化。

原型槽位打乱不变是cosine-max的数学对称性，**不是模型不用原型的证据**。

## H6和历史根因边界

测量复现、物理绕过及重复精度检查均通过，未发现测量假象。权重、bank、来源代码SHA均不变。

零初始化、随机adapter、主干视觉捷径、原型在持握/遮挡阶段的语义不足均可能参与形成小gate；小gate也可能是对低收益信号的适应。当前模型内干预和两个checkpoint不能区分这些历史训练成因，更不能说明把gate强行打开会改善r5的4/10。
