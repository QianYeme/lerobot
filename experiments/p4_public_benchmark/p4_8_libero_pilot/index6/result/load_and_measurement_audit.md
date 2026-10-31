# D0 load and measurement audit

PASS：upstream_manifest.json逐一记录原r5 ACT/Legacy/Fixed的5k/10k六个真实checkpoint、SHA、split、bank、stats及外部effective config SHA。checkpoint本身未嵌入完整config，未把外部文件伪称内嵌配置。P1的两个step均实际strict加载并完成前向；全部源文件/权重/原型在诊断后SHA不变。

固定输入为Dev9的27帧、Train35排序首/中/末三集的9帧，以及原bank4张参考图，共40个观察。input_lock.json保存episode/frame/cache索引、两相机RGB SHA、state SHA及gradient batch8来源。未读取Final或重建原型。

40输入×两个step×两个P1共160条测量：gate0与物理绕过整个ObjectConditioning.forward的全chunk最大差0；正常输出重复/显式fp32重复最大差0。54个与r5重合的10k Dev锚点h0 ON/OFF指标逐项复现误差0。未发现旧测量实现错误。

r5训练数值模式本就是float32、TF32off、AMPoff，与本轮显式fp32相同；这不是两种不同精度的对照，没有声称测试fp16/fp64策略。内部norm的float64累加仅用于统计梯度/权重差，不改变网络前向精度。

输入state/RGB/CNN features/DINO input与features/raw tokens/adapted object tokens/query/raw residual/encoder input和预测均做shape/dtype/finite检查。实际布局与F32信息见object_signal_scales.csv，不只依靠gate!=0判断使用强度。

D1强制alpha通过临时forward hook重算Q+alpha*R；D2同类hook使用临时可微标量。原gate参数值未写回，不保存权重。bank反事实只在诊断实例内暂换buffer并finally恢复；各checkpoint模型state哈希前后相同。
