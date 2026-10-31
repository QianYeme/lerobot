# r5 prototype geometry audit

G2_PROTOTYPE_PASS：Codex实际查看固定四帧拼图、全部3倍最近邻显示图及生成后的采样叠图；没有换帧、缩放采样半径或读取Dev/仿真目标真值。

Train35四分位episode813/908/1074/1166，帧38/48/37/37，均在native256×256 agentview图像中指定(192,120)。四帧目标位置接近，因此坐标相同；不是沿用formal6杯坐标。目标为位于橙色牛奶盒后方、夹爪下方的蓝色侧壁罐头，不是左侧红绿罐、篮子或机器人。

坐标x向右/y向下；训练/原型对已转换的数据图像不再flip。仿真raw图像仅由官方LiberoProcessor翻转两轴一次以对齐数据。旧r3的训练侧额外flip仅用于丢弃工程smoke，没有正式模型；r5纠正这个接入错误并重新验证，不改写历史证据。

严格采用rx=35*256/640=14、ry=35*256/480=18.6666667。DINO输入为原RGB0..1→F.interpolate224 bilinear align_corners=False→ImageNet单次标准化，不使用r3的cv2预缩放特征。patch中心回投256坐标，四帧均命中两个patch：中心(184,120)/(200,120)。绿色框是patch范围，绿色点是命中的中心，红点为Codex坐标，黄色轮廓是采样判据。圆/椭圆轮廓与完整patch并非目标轮廓，patch保留少量上下文；不声称每个像素都是目标真值。

4×384原型数值有限/单位范数，完整重复提取逐元素差0、文件字节一致，bank SHA见prototype_numeric_validation.json。相似原型不触发新增多样性阈值；本轮不以Dev表现重选bank。现有DINO缓存没有可靠git revision，明确标UNKNOWN，以权重SHA和源码树SHA冻结身份。

方法包含四个Codex辅助目标位置先验，用户无需打标，但不称完全无监督。独立Builder/Red-Team验收仍待交付后进行。
