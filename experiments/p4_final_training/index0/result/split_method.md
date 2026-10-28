# 候选划分

SPLIT_FREEZE_BLOCKED：缺可靠近重复分组/采集场次依据；不是因为缺R01–R15标签。
用户授权训练优先干净片段。已发现初始手入镜ep113固定Dev；其他119集用NumPy default_rng(20261028) permutation：前14 Dev、随后15 Final、剩余90 Train。规则在生成ID前固定，无模型误差筛选。split_manifest.csv仅提案；不生成正式train90/dev15/final15文件。
候选计数90/15/15，ID不重叠；尚不能保证近重复轨迹不跨集合。trajectory_nearest_neighbors.csv提供64点归一时间action原单位平均绝对距离排序，不是泄漏判决或物理距离。
手入镜并未声称只存在113集，稀疏概览可能漏短暂入镜。Dev偏向此扰动应单独解释；不把它作为Final成功率样本。所有候选Final内容本轮仅QC/划分使用，未做模型推理、原型或调参。
