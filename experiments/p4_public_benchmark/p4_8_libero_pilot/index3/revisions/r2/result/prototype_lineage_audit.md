# P4.8-r2 Prototype lineage audit

结论：**G2_PROTOTYPE_PROVENANCE_BLOCKED（路径 B）**。源代码可恢复，但原型生成依赖人工选定的目标位置；“没有逐帧框标签”并不等于“没有目标先验”。

## 可复核来源

- 构建函数：`scripts/p4_1_pilot.py:99–126`，完整源码和 SHA 位于 `source_audit_raw.json`。源脚本先声明 reviewed visible cup interiors，再硬编码四个 `(episode, frame, center)`。
- 四组输入为 `(0,194,550,325)`、`(29,194,448,279)`、`(58,194,421,251)`、`(86,194,505,300)`。中心周围按原图横纵轴各 35px 的椭圆筛选 16×16 patch 网格，取均值后 L2 归一化。不是全帧均值，也不是无监督聚类。
- 保存的 provenance 同样明确 `visible cup inner ellipse radius35px, executor reviewed Train images`，与脚本一致。
- 原 bank：`/root/autodl-tmp/lerobot-main/experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result/formal6_train90_prototypes.pt`；SHA256 `787a7084f1a387c71caaae635618458ec620b750d258df5df52578b1e7bc272f`；shape `[4,384]`，四向量范数均 1。
- 原 Train90 清单核验通过：四个源 episode 均属 Train；清单及 provenance 哈希见 `historical_train_membership.json`。本轮未读取 Final15 文件、像素、状态或动作。
- 历史已保存 initial_config 确认 adapter=256、hidden=512、4 prototypes、5 tokens、tau=.20、gate_init=0；不是误用配置默认的其它 adapter 维度。
- 原型侧输入为原始 `[0,1]` RGB，由 FrozenDINOv2Small resize 到224再单次 ImageNet normalize。完整模块源码也在 raw evidence 中。本地 `scripts/p4_2_1_repair.py:46–50` 的 assets() 按文件复制同一原型文件（本地源码单独保存于 repair_source_local_evidence.json；本轮服务器根目录未找到该脚本），不能解释为新增的无人工先验生成规则。

## 为什么不能原样迁移

当前 LIBERO 元数据没有可复用的目标 ROI/点位/Mask 字段（继承审计见 `inherited_index1/annotation_and_boundary_audit.json`）。杯子中心坐标、35px 尺度和四个源 episode 也不是 LIBERO 物体的有效标签。复制这些常数、复制透明杯 bank，或改成 KMeans/随机 patch/自动交互分割，都会改变被冻结的原型来源语义。

原始 ROI 的选择过程只有 executor-reviewed 来源说明，没有逐次人工点击记录；本轮不补造其历史。脚本、bank 和保存 provenance 三者足以证明存在目标先验，但不证明没有其它人为选择。

## Gate 后果

按 r2 第4节路径B及第6节第1项：不构造 LIBERO bank，不启动两组 P1 或 ACT 正式10k；仅登记三个 BLOCKED 作业。没有将 smoke、数学归一化计算或旧环境仿真当作模型验证。新的来源规则须由用户与规划助手批准后再执行。
