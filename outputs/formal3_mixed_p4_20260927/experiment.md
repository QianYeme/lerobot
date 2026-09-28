# Formal3 四路线 P4 训练合同

状态：`RUNNING`。用户已于 2026-09-27 明确授权；正式 100k 四路并行训练已在服务器 `screen -S formal3_mixed_p4` 启动。

## 数据合同

- Hugging Face：`QYyyyyyyy/formal3_kind_merged_nomaster_fit48`
- 训练：固定 48 episodes；开发验证：固定 12 episodes；seed=1000。
- 训练统计只由 48 个训练 episodes 计算。
- state=8、action=6，移除部署时不存在的主臂信号。
- DET 只监督 top 相机；gripper 相机仍作为普通策略视觉输入。

## 四个模型

1. `DIFFUSION_BASE`：纯 Diffusion，horizon=16，10 inference steps。
2. `DET_BASE`：ACTDet 检测辅助基线，不向策略显式注入检测结果。
3. `DET_C1_STATE`：FCOS top-1 `[cx,cy,w,h,confidence,visible]` detach 后投影并加到 state token；无 dropout/noise。
4. `DET_PHASE_W010`：DET_BASE 加四分类阶段辅助头，weight=0.10；阶段预测不回灌动作解码器。

所有模型均为 batch=8、100k steps、seed=1000、AMP关闭，每20k保存。正式训练前必须通过四路并发真实数据2-step预检，并由用户把本文件状态改为 `AUTHORIZED`。

## 训练完成与评估

- 四路100k训练于2026-09-28完成，退出码均为0，五个保存节点齐全。
- 第一阶段评估固定dev12：`9,11,17,25,26,28,30,31,43,52,55,59`；四个模型族并行、每族顺序评估20k到100k。
- 该评估是teacher-forced轨迹诊断，不报告为仿真或真机闭环成功率。
