# formal3 目标检测标注接入 Gate

状态：`TOP PASS / GRIPPER EXCLUDED_PENDING_REVIEW`。

## 输入

- CVAT 导出：`数据集/formal3/det/f3 0-29.zip`、`f3 30-59.zip`。
- 训练数据：`数据集/formal3/kind_merged_nomaster`，60 episodes / 35,917 frames，640×480。
- 导出格式：每个相机、每个 episode 一个嵌套 CVAT 1.1 `annotations.xml`，模式为 interpolation。

## 验收规则

- episode 0–59 各出现且只出现一次，CVAT task size 必须逐集等于 dataset length。
- 类别只能是 `cup`；每个启用相机每帧至多一个 active box。
- 框必须有限并满足 `0 <= x1 < x2 <= 640`、`0 <= y1 < y2 <= 480`。
- unknown/缺标不得静默作为完整背景监督；只有完整通过的相机才能接入损失。
- 标准化XML去除CVAT用户、邮箱、URL等元数据，只保留训练需要的轨迹框。

## 结果

- top：60/60集，35,917/35,917帧有唯一合法杯框，人工关键帧1,820，CVAT插值active框34,097，outside框0。Gate=`PASS`。
- gripper/wrist：60个任务存在，但ep4为0/599 active；另有ep42/43/44/45/46/47/49/50/51/55/57/58/59覆盖不完整。不能区分“杯子确实不可见”和“漏标”，Gate=`EXCLUDED_PENDING_REVIEW`。
- 当前ACTDet合同只启用top检测，因此仅将top安装到`kind_merged_nomaster/annotations/top/episode_000..059.xml`；gripper不进入检测损失。
- 跨两个导出包抽查ep0/4/29/30/42/59的首/中/末18帧，叠框与杯子同步，无可见episode或帧偏移。证据：`top_overlay_contact_sheet.jpg`。
- 真实8D批次两步联合smoke通过：frame0/frame200的FCOS正样本数49/42，det cls/reg/ctr均有限非零；相位头、backbone梯度及无标签部署输出同时通过。证据：`../formal3_phase_aux_p2_20260927/nomaster_detection_integration_results.json`。

## 限制

- CVAT interpolation不等于每帧人工标注；只有1,820个top关键帧是人工设置，其余框由CVAT插值。
- 本Gate证明数据、映射和训练接口可用，不证明检测精度或闭环抓取成功。
- 正式P3前仍需冻结互斥train/dev划分，并只用train episodes生成归一化统计。
