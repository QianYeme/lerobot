# formal3 四阶段标签导出合同

状态：PASS

## 问题与范围

本轮只回答：能否从 `formal3/kind_merged` 已录制的夹爪指令、`gripper.load` 和 `gripper.curr` 中，确定性导出可审计的四阶段候选标签。

本轮不修改模型、不启动训练、不操作真机，也不把自动标签声明为人工真值或完整力控实现。

## 数据合同

- 数据集：`数据集/formal3/kind_merged`
- 期望：60 episodes、35,917 frames、30 FPS
- 状态顺序以 `meta/info.json` 为准，必须包含 `gripper.pos`、`gripper.load`、`gripper.curr`
- 动作必须包含 `gripper.pos`
- 原始 parquet、视频和元数据只读，不原地写入

## 训练标签语义

- `approach`：episode 起点至闭爪动作开始前
- `grasp`：闭爪开始至人工确认的实际抬升开始前，包含整个闭合过程和可能存在的稳定保持
- `lift_place`：实际抬升开始至释放动作开始前
- `release`：开始张开至复位闭合确认前
- `unknown`：复位段、坏序列或无法可靠判定的范围

这些标签由遥测规则自动产生，默认 `reviewed=false`，不是人工真值。

## 审核事件与自动建议规则

- 夹爪张开：action `gripper.pos >= 30`，持续 3 帧
- 夹爪闭合：action `gripper.pos <= 25`，持续 3 帧
- 顺序：首次张开 → 抓取闭合 → 释放张开 → 复位闭合
- 闭爪区间：每集实际 `gripper.pos` 从本次开位到闭位相对行程的 10%–90%，持续 3 帧确认
- 释放区间：每集实际 `gripper.pos` 从本次闭位到开位相对行程的 10%–90%，持续 3 帧确认
- 抬升开始：自动候选暂取闭爪区间结束后一帧；必须由人工根据机械臂实际开始抬升/搬运的画面审核或修正
- load 基线：首次张开前最多 60 帧的 `abs(load)` 中位数
- 接触：`abs(load) >= baseline + 40`，持续 3 帧
- 实际闭合：state `gripper.pos <= 25`，用于核对夹爪是否跟随指令
- load 接触只作为诊断证据；不得单独否决命令顺序完整的阶段标签

人工审核对象是三个动作事件，而不是强行选择闭爪过程中的唯一一帧：闭爪区间、抬升开始、释放区间。五个边界必须非递减；模糊或冲突范围保持 `unknown`。

## 产物

- `phase_labels.parquet`：逐帧标签、phase_id、valid、reviewed
- `phase_annotations.json`：逐 episode 事件、区间、警告和来源
- `validation.json`：完整性、覆盖率、异常清单和 Gate 状态

## PASS 条件

- 60 episodes / 35,917 frames 一一覆盖，无重复、无越界、无重叠
- 四个事件对正常 episode 保持严格顺序
- 命令事件缺失或乱序必须降级为 `unknown`，不能静默补齐
- ep6 的 load 延迟只保留诊断提示，不得误报为坏 episode
- 审核脚本保存三个事件的边界和备注；用户明确确认当前片段后，episode 级 `reviewed=true`。任何后续编辑都会撤销确认；脚本必须拒绝越界或乱序边界。
- 单元测试、真实数据 smoke 和 `git diff --check` 通过

60 个 episode 的人工边界审核、JSON/CSV一致性检查和最终逐帧冻结均已完成。最终证据位于 `final_reviewed_labels/`，状态 `PASS`、`training_ready=true`；这只授权进入训练合同设计和真实批次 smoke，不等于已经授权高成本训练。
