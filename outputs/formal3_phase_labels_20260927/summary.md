# formal3 动作区间标签导出总结

状态：`PARTIAL_PASS`；`training_ready=false`

## 结果

- 60 episodes、35,917 frames 全量导出，无缺帧或重复帧。
- v3 候选有效标签 29,407 帧；`unknown` 6,510 帧。
- 分布：approach 12,808；grasp 2,745；lift_place 8,788；release 5,066；unknown 6,510。
- 闭爪和释放由每集实际夹爪相对行程 10%–90%表示为区间；抬升边界必须人工根据视频确认。
- ep6 候选为闭爪 208–272、抬升 273、释放 395–422；391 帧 load 触发只保留为诊断。
- 所有标签均为 `reviewed=false`，复位闭合后的帧保持 `unknown`。

## Builder Review

- 脚本按合同从原始 parquet 只读导出，不修改数据集。
- 输出逐帧 parquet、逐 episode JSON 和 validation JSON。
- 3 项直接回归测试通过，`py_compile` 和真实 60 集 smoke 通过。
- pytest 在收集项目 `tests/conftest.py` 时因本地缺少 `pyserial` 阻断，未记为通过。

## Red-Team Review

- 自动遥测标签不是人工真值，不能直接报告为正式阶段监督。
- 固定阈值仅来自当前 formal3 数据合同；真机必须重新测空载基线和三态标定。
- load 受基线漂移、运动和释放影响，不能单独作为阶段真值。
- 自动抬升候选取闭爪结束后一帧，只是初始化，不是人工真值。
- 本轮没有实现相位条件化模型、HOLD 状态机或真机力控闭环。

## 下一 Gate

按新版三个事件重新审核 60 集，结果写入 `review_ui_v2_intervals/`。在全部审核完成前不得启动正式训练。
