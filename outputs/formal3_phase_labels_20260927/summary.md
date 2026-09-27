# formal3 动作区间标签导出总结

状态：`PASS`；`training_ready=true`

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

60 集人工审核已完成，最终逐帧标签位于 `final_reviewed_labels/`：35,917帧全部覆盖，其中有效监督27,235帧、unknown 8,682帧。下一步建立相位辅助 ACTDet 的单变量训练合同并先跑真实批次 smoke；高成本训练仍需单独授权。
