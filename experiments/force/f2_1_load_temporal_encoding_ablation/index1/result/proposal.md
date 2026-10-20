# 下一步建议（不自动执行）

本轮优先保留A1单帧load，冻结A1 60k选中checkpoint与其normalization/输入语义；A0 80k作干净基线，A2 40k保留反例与审查证据。不要改用各组100k或训练loss最低来改变结果。

先交规划方Builder/Red-Team验收，讨论A0原数值故障、单seed与Dev选择偏差、A2臂代价。若需强化结论，提出新合同决定匹配复验/其他编码，而不是在本轮加current、换W或门控。若后续要真机，须另立部署输入历史与安全门；旧control_act_det不能直接用于这些6D+load新模型。

不自动进入F2.2，不自动做机器人测试、Force head、局部控制、上传HF或push GitHub。Authorization=PENDING_NEXT_DECISION。
