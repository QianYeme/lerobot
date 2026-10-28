# P4.0 审计交付

RESULT_READY=true
P4_0_EXECUTOR_AUDIT_COMPLETE=true
P4_0_COMPLETE=false（独立Builder/Red-Team待审）
OUTCOME=DATASET_BLOCKED
SPLIT_FREEZE_BLOCKED=true
TRAINING_RUN=false
ROBOT_RUN=false
DATASET_MODIFIED=false

数据revision：83a829af6df245ed53596dbc22b18c19f8bd3315。
120集93391帧，23文件齐全；双相机各93391帧完整解码。表格有限、逐集帧连续/时间单调、global index唯一，无完全相同state/action集。
逐集宽松概览QC：120通过粗动作链筛查（其中113有初始人手入镜备注），0明显失败；并非120集物理lift/hold精审全部通过，阶段物理高度/时长未定。无集被删除。全120集六列概览实际查看，非连续播放，不排除短暂异常。

原始state9D不直接符合P1的8D输入；可按名称取前8项、保留load/curr排除master，但本轮不擅自创建派生版本。采集单位/归一化实际配置无独立证据，因此P1_8D_6D_COMPATIBLE=UNKNOWN。按合同优先级先赋DATASET_BLOCKED，不代表数据损坏或不能训练。
区域标签已由用户取消；实际机械臂左抓右放，区域约10.5×13.5/6.5×6.5cm。近似起始圆检测103/120，非GT。近重复与场次依据不足，split仅候选90/15/15，ep113优先Dev，无正式冻结文件。

下一步最小补齐：确认采集位置单位/标定配置；批准明确的去master派生/输入映射及候选分组划分。无需为15区域补标签或重采整套。不得直接启动P4.1训练。
