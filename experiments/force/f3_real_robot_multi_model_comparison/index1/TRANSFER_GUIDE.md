# F3 r1 四模型迁移指南（暂不执行真机）

现场完整操作、安全门、历史参数及最后的四模型测试命令见同目录
[REAL_ROBOT_TEST_GUIDE.md](REAL_ROBOT_TEST_GUIDE.md)。统一入口为GitHub新增
`control_f3.py`，先阅读该指南第5节并完成现场审核，不要仅替换旧脚本模型路径。

私有HF仓库：`QYyyyyyyy/F3_R1_MULTI_MODEL_SO101`。原r0仓库
`QYyyyyyyy/F3_C1_LOAD_SO101`保留，仅对应本轮M1/M3，不包含全部四组。

| 本轮模型 | 确切来源 | 包内路径 |
|---|---|---|
| M0 | 历史P1_ocfix2_s1 100k，8D OC-ACT | models/M0 |
| M1 | F2.1 A0 retry1 80k，6D无反馈 | global |
| M2 | F2.1 A1 60k，6D＋全局单帧load | models/M2 |
| M3 | 与M1同一global＋F2.3 C1-load10k | global＋local/010000.pt |

M0不是纯ACT；其state顺序为五个机械臂关节.pos、gripper.pos、
gripper.load、gripper.curr。M1/M2/M3全局state仅前六项位置。
M2额外`observation.load`为Train归一化单帧[B,1]，不是第七个state。
M3全局不接收load/current；raw load、raw夹爪位置与native nominal动作
只送局部适配器，内部各归一化一次。不要把M0的curr带入M1/M2/M3。

## 下载与离线检查

```bash
# 本人账户认证后下载；不要在源码或日志内写token。
hf download QYyyyyyyy/F3_R1_MULTI_MODEL_SO101 --local-dir F3_R1
cd F3_R1
export TORCH_HOME="$PWD/torch_cache"
export PYTHONPATH="$PWD/source/src"
python source/scripts/verify_f3_bundle.py --bundle "$PWD" --device cuda
python source/scripts/verify_f3_multimodel_bundle.py --bundle "$PWD" --device cuda
```

先按environment_versions.json和源码依赖配置新环境。服务器验证的是
Python3.12/CUDA torch2.8，不能保证任意CPU/GPU配置的数值对照同样通过；
F2.3使用的TF32设置由校验器显式保留，新硬件仍需部署等价验证。
校验器不打开串口、相机或执行机器人动作，也不是实机Stage0的替代。

M0用自带policy_preprocessor/postprocessor及其safetensors；M1/M2/M3用
normalization.json与local_normalization.json的冻结Train统计。M1/M2
没有标准processor配置，不要默认通用record入口会自动补齐相同预处理。
相机语义均为top/gripper两路，但现场设备编号、标定必须新系统核验。

## 接线与共同参数仍待部署系统确认

冻结的HF包源码快照不含后续新增的统一入口。请从更新后的GitHub获取
`control_f3.py`及`f3_runtime.py`，其逐模型预处理、局部修正、逐步日志/两路
录像与审核门已有离线/假机器人验证；现场硬件和安全门仍须核验。
不要继续用旧control_act_det.py只换权重路径来运行四组。

全部模型要冻结同一控制频率、相机、硬件、temporal ensembling、限幅和平滑。
model_identity记录原checkpoint配置，但不等于现场公共参数已经冻结。
deployment_config.json中PENDING项必须现场补齐并在首个试次前固定，不能
看到结果后更改。M0历史差异不只load，不能把M0 vs M1当纯反馈消融。

M3残差上限保持±4.832699459791184原生位置单位，只有夹爪列可变。
disabled返回原动作；shadow只计算记录、实际执行M1 global-only动作。

## 后续实机顺序（本次不运行）

1. 四组身份和统一部署配置审核、离线/现场Stage0。
2. M3 shadow ×3，审核安全门，失败则禁止M3干预。
3. Round1：M0→M1→M2→M3；Round2、Round3相同，各组3次正式试次。
4. 若M3被shadow阻断，只保留每轮M0→M1→M2；不能重排或增加试次救结果。

每次reset模型/历史、恢复起始姿态和杯位，保留中止/失败、全步遥测与视频。
M3严重调节器相关危险事件一次就停止其后全部干预，不调参续跑。
不做Final12，不训练，不扩点位/杯型/任务，不自动进入下一轮。
计划15次（含3 shadow）≠已采集15次；本包没有任何真机结果。
