# R5.10：三模型真机迁移验证指南

参考：`docc/指南/真机测试经验与代码指令参考.md`。直接运行，无Stage0、shadow、代码指纹审批或持久STOP锁；保留原有物理限制；按用户最新授权，每次人工复位，不要求Home文件或容差。云服务器只做打包/上传，真机命令在连接设备的Linux电脑运行。本轮尚未执行真机，不能声称已验证闭环表现。

## 1. 准确的模型身份

|本轮ID|冻结模型|离线100步arm MAE|角色|
|---|---|---:|---|
|M1|Full60 / G-Load / deterministic clip10 / seed3 /80k|8.1822|强单点对照|
|M2|Full60 / G-Load / deterministic clip5 / seed2 /80k|17.5421|clip5救回模型|
|M3|Full60 / G-Load / clip5 / seed2 /均匀平均60k与80k|14.1284|SWA迁移候选|
|M4|Full60 / G-Load / clip5 / seed3 /均匀平均60k与80k|10.8723|仅可选，不延误前三组|

全部是6D关节位置state＋独立单帧load输入、6D位置动作的New-G/G-Load，无主臂信号、无current策略输入、无New-GL局部调节器。M1不是旧R5的Train36 Strong-G。M3/M4是离线参数平均模型，运行时只加载一个策略，不是两个动作输出投票，也不是60k后再训练20k。

四组必须使用同一个部署入口、Full60 normalizer、prototype、相机、预处理与运行参数。runtime收到RGB uint8后CPU float32/255，再传GPU做ImageNet规范化；state/load/action按各自保存的Full60统计量，load减除/除法保持float32。不能改回float64舍入路径、互换相机或混用旧Train36统计量。模型包包含所需normalizer、prototype与DINO/ResNet离线缓存，推理无需完整formal4数据集。

## 2. 下载并使用发布代码

仓库：<https://huggingface.co/QYyyyyyyy/R5_10_TRANSFER_SO101>。同仓库包含模型文件、缓存、代码归档、这份指南、decision、身份/校验清单及Home检索说明。建议用新的干净项目目录，保留原来能控制机械臂的conda环境、标定与个人修改。

```bash
mkdir -p ~/lerobot_r5_10
cd ~/lerobot_r5_10
git clone --branch det https://github.com/QianYeme/lerobot.git .
mkdir -p outputs/deploy/R5_10
hf download QYyyyyyyy/R5_10_TRANSFER_SO101 --local-dir outputs/deploy/R5_10 --include "models/**" "normalization/**" "assets/**" "torch_cache/**" "model_identity.json" "manifest.json"
python -m pip install -e . --no-deps
python src/lerobot/scripts/control_r5_10.py --help
```

沿用已经可用的LeRobot/SO101环境，不为本轮盲目升级torch/CUDA。发布在服务器Python3.12、torch2.8.0+cu128验证，不代表其它系统已实测。`hf`命令不存在时，可用同环境的`huggingface_hub.snapshot_download`下载同一仓库；不要下载旧R5/F5包替代。如果直连受限再考虑镜像，镜像须有同一发布revision并通过文件校验；HF发布回执会给出精确commit，可用`--revision <commit>`固定下载。

根目录`checksums.json`记录发布文件的字节数与SHA256，可在首次下载后核对；正常启动只做所选模型/normalizer的数据一致性检查，不设代码指纹审批。不要在未保存本地修改的项目里直接解包覆盖。

## 3. 每次人工复位，直接启动

按用户明确授权（见`MANUAL_RESET_OVERRIDE.md`），每次由操作者手动将机械臂调整到统一起始点，再启动策略。**不需要数值化Home姿态、容差或Home JSON，也没有相应的启动检查。** 不自动将机械臂运动至Home，不要求额外Stage0/shadow或文件审批。

启动时仍在第一条自主动作前记录六关节位置、夹爪位置、load、两张相机首帧和模型/杯位，便于事后理解起姿差异。`initial_state.json`的status为`RECORDED_MANUAL_RESET`，valid为null，表示记录完成，不代表软件验证了物理起姿。试次有效性由操作者结合现场和视频人工填写。

HF原发布源码归档和指南仍包含旧Home要求，**不要解包旧源码覆盖本轮GitHub代码**。按第2节下载GitHub det分支代码，HF只下载模型和推理资产；模型权重无需重新下载训练或修改。

## 4. 接线、相机与人工复位

从臂：`so101_follower /dev/ttyACM0 robot.id=nn`；top=4、gripper=2，RGB640×480/30Hz，腕部MJPG。以实际画面确认没有接反，保持原训练视角、桌面、光照、杯型和水位。若必须改端口/id/相机，开测前统一改所有模型命令并记录，不能某模型失败后单独更换。标定id必须匹配当前机械臂，不能为让文件加载而随意改成另一个id。

主臂仅用于两次试验之间慢速人工复位；先结束策略释放串口，再运行teleop，结束teleop后才启动下一策略：

```bash
python src/lerobot/scripts/lerobot_teleoperate.py   --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn   --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=nn
```

云服务器不直接连接机械臂。碰撞、挤杯、倾倒、洒水、失控或通信异常立即人工停止；急停/断电随手可及。`robot.max_relative_target=5`沿用参考，退出可能卸力，提前安全支撑杯子与机械臂。load是电机负载信号，不能当已经标定的牛顿接触力。

## 5. 预算、杯位和固定运行参数

在`outputs/r5_10/session_01/positions.md`记录既有A/B/C杯位、相机设置、光照、水位、人工复位方式和一张三点位照片，不扩展工作区。固定顺序：

|位置|依次运行|
|---|---|
|A|M1→M2→M3|
|B|M1→M2→M3|
|C|M1→M2→M3|

共9次主比较。仅在含糊对比、人工确认起始无效或矛盾行为时，预先注明原因，最多3次定向确认；不自动重复每格。M4仅前三模型干净完成且时间允许时选择，不能延误主比较；可选M4和确认共享剩余计划预算，计划总数不超过12。所有无效与中断尝试也保留，预算不足需说明，不悄悄反复挑成功。

固定共同参数：action_steps=1、temporal_ensemble_coeff=0.01、relative_target=5、30Hz、每次最多60秒、30秒人工复位参考。它们在文末CLI中可人工修改，但正式比较必须开测前统一冻结，不能给不同模型事后调参。程序没有自动Home运动、自动电机标定、shadow或持久STOP锁。

首轮设置`export R510_POSITION=A`，依次复制末尾M1/M2/M3命令；再改B、C。每条命令只跑一次，不会自动进入下一试次。Ctrl+C/Esc停止，右方向键或终端输入`end`回车记`OPERATOR_END`；不会创建STOP锁。重跑同名ID会把旧目录移到`_prev_时间戳`保留，不删证据；允许重跑不等于科研预算允许无限追加。定向确认使用`--trial-kind=confirmation --trial-id=R5_10_CONFIRM_M2_B_01`等唯一ID并事前记录原因。

## 6. 七个阶段和辅助规则

本轮P编号按R5.10定义，不能沿用旧R5里P3=抬升的含义：

|阶段|实际观察|
|---|---|
|P0|起步朝当前杯位接近，而非固定中间点|
|P1|闭爪前杯子处于合理抓取几何区域|
|P2|发出闭爪/出现闭合尝试，不等于抓稳|
|P3|物理抓持确实建立|
|P4|杯子确实离开桌面，不能把空抬当抬升|
|P5|确实持杯保持，记录可见时长，不事后加新阈值|
|P6|完成受控放下与释放|

根据视频独立填PASS/FAIL/UNKNOWN。提前中断后没发生/没看到的阶段填UNKNOWN；程序正常退出不等于任务完成，Ctrl+C也不自动等于模型失败。只有七阶段自主完成且无辅助才能记自主完成。

自主明确不能完成后才允许辅助。输入`assist`回车先记录进入B辅助阶段；辅助后完成不能算自主成功。需要进入工作区时先安全停止运动，不把手伸入运动区域。记录首个自主失败阶段、辅助类型和B阶段后续表现，不丢弃辅助段。

手机视频开头口述/拍纸条标识`模型ID/位置/TRIAL_ID`。视频判物理语义，CSV判命令/状态/运行时间，只在试次身份层面关联，不强行做逐帧时间同步。

## 7. 输出与交回材料

每次目录`outputs/r5_10/session_01/R5_10_Mx_A或B或C_01/`包含：configuration.json、initial_state.json、首帧与两路视频、同ID CSV、summary.json、stage_review.json以及可能的错误/辅助事件。CSV记录state/nominal/requested/sent、load/current、推理耗时/overrun和自主A/辅助B；current只作遥测，不进入策略。配置保留人工复位方式、模型身份、输入顺序、CLI参数与本轮阶段语义。

试验后按视频填写stage_review.json：validity=VALID仅用于初始与场景有效的试次；模型抓取失败仍是有效的任务失败，不能改成setup无效。物理失败分类：APPROACH_DIRECTION_ERROR、GRASP_GEOMETRY_ERROR、CLOSE_TIMING_ERROR、NO_PHYSICAL_GRASP、LIFT_FAILURE、HOLD_FAILURE、PLACE_FAILURE、RUNTIME_FAILURE。填在`first_autonomous_failure_stage`并在notes解释。

```bash
python scripts/summarize_r5_10.py --session-dir outputs/r5_10/session_01
```

该脚本只汇总人工阶段与CSV运行统计，含所有归档/确认尝试，不自动判视频或挑“最好一次”。`*_reviewed.csv`保留每次身份、有效性、mean/p95/max latency、overrun和辅助；正式9格按预先计划和全部尝试记录解释。多次有效同格需注明，不能把confirm/归档混成3次主比较。报告x/有效次数，保留UNKNOWN/无效/未测数量，不作总体成功率或显著性结论。

交回整个session（含_prev）、手机视频、positions.md、人工复位说明及备注。只有3次/模型，离线排序和真机排序可能不同；当前包发布不代表真机迁移已通过。

## 8. 每个模型的直接运行命令

以下统一采用已有参数。**先手动调整到统一起始点**，确认相机和杯位后直接运行。

```bash
export R510_POSITION=A
```

### M1：clip10 / seed3 /80k，主对照

```bash
mkdir -p "$PWD/outputs/r5_10/session_01"
python src/lerobot/scripts/control_r5_10.py \
  --bundle="$PWD/outputs/deploy/R5_10" \
  --session-dir="$PWD/outputs/r5_10/session_01" \
  --execute --model M1 --position="${R510_POSITION:-A}" --trial-kind=primary \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_10/session_01/M1_${R510_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M2：clip5 / seed2 /80k

```bash
mkdir -p "$PWD/outputs/r5_10/session_01"
python src/lerobot/scripts/control_r5_10.py \
  --bundle="$PWD/outputs/deploy/R5_10" \
  --session-dir="$PWD/outputs/r5_10/session_01" \
  --execute --model M2 --position="${R510_POSITION:-A}" --trial-kind=primary \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_10/session_01/M2_${R510_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M3：clip5 / seed2 /SWA60+80

```bash
mkdir -p "$PWD/outputs/r5_10/session_01"
python src/lerobot/scripts/control_r5_10.py \
  --bundle="$PWD/outputs/deploy/R5_10" \
  --session-dir="$PWD/outputs/r5_10/session_01" \
  --execute --model M3 --position="${R510_POSITION:-A}" --trial-kind=primary \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_10/session_01/M3_${R510_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M4：仅可选：clip5 / seed3 /SWA60+80

```bash
mkdir -p "$PWD/outputs/r5_10/session_01"
python src/lerobot/scripts/control_r5_10.py \
  --bundle="$PWD/outputs/deploy/R5_10" \
  --session-dir="$PWD/outputs/r5_10/session_01" \
  --execute --model M4 --position="${R510_POSITION:-A}" --trial-kind=optional \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_10/session_01/M4_${R510_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```
