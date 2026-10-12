# G1 真机位置泛化测试指南

## 1. 目的与冻结条件

本指南对应 `G1_POSITION_REAL_ROBOT_GENERALIZATION`，用于比较三个冻结模型在 Formal4 训练工作区内不同杯子位置上的真实任务泛化能力：

- `B3_ACT`：ACT baseline；
- `P1_OCACT`：object-aware 候选；
- `R3_FILM`：FiLM 条件化候选。

本轮禁止重新训练、替换 checkpoint、修改模型结构、修改相机/预处理/归一化、读取 Final Test 或人工纠正策略动作。人工只允许在安全风险出现时执行 `SAFETY_ABORT`。

## 2. 推荐部署方式

机械臂和两路相机应直接连接本地 Linux 控制主机。远程 GPU 服务器用于训练和离线评估；没有可靠 USB/串口转发和断线急停时，不要通过云服务器控制机械臂。

需要 SO-101 follower、固定位置的 top/wrist 相机、半杯水的透明塑料杯和可立即断电/停止动作的急停方式。机械臂运动范围内不得有人员、线缆和易损物。

进入项目并激活与训练一致的环境：

```bash
cd ~/lerobot/lerobot-main
source .venv/bin/activate       # 按实际环境修改
git status --short
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls -l /dev/ttyACM* /dev/video* 2>/dev/null || true
```

没有串口或没有两路相机时停止，不要启动策略。

## 3. 相机与机械臂预检

确认 top 画面覆盖桌面和杯子工作区，wrist/gripper 画面能看到夹爪和杯子接近区域；两路画面不能黑屏、冻结、错位或明显过曝。确认机械臂状态连续可读，关节角无跳变，夹爪可低速张开/闭合。

相机索引必须现场核验，不要直接假设旧编号。记录：

```text
deployment_host = <主机名>
robot_port      = <例如 /dev/ttyACM0>
top_camera      = <设备索引或路径>
wrist_camera    = <设备索引或路径>
resolution/FPS  = <实际值>
code_commit     = <git rev-parse HEAD>
```

把机械臂置于固定的 `Home + Open` 姿态；固定底座、相机位置、光照、桌面和杯子外观。

### 主臂控制从臂（仅用于预检/手动复位）

如果现场同时连接 leader 和 follower，可先用下面命令确认主从通信、关节方向和手动复位。这里必须补上 `--robot.id=nn`；反斜杠必须放在行尾，不能在反斜杠后继续写参数：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=nn \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=nn \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

该命令不是模型推理命令。进入策略推理前应停止 teleoperate，确保 leader 不再向 follower 发送动作；`control_act_det.py` 只连接 follower。leader 仅用于人工把机械臂安全放回 Home + Open，不能在正式 autonomous trial 中提示模型或修正动作。

## 4. 安全门

只有以下条件全部满足才能运行模型：

- 杯中约半杯水，桌面干燥；
- 机械臂为 Home + Open；
- 夹爪、底座、相机和线缆无碰撞风险；
- 操作者手能触及急停；
- `max_relative_target` 保持部署合同中的安全值；
- 日志和视频目录可写。

出现碰撞、持续发散、夹爪挤压杯子、杯子倾倒、通信异常、相机掉帧或无法立即停止时，立刻人工停止，标为 `F9_SAFETY_ABORT`，不要当作普通任务失败。

## 5. 场景、位置和复位

工作区固定为横向 P0 ±20 mm、纵向 P0 ±15 mm。每次 trial：

1. 机械臂回到 Home + Open；
2. 按 manifest 指定 stratum 自然随机放置杯子，不要总放在格子中心；
3. 保持约半杯水；
4. 检查两路画面；
5. 清空 temporal buffer/action queue；
6. 只允许策略完成 Reach → Grasp → Lift → Hold → Lower → Release/Place → Retreat。

正式计划是 `trial_manifest.json`：Phase A 为 9 次，Phase B 为 72 次。不得自行改模型顺序或试验数量。

## 6. Phase A：安全/功能检查

每个模型在中心位置运行 3 次，共 9 次。Phase A 不计入主成功率，只检查 checkpoint 加载、相机/状态/动作管线、Home 姿态和明显安全风险。

任一模型出现部署错误，先标记 `F10_DEPLOYMENT_ERROR` 并修复，不进入 Phase B。出现安全风险则停止整个测试。

## 7. Phase B：正式位置泛化

12 个 position strata，每个模型每个 stratum 2 次，共 72 次。执行顺序：

- Block 1：`B3_ACT → P1_OCACT → R3_FILM`
- Block 2：`R3_FILM → B3_ACT → P1_OCACT`
- Block 3：`P1_OCACT → R3_FILM → B3_ACT`

每次换模型必须停止上一会话、回 Home + Open、清空缓存并重新确认相机。不得继承上一 trial 的隐状态。

## 8. 现场记录

每个 trial 完成后立即填写：

```text
trial_id / model / block / stratum / repeat
realized_x_mm / realized_y_mm
Reach / Grasp / Lift / Hold / Place / Overall (0/1)
failure_type: F1-F10 或 NONE
center_bias_event (0/1)
mean_latency_ms / p95_latency_ms / dropped_frames
video_top / video_wrist / notes
```

杯子脱离桌面并稳定约 0.5 秒才算 Lift 成功；抓起后立即滑落为 `F5`，保持阶段滑落为 `F6`。Lift 是本轮主要指标，Overall 是次要指标。每个视频必须与 trial_id 一一对应。

未完成 81 次前，不得写入：

```text
POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE = TRUE
```

完成后更新 `trial_results.json`、`trial_results.csv`、`failure_analysis.json`、`position_breakdown.json`、`resource_metrics.json` 和 `summary.md`，并停在 G1，不自动进入 Force Track 或重新训练。

## 9. 真机推理命令

先在真机控制主机下载服务器上传的冻结 checkpoint。不要下载其它候选或重新训练模型：

```bash
hf auth login
mkdir -p models/g1
hf download QYyyyyyyy/G1_B3_ACT_100k --repo-type model --local-dir models/g1/B3_ACT
hf download QYyyyyyyy/G1_P1_OCACT_100k --repo-type model --local-dir models/g1/P1_OCACT
hf download QYyyyyyyy/G1_R3_FILM_100k --repo-type model --local-dir models/g1/R3_FILM
```

下载后确认每个目录至少包含 `config.json`、`model.safetensors`、`train_config.json`、`policy_preprocessor.json`、`policy_postprocessor.json` 及两个 processor safetensors 文件。下面命令一次只启动一个模型，并使用这些本地目录：

```bash
cd ~/lerobot/lerobot-main
source .venv/bin/activate
export ROBOT_PORT=/dev/ttyACM0
export TOP_CAM=4
export WRIST_CAM=2
export DATASET_ROOT=/home/lyj/lerobot/数据集/formal4
export LOG_DIR=experiments/position/g1_real_robot_generalization/index1/runtime_logs
mkdir -p "$LOG_DIR"
```

### B3_ACT

```bash
python src/lerobot/scripts/control_act_det.py \
  --policy.path=/ABSOLUTE/PATH/TO/models/g1/B3_ACT \
  --policy.n_action_steps=1 \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=g1_B3_ACT --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$LOG_DIR/B3_ACT_$(date +%Y%m%d_%H%M%S).log"
```

### P1_OCACT

```bash
python src/lerobot/scripts/control_act_det.py \
  --policy.path=/ABSOLUTE/PATH/TO/models/g1/P1_OCACT \
  --policy.n_action_steps=1 \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=g1_P1_OCACT --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$LOG_DIR/P1_OCACT_$(date +%Y%m%d_%H%M%S).log"
```

### R3_FILM

```bash
python src/lerobot/scripts/control_act_det.py \
  --policy.path=/ABSOLUTE/PATH/TO/models/g1/R3_FILM \
  --policy.n_action_steps=1 \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=g1_R3_FILM --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$LOG_DIR/R3_FILM_$(date +%Y%m%d_%H%M%S).log"
```

每次 trial 结束后按 `Ctrl+C` 停止进程，确认机械臂安全，再启动下一个模型。Ctrl+C 本身不算完成一次 trial；必须填写该 trial 的阶段结果、失败类型、位置和视频路径。

