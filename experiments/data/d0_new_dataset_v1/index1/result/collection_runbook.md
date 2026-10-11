# D0 New Dataset V1 — Collection Runbook

## Current readiness

`READY_FOR_HARDWARE_PREFLIGHT`。当前计算机未检测到机械臂串口或摄像头，因此不要运行正式录制命令。

## Frozen collection order

使用：

- `results/experiments/D0_NEW_DATASET_V1/collection_schedule.csv`
- `results/experiments/D0_NEW_DATASET_V1/collection_schedule.json`

顺序由 seed `20261008` 固定。五轮中每轮覆盖十二个 stratum 一次；不要改成同一 stratum 连采五次。`actual_cx_px/actual_cy_px` 在采后离线估计，采集现场不凭目测填造。

## Before recording

1. 固定机器人底座、两路相机、桌面、杯型、水位和 Home pose。
2. 用低显著度四角边界标出约 `40 × 30 mm` nominal region，不画内部网格。
3. 确认 leader/follower 串口及 top/gripper 相机路径。
4. 运行相机与串口发现工具，并保存输出到 raw evidence。
5. 先录制一个单独的 smoke dataset；确认 schema 包含：
   - `gripper.pos`
   - `gripper.load`
   - `gripper.curr`
   - 两路相机
   - 六维 action
6. 验证保存 action 等于 `robot.send_action()` 返回的实际发送目标，而非限幅前 leader 指令。

## Smoke command template

将尖括号替换为现场实际设备路径。smoke 与正式数据必须使用不同目录。

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<FOLLOWER_ID> \
  --robot.cameras='{top: {type: opencv, index_or_path: <TOP_CAMERA>, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: <WRIST_CAMERA>, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=<LEADER_ID> \
  --dataset.repo_id=local/d0_new_dataset_v1_smoke \
  --dataset.root=<ABSOLUTE_SMOKE_ROOT> \
  --dataset.single_task="Pick up, hold, lower, place, release, and return home with a half-filled transparent cup" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=45 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.vcodec=h264 \
  --dataset.encoder_threads=2 \
  --display_data=true
```

正式录制必须在 smoke 的 schema、相机语义、帧率、telemetry 和 action 语义全部通过后才开始。正式目录一旦开始，不中途改变硬件、分辨率、FPS、Home、phase 顺序或 nominal region。

## Per-episode operation

按 schedule 当前行摆杯，然后严格执行：

```text
Home/Open 0.3–0.5 s
→ Reach
→ Fine Alignment
→ Pre-Grasp 0.2–0.3 s
→ Grasp
→ Grasp Stabilization 0.5–0.8 s
→ Lift ~60 mm
→ Hold ~1.0 s
→ Lower
→ Place Stabilization 0.3–0.5 s
→ Release
→ Release Stabilization 0.3–0.5 s
→ Retreat 50–80 mm
→ Return Home/Open 0.3–0.5 s
```

从 Grasp Stabilization 到 Place Stabilization，leader gripper target 必须基本恒定闭合。不要为了“救漂亮数据”删除 TASK_FAILURE。录制后在 schedule 中填写 `VALID/HARDWARE_INVALID/OPERATOR_INVALID/TASK_FAILURE`、阶段结果和备注。

## Stop conditions

- 通信中断、相机丢帧或画面语义错误；
- 机械臂碰撞、失控或异常电流；
- 杯子倾倒或洒水产生安全风险；
- 数据 schema 缺少 load/current；
- 实际发送 action 未被保存；
- 采集协议中途发生未经批准的变化。

停止后保留现场和已有 raw evidence，不覆盖或删除失败 episode。
