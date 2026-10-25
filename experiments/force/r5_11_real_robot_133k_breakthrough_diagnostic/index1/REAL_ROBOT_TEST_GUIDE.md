# R5.11 真机指南：历史133k模型突破诊断

依据本目录decision及`docc/指南/真机测试经验与代码指令参考.md`：人工复位后直接运行，无数值Home文件/容差、Stage0、shadow、代码指纹审批或持久STOP锁。云服务器仅导出、核验和上传模型；以下真机命令在连接串口和相机的Linux主机执行。发布准备不是完成真机实验。

## 1. 四模型与本次上传范围

|试次ID|模型身份|模型包|输入/输出|
|---|---|---|---|
|R0|P1-Hist，历史P1/M0|已有F5/models/P1-Hist|8D state（6关节+load+current），原预处理器，6D动作|
|R1|Train36 New-GL，F4.2 preferred Strong-Global GL|已有F5 Strong-G+local/New-GL.pt|6D state+独立单帧load，原Train36归一化+原local，6D动作|
|R2|R5.10 M1，Full60 clip10/seed3/80k|已有R5_10/models/M1|6D state+单帧load，Full60归一化，无local，6D动作|
|R3|原R5.3 Full60/seed3/133000|本次R5_11/models/R3|6D state+单帧load，Full60归一化，无local，6D动作|

本次只上传R3，未重新上传R0/R1/R2，更不包含clip5/SWA。R3权重SHA256：`d094de659f58df3e55cd6284a83729cfd8658fb1d4b19f73d6a8cb634a61fc0f`。它来自原R5.3历史训练轨迹，不是把R2继续训练到133k，不是新的确定性重训。因此本轮比较“现有133k是否更有用”，不能单独归因于步数。

R0原权重SHA：`0b0270f6a0c0d37a23f2a8f5a50be3cddd39f923f9159f4576ad39761c805594`。
R1 global SHA：`e122e0e6574cf9e3f1a655b90bdf0de210b769e05dce0663ac451b20aefbd098`；local SHA：`695d6fa89f3e823f6236dca38c8e684c461009d5b9c357cfd1cb9f0d8dffea81`。
R2的身份由R5.10冻结model_identity.json验证，不能用R5旧Full60/80k或R5.3重训80k替代。R0/R1不是干净的数据量因果对照，不把它们压成6D G-Load输入。

## 2. 下载新增模型及本轮代码

仓库：<https://huggingface.co/QYyyyyyyy/R5_11_FULL60_133K_SO101>。只有一个策略权重，另附该模型Full60 normalizer、prototype、DINO/ResNet离线推理缓存、源码归档、decision和本指南；无需完整formal4数据集。

沿用已经可控制SO101的conda环境，不盲目升级torch/CUDA。建议新目录，避免覆盖已有个人改动：

```bash
mkdir -p ~/lerobot_r5_11
cd ~/lerobot_r5_11
mkdir -p outputs/deploy/R5_11
hf download QYyyyyyyy/R5_11_FULL60_133K_SO101 --local-dir outputs/deploy/R5_11
tar -xzf outputs/deploy/R5_11/R5_11_project_files.tar.gz -C .
python -m pip install -e . --no-deps
python src/lerobot/scripts/control_r5_11.py --help
```

源码归档是本轮入口的完整匹配快照；若GitHub尚无此入口，用该归档。不要用旧F5/R5项目归档覆盖本轮代码。`checksums.json`及发布回执记录字节数/hash和精确revision，可用`hf download ... --revision <commit>`固定版本。镜像仅作为网络备用，仍需核对相同内容；旧镜像曾误报文件大小，勿关闭校验来掩盖不一致。

## 3. 三个参考模型复用已有文件，不需要重新上传

后面的命令预期这些目录：

```text
outputs/deploy/F5/model_identity.json
outputs/deploy/F5/models/P1-Hist/
outputs/deploy/F5/models/Strong-G/
outputs/deploy/F5/local/New-GL.pt
outputs/deploy/R5_10/model_identity.json
outputs/deploy/R5_10/models/M1/
outputs/deploy/R5_11/models/R3/
```

如果新目录下没有旧包，可把原主机已经核验的`outputs/deploy/F5`、`outputs/deploy/R5_10`完整复制或链接到这里。只缺少旧包时才下载既有HF发布：

```bash
# F5既有包包含P1-Hist和Train36 New-GL，不下载/解包旧源码。
hf download QYyyyyyyy/F5_UNIFIED_SO101 F5_deploy_bundle.tar.gz --local-dir outputs/deploy
tar -xzf outputs/deploy/F5_deploy_bundle.tar.gz -C outputs/deploy

# R2只需要既有R5.10 M1资产；其它clip5/SWA权重不必下载。
mkdir -p outputs/deploy/R5_10
hf download QYyyyyyyy/R5_10_TRANSFER_SO101 --local-dir outputs/deploy/R5_10 --include "models/M1/**" "normalization/**" "assets/**" "torch_cache/**" "model_identity.json" "manifest.json"
```

旧F5归档中模型文件被保留是原发布格式，不代表本轮重新上传或增加测试模型。三模型包的normalizer/prototype/cache需跟随各自包，不能手动共用Full60统计量替代Train36/P1处理器。

## 4. 设备与人工复位

- 从臂：SO101 follower，`/dev/ttyACM0`，`robot.id=nn`；主臂复位：`/dev/ttyACM1`，`teleop.id=nn`。
- 顶部相机top=4，腕部gripper=2，RGB640×480/30Hz，腕部MJPG。查看画面确认相机没有互换，保持原标定、杯型、半杯水、灯光、工作台与A杯位。
- 每次操作者手动复位到既有正常起姿，沿用同一夹爪开口习惯；不要求Home JSON，不自动运动至Home。明显摆错或设备异常人工标记INVALID_SETUP，保留证据，不计策略失败。
- 新入口在首个自主动作前记录joint state、gripper.pos、load及两张相机首帧，initial_state.valid=null，不宣称机器核验起姿。P1的8D state顺序与独立load/local差异写入配置。
- 保持插电、独显及同一性能模式。负载是电机遥测，不是牛顿力，也不单独证明夹稳。

复位时先结束策略释放设备，再用主臂缓慢调整；退出teleop后再运行策略，避免串口/相机双占：

```bash
python src/lerobot/scripts/lerobot_teleoperate.py \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=nn
```

碰撞、挤杯、倾倒、洒水、失控或通信异常立即人工停止，急停/断电随手可及。退出可能卸力，提前支撑杯子与机械臂。max_relative_target保留，不能在模型失败后单独去掉。

## 5. 本轮流程：先A，不能自动跑B/C

1. Stage A：只在A位依次R0→R1→R2→R3，每模型一次有效自主尝试，共4次。
2. 若R3明显差于R2：结束133k测试，不跑B/C。
3. 若R3明显更好，或两者含糊：仅追加一次R2、R3的A位配对确认，不自动重跑P1/New-GL。用唯一trial-id标记确认。
4. 只有R3的A位改善在确认后重现，才进入Stage B：B位R1→R2→R3，C位同顺序，共最多6次。不把“含糊”当改善。
5. 建议上限12次有效自主尝试（4+2+6），结论清楚即停止，不自动消耗预算。所有无效/中断/错误仍保留记录；不挑最好一次。

确认可在R2/R3命令把`--trial-kind=stage_a`改为`--trial-kind=confirmation`并加`--trial-id=R5_11_R2_A_CONFIRM_01`或R3对应ID。满足Stage B条件后将position改B/C、trial-kind改stage_b。入口只运行一次，不自动判比较结论或排队进入Stage B。

共同设置：30Hz、最多60秒、人工复位参考30秒、n_action_steps=1、temporal_ensemble_coeff=0.01、max_relative_target=5。后两项始终在CLI可人工修改；正式比较前统一冻结，不允许按模型事后各调一套。cudnn.benchmark=False、确定性推理路径及TF32沿用已验证部署实现。

Ctrl+C/Esc停止，右方向键或终端输入`end`回车正常提前结束；没有持久STOP锁。Ctrl+C仅是终止方式，不自动判任务失败。同名重测自动归档为_prev_时间戳，不覆盖旧证据。

## 6. 观察与交回材料

P0朝杯接近；P1合理抓取几何；P2闭爪尝试；P3物理抓持；P4杯离桌；P5保持；P6放下并释放。闭爪不等于抓稳，空抬不算抬杯，hold时长不凭一帧猜测。

每次填写stage_review.json的A阶段、validity及失败原因。有效场景填VALID；未知/遮挡保持UNKNOWN；明显起姿/设备错误写INVALID_SETUP并解释。失败标签：APPROACH_DIRECTION_ERROR、APPROACH_STALL、PREMATURE_RETREAT、GRASP_GEOMETRY_ERROR、CLOSE_TIMING_ERROR、NO_PHYSICAL_GRASP、LIFT_FAILURE、HOLD_FAILURE、PLACE_FAILURE、RUNTIME_FAILURE。

只有自主完成已明确不可能，才输入`assist`回车标记B阶段并允许人工辅助；标记后不再视作完整自主成功。记录辅助方式及后续阶段能力，不删失败试次。原始phase A/B是自主/辅助，和Stage A/B实验阶段不是同一含义。

CSV独立记录state/nominal/requested/sent、load/current、推理/processing时间及overrun；手机视频记录物理阶段。拍摄时口述“R5.11、R几、A/B/C、确认或首轮”，保留同名文件及全部session目录（包括_prev）。视频/CSV不强制帧行对齐。汇总mean/p95/max latency、overrun和设备错误；任务进度不能只用关节变化或load推断。

报告阶段完成次数/有效次数、辅助和无效数量，不作小样本总体成功率。离线133k MAE和发布时预测复现，只是训练内诊断/工程核验，不证明真机改善。完整结果仍需人工阶段评审与Builder/Red-Team审核，R5_11_COMPLETE保持false直到合同满足。

## 7. 每个模型的独立运行命令（Stage A）

以下四条按R0→R1→R2→R3运行；每次先人工复位、重新确认杯位。只连接设备的本地Linux执行。

### R0：P1-Hist历史参考

```bash
mkdir -p "$PWD/outputs/r5_11/session_01"
python src/lerobot/scripts/control_r5_11.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/r5_11/session_01" \
  --execute --model R0 --position=A --trial-kind=stage_a \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_11/session_01/R0_A_$(date +%Y%m%d_%H%M%S).log"
```

### R1：Train36 New-GL参考

```bash
mkdir -p "$PWD/outputs/r5_11/session_01"
python src/lerobot/scripts/control_r5_11.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/r5_11/session_01" \
  --execute --model R1 --position=A --trial-kind=stage_a \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_11/session_01/R1_A_$(date +%Y%m%d_%H%M%S).log"
```

### R2：Full60/80k，原R5.10 M1

```bash
mkdir -p "$PWD/outputs/r5_11/session_01"
python src/lerobot/scripts/control_r5_11.py \
  --bundle="$PWD/outputs/deploy/R5_10" \
  --session-dir="$PWD/outputs/r5_11/session_01" \
  --execute --model R2 --position=A --trial-kind=stage_a \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_11/session_01/R2_A_$(date +%Y%m%d_%H%M%S).log"
```

### R3：原R5.3 Full60/133k，本轮新增

```bash
mkdir -p "$PWD/outputs/r5_11/session_01"
python src/lerobot/scripts/control_r5_11.py \
  --bundle="$PWD/outputs/deploy/R5_11" \
  --session-dir="$PWD/outputs/r5_11/session_01" \
  --execute --model R3 --position=A --trial-kind=stage_a \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5_11/session_01/R3_A_$(date +%Y%m%d_%H%M%S).log"
```
