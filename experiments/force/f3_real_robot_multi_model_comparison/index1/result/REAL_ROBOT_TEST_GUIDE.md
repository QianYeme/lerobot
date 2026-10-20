# F3 r1 四模型真机对比测试指南

> 历史版本，已被`../REAL_ROBOT_TEST_GUIDE.md`替代。请勿执行本文旧命令。M3差异已复现为TF32及batch依赖，不再判定为权重错配；详见`OFFLINE_ACCEPTANCE.md`与`precision_batch1_audit.json`。现场安全门仍未通过。

对应 `decision_F3_REAL_ROBOT_MULTI_MODEL_COMPARISON_r1.md`。本轮是**探索性多模型真机对比**，不是 Final12，不读 Final、不训练、不调参、不扩展任务。

> 状态：`REAL_ROBOT_NOT_RUN`。本指南给出真机执行协议与命令模板，但**统一控制循环接线、M3 局部残差接线、状态选择、原型路径修正等部署前置仍 PENDING**（见 §3）。未完成这些前置并现场 Stage0 前，不得启动策略。

---

## 1. 目的与冻结范围

在同一场真机会话中比较四个存活模型家族，回答：**执行器反馈策略（无反馈 / 全局单帧 load 融合 / 局部 load 残差）在真实闭环抓取中哪个更有效**。

四个冻结模型（全部位于 `/home/lyj/lerobot/outputs/deploy/F3_R1/`）：

| 模型 | 来源 | state 维度/字段 | 额外输入 | checkpoint 路径 |
|---|---|---|---|---|
| M0 | 历史 P1_ocfix2_s1 100k | 8D：5关节.pos + gripper.pos + gripper.load + gripper.curr | 无 | `models/M0` |
| M1 | F2.1 A0 retry1 80k | 6D：5关节.pos + gripper.pos | 无 | `global` |
| M2 | F2.1 A1 60k | 6D：5关节.pos + gripper.pos | `observation.load`（单帧，全局融合） | `models/M2` |
| M3 | 同 M1 全局 + C1-load10k 局部 | 6D：同 M1 | 局部适配器：raw_load + gripper.pos + nominal_gripper | `global` + `local/010000.pt` |

M0 **不是纯 ACT**（state 内含 load + curr）。M1/M2/M3 全局 state 只取前六项位置；M2 的 `observation.load` 是额外单帧输入，不是第七个 state；M3 全局不接收 load/current，load 只进局部适配器。

模型身份、输入顺序、sha256 见 `result/model_identity.json`（与包内一致）。禁止替换 checkpoint、改 bound、改 state 顺序、加 current/W15/gate、重新训练。

---

## 2. 关键科学对照

| 对照 | 问题 |
|---|---|
| M0 vs M1 | 历史 8D P1 相对干净 6D 无反馈是否有真机优势（谨慎解读，M0 差异不止 load） |
| M2 vs M1 | 全局单帧 load 融合相对无反馈是否改善 |
| M3 vs M1 | 局部 load 残差调节相对干净 6D 是否改善 |
| **M3 vs M2（主）** | 局部 load 调节 vs 全局 load 融合，哪个真机闭环更优 |

共同条件：同一机械臂、相机、杯型、半杯水、任务、控制频率、部署主机、后处理、temporal ensembling、限幅。只允许模型架构差异。

---

## 3. 关键部署前置（PENDING，真机前必须完成）

以下项当前**未完成**，按 `deployment_config.json` 均为 `PENDING`。真机前必须先闭合并现场 Stage0：

1. **统一四模型入口 / 状态选择**：现 `src/lerobot/scripts/control_act_det.py` 从 follower 读 9D state（5关节 + gripper.pos + load + curr + master=0.0），按数据集元数据构帧。四个模型 state 维度不同（8D / 6D / 6D+load），尚无统一按模型选择/裁剪 state 的入口。必须新增显式按模型选择（M0 取前8、M1/M3 取前6、M2 取前6 + 单帧 load），并冻结与 `model_identity.json` 一致。
2. **M3 局部残差接线**：`control_act_det.py` 尚未接入 `LocalLoadRegulator`（`source/src/lerobot/policies/act/local_load_regulator.py` 提供 `disabled`/`shadow`/`regulated` 纯函数）。M3 shadow 与 regulated 都需在“全局后处理/ensembling 之后、机器人安全限幅之前”插入局部适配器。默认模式必须是 `disabled`。
3. **原型路径修正**：三份 `config.json` 的 `object_conditioning_prototype_path` 指向远程 `/root/autodl-tmp/F2_1_r1_20261019/inputs/cup_prototype_bank_v1.pt`（M1/M2/M3）或相对路径（M0），必须改为本机 `assets/cup_prototype_bank_v1.pt` 的绝对路径，并核验字节与 manifest 一致（同 G1 的修正流程）。
4. **DINOv2 / ResNet 缓存**：离线缓存 `torch_cache/hub/checkpoints/`（`dinov2_vits14_pretrain.pth`、`resnet18-f37072fd.pth`）与 `torch_cache/hub/facebookresearch_dinov2_main/`。部署进程需 `TORCH_HOME` 指向该 `torch_cache` 或等价离线加载，不能联网拉取。
5. **共同频率 / ensembling / 限幅 / 平滑**：四模型冻结同一控制频率、`temporal_ensemble_coeff`、`max_relative_target`、限幅与平滑。`deployment_config.json` 中 `control_frequency`/`temporal_ensembling`/`action_clipping`/`smoothing` 仍 `PENDING_COMMON`，必须在首个试次前固定，不得看到结果后更改。
6. **日志/视频/急停**：统一控制日志（每步 target/sent/实际夹爪/局部 delta）、逐 trial 视频、串口/相机失联急停、重置与停止机制。

**离线校验现状（本机 2026-10-20 复跑）**：

- 下载字节完整性：包内 639 个非 `.pyc` 文件全部与 manifest 一致；114 个 `__pycache__/*.pyc` 不一致（Python 字节码含源 mtime，跨机/跨补丁不可复现，导入时自动重编译，属打包缺陷，非损坏）。
- `verify_f3_multimodel_bundle.py`：**PASS**（M0=8D、M1=6D、M2=6D+load，M3 与 M1 同 global，前向有限）。
- `verify_f3_bundle.py` 的 **M3 局部适配器检查 FAIL**：包内 `validation/local_fixture.npz` 的 `expected_delta` 与包内 `local/010000.pt` 前向输出不一致（max_err≈1.77e-2，fp64 真值与输入顺序排列均已排除计算/顺序问题）。`local/010000.pt` 自身 sha256 与 `model_identity.json`/manifest 完全一致，说明是**打包时 fixture 与 checkpoint 不匹配**（fixture 由另一份权重生成），不是下载或环境问题。因此 **M3 局部适配器在本机无法通过离线 fixture 校验**，M3 regulated 必须先解决此不一致（重生成 fixture 或确认正确 checkpoint）再谈真机；shadow 亦需在能复现局部输出后再授权。

---

## 4. 部署方式与硬件

机械臂和两路相机直连本地 Linux 控制主机；不要通过无可靠串口/USB 转发与断线急停的云服务器控制机械臂。

需要 SO-101 follower、固定 top/wrist 相机、半杯水透明塑料杯、可立即断电/停动作的急停。臂运动范围内不得有人员、线缆和易损物。

```bash
cd ~/lerobot
git status --short
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls -l /dev/ttyACM* /dev/video* 2>/dev/null || true
```

无串口或两路相机时停止，不启动策略。

---

## 5. 相机与机械臂预检

top 画面覆盖桌面与杯子工作区，gripper/wrist 画面能看到夹爪与杯子接近区；两路不黑屏、不冻结、不错位、不过曝。机械臂状态连续可读、关节角无跳变、夹爪可低速开合。

**相机索引必须现场核验**（内核枚举顺序不稳定，不假设旧编号），记录：

```text
deployment_host = <主机名>
robot_port      = <例如 /dev/ttyACM0>
top_camera      = <设备索引或路径>
gripper_camera  = <设备索引或路径>
resolution/FPS  = <实际值>
code_commit     = <git rev-parse HEAD>
```

机械臂置于固定 `Home + Open`，固定底座、相机、光照、桌面与杯外观。

leader 仅用于人工复位（可选，预检/手动复位用，参考命令见下），正式 trial 前必须停止 teleoperate，`control_act_det.py` 只连 follower，leader 不提示或修正模型：

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

---

## 6. 安全门（全满足才运行）

- 杯中约半杯水，桌面干燥；
- 机械臂为 `Home + Open`；
- 夹爪、底座、相机、线缆无碰撞风险；
- 操作者手可触及急停；
- `max_relative_target` 保持部署合同安全值；
- 日志与视频目录可写。

出现碰撞、持续发散、夹爪挤压杯子、杯子倾倒、通信异常、相机掉帧或无法立即停止时，立即人工停止，标 `SAFETY_ABORT`（`BLOCKED_SAFETY`），不当普通任务失败。M3 一次调节器相关严重危险事件即停止其后全部干预试次，不调参续跑。

---

## 7. 试次协议

### 7.1 M3 shadow（必须先做）

M3 唯一引入在线动作修正模块，必须先 shadow：

```text
M3 shadow × 3 次有效试次
```

shadow 期间局部 MLP 运行、delta 记录、**修正不执行**；实际执行仍是冻结的 M3 global-only 6D 策略。

### 7.2 M3 shadow PASS 判据（8 条，全满足）

1. 臂动作不被改动；
2. 所有局部残差在冻结 bound（±4.832699459791184）内；
3. 无 NaN/Inf；
4. 无危险交变符号振荡；
5. near-bound 输出不占主导；
6. 残差行为无明显集中在有害的释放/张开；
7. 局部调节器 disable 路径可用；
8. 日志/视频完整。

shadow 失败 → M3 正式干预被阻断，只保留 M0/M1/M2。

### 7.3 正式试次（blocked rotation）

```text
Round 1: M0 → M1 → M2 → M3
Round 2: M0 → M1 → M2 → M3
Round 3: M0 → M1 → M2 → M3
```

每模型 3 次有效正式试次（4×3=12），加 shadow 3 次，共计划 15 次执行。M3 被阻断时每轮只 M0→M1→M2（9 次）。不得根据结果重排或增加试次。

### 7.4 试次间复位

每次 trial 前：回 `Home + Open` → 按协议恢复杯位（不总放格子中心）→ 检查两路画面 → 确认半杯水 → 清空 temporal buffer/action queue → 按模型正常部署路径 reset 状态/历史。记录任何复位偏差。换模型必须停止上一进程、回 `Home + Open`、清缓存、重确认相机，不得继承上一 trial 隐状态。

### 7.5 试次有效性

有效正式试次须满足：checkpoint/config 正确；日志完整；相机录像完整；杯/设置符合协议；无无关硬件故障；安全事件前或任务完成后无人工干预。安全干预试次保留在记录中，不得隐藏失败。

---

## 8. 真机推理命令（参数沿用 G1 指南，部署前置完成后再用）

现场需先完成 §3 的统一入口/状态选择/M3 接线/原型路径/DINO 缓存，下列命令是**沿用上次 G1 指南参数**的模板，`--policy.path` 按模型替换。当前 `control_act_det.py` 尚未支持四模型统一选择与 M3 局部接线，**直接照抄不会得到正确结果**。

```bash
cd ~/lerobot
source .venv/bin/activate            # 按实际环境修改
export ROBOT_PORT=/dev/ttyACM0
export TOP_CAM=4
export WRIST_CAM=2
export TORCH_HOME=/home/lyj/lerobot/outputs/deploy/F3_R1/torch_cache
export BUNDLE=/home/lyj/lerobot/outputs/deploy/F3_R1
export LOG_DIR=experiments/force/f3_real_robot_multi_model_comparison/index1/runtime_logs
mkdir -p "$LOG_DIR"
```

每次只启动一个模型；M0 用 `--policy.path=$BUNDLE/models/M0`，M1/M3 用 `--policy.path=$BUNDLE/global`，M2 用 `--policy.path=$BUNDLE/models/M2`。命令骨架（沿用 G1 参数）：

```bash
python src/lerobot/scripts/control_act_det.py \
    --policy.path="$BUNDLE/models/M0" \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --robot.max_relative_target=2 \
    --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id=nn \
    --robot.cameras="{top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --dataset.repo_id=<对应模型 state schema 的冻结数据集 root> \
    --dataset.root=<数据集 root> \
    --dataset.single_task="Cup pick and place" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
    2>&1 | tee "$LOG_DIR/<model>_$(date +%Y%m%d_%H%M%S).log"
```

> 注：G1 指南中 `--robot.max_relative_target` 出现两次（2 与 5），按 STATUS 记录实机已“下调至 2”，这里统一取 `2`。`--dataset.root/--dataset.repo_id` 必须与各模型 state 维度匹配（M0=8D、M1/M3=6D、M2=6D+load），这是 §3-1 要解决的统一入口问题，不能用一个 9D/8D root 套所有模型。

每次 trial 结束按 `Ctrl+C` 停止，确认安全后再启动下一模型；`Ctrl+C` 本身不算完成一次 trial，必须填写该 trial 阶段结果、失败类型、路径。

---

## 9. 现场记录与指标

每个 trial 立即填写：

```text
trial_id / model / round / repeat
grasp_established / lift_achieved / approx_lift_height / hold_duration
visible_slip / premature_release / excessive_deformation
stable_transport / stable_placement / correct_release
full_task_chain_completion / operator_emergency_intervention
video_top / video_wrist / notes
```

同时保留每步 `gripper 命令 / gripper.pos / gripper.load / gripper.current(若有) / 臂 state/action / 视频 / 控制日志`。

Lift 以杯离桌并稳定约 0.5s 计；抓起后立即滑落与保持阶段滑落分开记录。每个视频与 trial_id 一一对应。

### M3 专属指标（若 M3 真机）

```text
delta_gripper 时间线、mean/median/max |delta_gripper|、near-bound 比例、
sign-change 率、非零比例、主要修正事件
```

臂输出必须不被局部调节器改动。

### 统一描述表（每 trial + 每模型汇总）

| Model | Feedback strategy | Grasp | Lift | Hold | Slip | Deformation | Place | Release | Full chain | Safety |
|---|---|---|---|---|---|---|---|---|---|---|

### 结果分类（完成后只填一个）

`M3_REAL_ROBOT_PROMISING` / `M2_GLOBAL_LOAD_REMAINS_BEST` / `ACTUATOR_FEEDBACK_NOT_CLEAR_ON_ROBOT` / `M3_UNSAFE_OR_DESTABILIZING` / `UNRESOLVED`。禁止 inferential 显著性，只允许 pilot advantage / 一致行为信号 / feasibility 差异 / 描述性对比。

---

## 10. 安全停止条件（任一触发立即停当前试次）

危险过闭合、可见挤压杯子、高频夹爪振荡、意外反复收紧、干预导致的提前释放、非有限动作、M3 残差超 bound、局部代码改动臂动作、硬件安全风险。

---

## 11. 完成判定与交付

`F3_R1_COMPLETE=true` 需满足（见 decision §24）：身份全冻结、共同部署配置审计、M3 shadow×3 或正式阻断、所有未阻断模型计划有效试次完成、安全事件全保留、逐 trial 指标完整、统一对比表完整、M3 机制指标（若测）、视频/日志索引、Final12 未动、Builder/Red-Team 证据完整、结果分类已定。

结果包路径 `result/`，需补齐 `summary.md`、`validation.md`、`model_identity/`、`deployment_config.json`、`trial_manifest.csv`、`per_trial_metrics.csv`、`model_summary.csv`、`m3_shadow_metrics.csv`、`m3_shadow_review.md`、`m3_regulator_metrics.csv`、`safety_review.md`、`mechanism_review.md`、`video_index.md`、`realtime_log_index.md`、`leakage_audit.md`、`figures/`、`proposal.md`。

未完成全部计划试次前，不得置 `F3_R1_COMPLETE=true`，不得进入 Final12、下一轮或重新训练。
