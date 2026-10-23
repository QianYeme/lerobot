# R5：五模型真机定位与抓取对照指南

按`docc/指南/真机测试经验与代码指令参考.md`的直接运行方式编写。只执行R5，不采集新训练数据、不重训、不改变模型。**目前真机0次，R5_COMPLETE=false。** 云服务器只用于准备/上传，下面的真机指令必须在连接机械臂与两相机的Linux电脑上运行。

## 1. 五个模型与固定测试顺序

| CLI模型 | 实际模型 | 配对关系 |
| --- | --- | --- |
| M1 | Strong-G-36-S3，Train36 seed3/80k | 旧global参考 |
| M2 | Strong-G-60-S3，Full60 seed3/80k | Full60 global |
| M3 | New-GL-36，M1+F4.2 local seed3/7k | 旧双路 |
| M4 | New-GL-60，M2+Full60 local seed3/7k | 新双路 |
| M5 | G60-BadSeed-S4，Full60 seed4/80k，无local | 真机前固定的弱seed对照 |

M5依据R4修复后arm/gripper综合误差及空间诊断预先固定为S4，不是看到真机表现再挑。M1/M3共享旧global，M2/M4共享新global；旧、新组分别使用Train36/Full60 normalizer，不能混用。五条路径均是6D位置state+独立单帧load，无主臂信号/current策略输入。local只修正gripper，不改arm。

只做同一熟悉工作区A/B/C，不扩大范围，不新增抖动分布；同杯型/水位/相机/桌面和Home+Open复位方式。正式预算共15次，不追加失败后的“救援重试”。

| 轮次 | 在终端设置 | 依次复制文末指令 |
| --- | --- | --- |
| A | `export R5_POSITION=A` | M1 → M2 → M3 → M4 → M5 |
| B | `export R5_POSITION=B` | M2 → M3 → M4 → M5 → M1 |
| C | `export R5_POSITION=C` | M3 → M4 → M5 → M1 → M2 |

不要按文末M1→M5顺序机械重复B/C轮。若弱seed因硬件或安全原因确实无法运行，保留具体原因；12个核心试次仍需完成。出现危险先停止，不为凑满预算继续。

## 2. 下载项目与模型包

推荐新建或使用无本地未提交修改的项目副本，不覆盖个人代码。GitHub使用`det`分支：

```bash
git clone -b det https://github.com/QianYeme/lerobot.git
cd lerobot
# 已有项目：先保存本机修改，再git pull --ff-only origin det。
mkdir -p outputs/deploy
hf download QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101 \
  R5_deploy_bundle.tar.gz R5_project_files.tar.gz checksums.json \
  --local-dir outputs/deploy
```

若直连受限，可先`export HF_ENDPOINT=https://hf-mirror.com`重跑；镜像可能尚未同步最新版，以仓库中的发布receipt/revision为准，不用旧F5包冒充R5。

模型包包含3个global、2个local、各自normalizer、cup prototype和DINO/ResNet离线缓存，**真机推理不需要下载完整formal4数据集**。项目文件包用于无法拉取GitHub时的代码备用；常规使用已更新的GitHub代码即可。

下载后核对两个归档：

```bash
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('outputs/deploy'); checks=json.loads((root/'checksums.json').read_text())
for name in ['R5_deploy_bundle.tar.gz','R5_project_files.tar.gz']:
    p=root/name
    with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert p.stat().st_size==checks[name]['bytes'] and actual==checks[name]['sha256'], name
    print(name,'OK')
PY
tar -xzf outputs/deploy/R5_deploy_bundle.tar.gz -C outputs/deploy
# 仅当无法取得更新的GitHub代码时，在已备份或新项目副本中使用备用代码包：
# tar -xzf outputs/deploy/R5_project_files.tar.gz -C .
python src/lerobot/scripts/control_r5.py --help
```

沿用已有能正常控制SO101的LeRobot环境，不为本轮盲目升级torch/CUDA。包在服务器Python3.12/torch2.8.0+cu128/RTX5090上验证，不能据此保证其它系统；缺包时按项目依赖安装，已有环境可`python -m pip install -e . --no-deps`。执行入口自动设置`cudnn.benchmark=False`且TF32保持True，不能改回True。

## 3. 一次基本检查，不加shadow/审批锁

- 从臂`so101_follower`、`/dev/ttyACM0`、`robot.id=nn`；实际标定id须与当前实物匹配。历史存在`nn`/`nnn`不同标定，文件能加载不代表关节语义匹配。**不要为了让命令启动而随意切换id。**
- top index4，gripper index2，RGB640×480/30fps，腕部MJPG；以画面确认没有接反，沿用训练视角。相机/串口/id若需适配，在正式测试前统一改五条指令并记录，不能某模型失败后另换。
- 策略推理不需要主臂。主臂`so101_leader`、`/dev/ttyACM1`、`teleop.id=nn`仅用于两试次之间慢速复位；先停止策略释放从臂串口，再运行下面的teleop；复位完成后停止teleop，不能双程序同时占串口。
- 代码直接核对所选checkpoint/config/normalizer及local精确配对；没有Stage0、shadow、代码指纹审批或持久STOP锁，也不会自动标定/自动归零。
- 保留既有物理限制：`max_relative_target=5`和local ±4.832699459791184。急停/断电随手可及，碰撞、挤压/倾倒、洒水、通信异常立即停止。程序退出可能释放力矩，提前安全支撑机械臂和杯子。

```bash
python src/lerobot/scripts/lerobot_teleoperate.py \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=nn
```

在`outputs/r5/session_01/positions.md`记录A/B/C相对固定桌边的位置、杯子水位、Home+Open方式和一张三点位照片。不要求新增厘米坐标，也不能根据失败移动杯位。

## 4. 共同参数与中止方式

五条指令都显式使用`n_action_steps=1`、`temporal_ensemble_coeff=0.01`、`max_relative_target=5`、30Hz、最多60秒/试次、30秒手动复位时间参考。ensemble和relative limit都可在CLI改，但**正式R5只能开测前统一冻结，不得看到结果后给不同模型调参**。

每条指令是一段试次，不会自动跑下一段。按右方向键正常提前结束，记OPERATOR_END；Ctrl+C或Esc停止，记SAFETY_ABORT，不写持久锁。正确抓起后手动结束不是失败证据；未看到的保持/放下/释放填UNKNOWN，不能自动当PASS。程序退出后自行按固定方式复位、等设备释放，再手动启动下一模型。

代码允许同名目录重跑并把旧目录移到`_prev_时间戳`保留证据，但R5科研预算**不允许因模型失败追加救援重复**。相机打开失败且尚未发动作等工程无效尝试须保留日志、说明原因，不得悄悄删除后挑一条成功。

## 5. 看哪些阶段，如何记录

| 阶段 | PASS的观察依据 |
| --- | --- |
| P0 目标定向接近（主指标） | 起步明显朝当前杯位，而非先伸向固定中间位置再缓慢追踪 |
| P1 可抓取几何（主指标） | 闭爪之前、未经人工帮助，杯子进入合理夹持区域 |
| P2 闭爪 | 出现实际闭爪动作；不等于已抓稳 |
| P3 抬升 | 杯子确实离开支撑面，不是机械臂空抬 |
| P4 保持 | 可见持杯保持；记录实际持续时长，不给本轮另加未冻结时长阈值 |
| P5 放下 | 杯子受控回到支撑面 |
| P6 释放 | 放稳后打开夹爪释放 |

每阶段只能PASS/FAIL/UNKNOWN；视频不足、遮挡、提前终止时按证据填UNKNOWN，不能凭CSV动作值脑补。完整自主成功须整链真实可见且无辅助。

自主明确失败后才可有限辅助查看后续动作。在终端输入`assist`并回车，先标记自主A结束、辅助B开始，再用安全方式辅助；**不要把手伸入运动机械臂区域**。如需进入工作区，应先按已有停机安全操作停止运动，不为保持日志连续而冒险。记录辅助类型和自主首次失败阶段，辅助后成功绝不计自主成功。

手机视频开头口述或拍纸条写清`MODEL_ID / POSITION / TRIAL_ID`，如`M4 / B / R5_M4_B_01`。运行端同时自动保存相机视频与CSV，视频和CSV只按共同身份关联，不强行做逐帧时间对齐。

## 6. 输出在哪里，测试后交回什么

每试次目录为`outputs/r5/session_01/R5_Mx_A或B或C_01/`：

- `configuration.json`：精确模型、normalizer、相机/id/ensemble/limit和benchmark=False。
- `R5_Mx_位置_01.csv`：state/nominal/requested/sent、load、gripper.pos、nominal/delta/final gripper及自主/辅助phase。
- 同身份top/gripper视频、首帧、`summary.json`、`stage_review.json`；M3/M4另有`local_regulator_metrics.json`（含自主/辅助分段统计）。
- session级`trial_manifest.csv`与终端log。退出状态不等于行为成败；`stage_review.json`初始UNKNOWN，需逐条根据视频观察填写，validity写VALID/HARDWARE_INVALID/OPERATOR_INVALID/TASK_FAILURE等事实分类，不把抓取失败当工程无效。

测试后运行：

```bash
python scripts/summarize_r5.py --session-dir outputs/r5/session_01
```

它只汇总已审核的阶段和已有local日志，不自动判视频、不替你宣布结果。把整个session、手机视频和人工备注交回；不要只留下“最好”的一段。R5主要看P0/P1的每模型x/3，同时保留UNKNOWN/无效数量；15次属于描述性小样本，不能写成统计显著的泛化结论。

## 7. 五个模型的直接运行指令

先按第1节设置`export R5_POSITION=A`，A轮按M1→M5执行；B/C轮按表中旋转顺序执行。每块都可以直接复制，路径参数/相机/id可在开测前统一调整。

### M1：Strong-G-36-S3

```bash
mkdir -p "$PWD/outputs/r5/session_01"
python src/lerobot/scripts/control_r5.py \
  --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/session_01" \
  --execute --model M1 --position="${R5_POSITION:-A}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5/session_01/M1_${R5_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M2：Strong-G-60-S3

```bash
mkdir -p "$PWD/outputs/r5/session_01"
python src/lerobot/scripts/control_r5.py \
  --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/session_01" \
  --execute --model M2 --position="${R5_POSITION:-A}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5/session_01/M2_${R5_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M3：New-GL-36

```bash
mkdir -p "$PWD/outputs/r5/session_01"
python src/lerobot/scripts/control_r5.py \
  --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/session_01" \
  --execute --model M3 --position="${R5_POSITION:-A}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5/session_01/M3_${R5_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M4：New-GL-60

```bash
mkdir -p "$PWD/outputs/r5/session_01"
python src/lerobot/scripts/control_r5.py \
  --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/session_01" \
  --execute --model M4 --position="${R5_POSITION:-A}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5/session_01/M4_${R5_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```

### M5：G60-BadSeed-S4

```bash
mkdir -p "$PWD/outputs/r5/session_01"
python src/lerobot/scripts/control_r5.py \
  --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/session_01" \
  --execute --model M5 --position="${R5_POSITION:-A}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/r5/session_01/M5_${R5_POSITION:-A}_$(date +%Y%m%d_%H%M%S).log"
```
