# P4.3 四模型真机测试指南

本指南遵循 `docc/指南/真机测试经验与代码指令参考.md`。仅准备部署，不自动运行机械臂。

| 命令模型名 | 模型 | checkpoint | 预处理 |
|---|---|---|---|
| L1 | Legacy P1，seed 1 | 80k | 原 Legacy 双重归一化 |
| F3 | Fixed P1，seed 3 | 80k | 显式 raw Top，DINO 单次归一化 |
| A1 | Original ACT，seed 1 | 80k | 无 DINO |
| L1_160K | Legacy P1，seed 1，L1 续训 | 160k | 原 Legacy 双重归一化 |

F3 与 L1 的 seed 不同，不是严格的单变量因果对照。L1 与 L1_160K 可比较继续训练后的行为变化；不预先假定 160k 更好。

## 下载与安装

在连接机械臂的 Linux 主机使用本项目环境。服务器不连接机械臂。本轮离线校验环境为 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0；源码要求 Python ≥3.12。已有兼容环境可直接使用，CUDA wheel 与现场显卡驱动应匹配，不需要为了下载模型重复安装 PyTorch。模型仓库：
https://huggingface.co/QYyyyyyyy/P4_3_FORMAL6_SO101

```bash
hf download QYyyyyyyy/P4_3_FORMAL6_SO101 --local-dir outputs/deploy/P4_3
unzip -o outputs/deploy/P4_3/runtime_source.zip -d outputs/deploy/P4_3/runtime_source
pip install -e "outputs/deploy/P4_3/runtime_source[feetech]"
mkdir -p outputs/p43/session_01
export BUNDLE="$PWD/outputs/deploy/P4_3"
export SESSION="$PWD/outputs/p43/session_01"
```

源代码快照包含本轮运行入口及依赖源码，不覆盖当前 Git 工作区。四模型、Train90 统计、杯原型、DINO/ResNet 离线缓存均在 bundle 中。实时推理不需要下载 formal6 视频数据集。禁止直接将 F3 用 Legacy 的 control_act_det 入口运行。

可先不连接机械臂做一次代码冒烟；零图像输出不是定位能力或真机验证：

```bash
python -m lerobot.scripts.control_p43 --bundle="$BUNDLE" --session-dir="$SESSION" --dry-run --model L1_160K --position P01 --repeat 1 --policy.device=cuda
```

## 现场准备与试次

从臂 `/dev/ttyACM0`、id `nn`；顶部相机 4，腕部相机 2，RGB 640×480、30fps。沿用采集时同一套 nn 标定，明确 `use_degrees=true`；夹爪使用原生 0–100 范围。不要在新机械臂上套用旧标定。该入口自主推理不需要连接主臂；若需单独遥操作复位，历史主臂为 `/dev/ttyACM1`、id nn，先结束自主程序再操作。

每次手动摆回相同 Home 并打开夹爪，不要求数值 Home 姿态或容差。保持半杯水、杯型、相机位置/曝光、照明、抓取与放置区域一致。机械臂视角左侧抓取、右侧放置，图像可能反向。P01–P06 是现场六个固定杯位，先记下位置示意并保持不变，不把未经测量的像素当毫米坐标。

`trial_plan_frozen.csv`：原 L1/F3/A1 的 36 次分块对比（六位置×两重复×三模型，随机种子 20261029），之后追加独立标记的 L1_160K 12 次（六位置×两重复）。这是用户追加 160k 的补充块，不冒充原 decision 三模型预注册。每次只手动启动一条指令，将 `--position` 与 `--repeat` 改成表中值。此指南不自动连续操作机械臂。

急停/断电方式随手可及。碰撞、异常夹持、掉杯、洒水或通信异常立即停止。运行时不要伸手移动杯子或机械臂。右方向键或终端输入 `end` 回车可正常提前结束；Ctrl+C/Esc 可中止，不写阻塞下一次运行的 STOP 锁。退出会尝试断开与清理；强杀/断电无法保证 finally 执行。下一次启动创建新策略并 reset。

`--policy.temporal_ensemble_coeff=0.01`、`--robot.max_relative_target=5` 都是命令行参数，可人工修改；公平对比时四模型保持一致并记录改动。这里均用 n_action_steps=1、时序集成、30fps。不要将“时序平滑”当成接触力控制；这四个候选没有 New-GL 的 local regulator。

## 记录与结果解释

每次自动保存命名 CSV、顶部/腕部视频、首帧、配置、summary、阶段审核模板；同名重测会归档旧目录。中途完成动作后可正常提前结束，不必留下待机。人工审核接近、闭爪、抬升、保持、搬运、放置和释放；优先比较完整动作链、杯位跟踪、抬升高度/保持、抽搐和异常。只有人工审核后才能写成功/失败，不能将进程退出码当成功率。帮助恢复不算自主成功。请另写现场观察，失败记录也保留。

下面四条指令分别复制运行，不要同时连接同一机械臂。先完成上述下载设置 BUNDLE/SESSION。每条默认 P01、第 1 次重复；后续按冻结表修改。

## 各模型直接运行指令
### L1

```bash
python -m lerobot.scripts.control_p43 \
  --bundle="$BUNDLE" --session-dir="$SESSION" \
  --execute --model L1 --position P01 --repeat 1 \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=160 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$SESSION/L1_$(date +%Y%m%d_%H%M%S).log"
```

### F3

```bash
python -m lerobot.scripts.control_p43 \
  --bundle="$BUNDLE" --session-dir="$SESSION" \
  --execute --model F3 --position P01 --repeat 1 \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$SESSION/F3_$(date +%Y%m%d_%H%M%S).log"
```

### A1

```bash
python -m lerobot.scripts.control_p43 \
  --bundle="$BUNDLE" --session-dir="$SESSION" \
  --execute --model A1 --position P01 --repeat 1 \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$SESSION/A1_$(date +%Y%m%d_%H%M%S).log"
```

### L1_160K

```bash
python -m lerobot.scripts.control_p43 \
  --bundle="$BUNDLE" --session-dir="$SESSION" \
  --execute --model L1_160K --position P01 --repeat 1 \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$SESSION/L1_160K_$(date +%Y%m%d_%H%M%S).log"
```
