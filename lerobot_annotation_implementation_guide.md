# LeRobot 附件标注方法完整实现指南
ls /dev/ttyACM*
lerobot-find-cameras opencv

sudo chmod 666 /dev/ttyACM*
sudo chmod 666 /dev/video*

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=nn

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=nn

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=nn \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=nn

lerobot-teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=nn     --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}"     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=nn     --display_data=true

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=nn \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=nn \
    --display_data=true

lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}' \
  --robot.id=nn \
  --display_data=false \
  --dataset.repo_id=formal_A \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=1000 \
  --policy.path=outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model

python src/lerobot/scripts/control_act_det.py \
      --policy.path=outputs/train/2026-08-08/01-17-03_act_det/checkpointsE5/last/pretrained_model  \
      --policy.temporal_ensemble_coeff=0.01 \
      --policy.n_action_steps=1 \
      --robot.max_relative_target=5.0 \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=nn \
      --robot.cameras='{gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}' \
      --dataset.repo_id formal1_B \
      --dataset.root /home/lyj/lerobot/数据集/formal1_B \
      --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
      --dataset.num_episodes 1 \
      --dataset.episode_time_s 60 \
      --dataset.fps 30

python src/lerobot/scripts/control_act_det.py \
      --policy.path=outputs/train/2026-08-08/01-17-21_act_det/checkpointsE6/last/pretrained_model \
      --policy.annotation_dir=/home/lyj/lerobot/数据集/formal1_B/annotations \
      --policy.temporal_ensemble_coeff=0.01 \
      --policy.n_action_steps=1 \
      --robot.max_relative_target=5.0 \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=nn \
      --robot.cameras='{gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width:
  640, height: 480, fps: 30}}' \
      --dataset.repo_id formal1_B \
      --dataset.root /home/lyj/lerobot/数据集/formal1_B \
      --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
      --dataset.num_episodes 1 \
      --dataset.episode_time_s 60 \
      --dataset.fps 30

python src/lerobot/scripts/control_act_det.py \
    --policy.path=outputs/train/E5_det_B_240k \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=nn \
    --robot.max_relative_target=5 \
    --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width:
  640, height: 480, fps: 30}}' \
    --dataset.repo_id formal1_B \
    --dataset.root /home/lyj/lerobot/数据集/formal1_B \
    --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
    --dataset.num_episodes 1 \
    --dataset.episode_time_s 60 \
    --dataset.reset_time_s 15 \
    --dataset.fps 30

outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model   
  ┌──────┬────────────────────────────────────────────────────────────────────────────────┬───────────┐
  │ 模型 │                          --policy.path= 的 checkpoint                           │  数据集    │
  ├──────┼────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E2   │ outputs/train/2026-08-07/02-56-09_act_det/checkpointsE2/last/pretrained_model   │ formal_A  │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E3   │ outputs/train/2026-08-07/16-25-39_act_det/checkpointsE3/last/pretrained_model   │ formal_A  │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E3b  │ outputs/train/2026-08-07/16-26-04_act_det/checkpointsE3b/last/pretrained_model  │ formal_A  │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E4   │ outputs/train/2026-08-07/16-28-36_act/checkpointsE4/last/pretrained_model       │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E5   │ outputs/train/2026-08-08/01-17-03_act_det/checkpointsE5/last/pretrained_model   │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E6   │ outputs/train/2026-08-08/01-17-21_act_det/checkpointsE6/last/pretrained_model   │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E6b  │ outputs/train/2026-08-08/01-17-41_act_det/checkpointsE6b/last/pretrained_model  │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E7   │ outputs/train/2026-08-08/12-32-41_act_det/checkpointsE7/last/pretrained_model   │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E7b  │ outputs/train/2026-08-09/00-03-00_act_det/checkpointsE7b/last/pretrained_model  │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E8   │ outputs/train/2026-08-08/12-33-40_act_det/checkpointsE8/last/pretrained_model   │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E9   │ outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model   │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E_A0 │ outputs/train/2026-08-09/00-03-19_act_det/checkpointsE_A0/last/pretrained_model │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E_A1 │ outputs/train/2026-08-09/00-03-39_act_det/checkpointsE_A1/last/pretrained_model │ formal1_B │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ E_A2 │ outputs/train/2026-08-09/09-02-51_act_det/checkpointsE_A2/last/pretrained_model │ formal1_B │
  └──────┴─────────────────────────────────────────────────────────────────────────────────┴───────────┘
## 1. 核心问题解答

### 1.1 tasks.jsonl 文件在哪里创建？

**不在 LeRobot 仓库中创建**，而是在**数据集缓存目录**中创建。

数据集存储路径：
```
~/.cache/huggingface/lerobot/[repo_id]/
```

例如，您的数据集 `TommyZihao/lerobot_zihao_dataset_a` 的完整路径为：
```
~/.cache/huggingface/lerobot/TommyZihao/lerobot_zihao_dataset_a/
```

完整的数据集目录结构：
```
~/.cache/huggingface/lerobot/TommyZihao/lerobot_zihao_dataset_a/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.jsonl          # ← 在这里创建
│   └── episodes/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
│       └── ...
└── videos/
    ├── observation.images.gripper/
    ├── observation.images.side/
    └── observation.images.top/
```

### 1.2 三摄像头配置修改

您的原始命令使用单摄像头（front），需要修改为三摄像头配置（夹爪、侧、俯视角）。

**修改前（单摄像头）：**
```bash
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}}"
```

**修改后（三摄像头）：**
```bash
--robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, top: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}}"
```

**摄像头名称说明：**
- `gripper`：夹爪摄像头（对应原 "夹爪"）
- `side`：侧面摄像头（对应原 "侧"）
- `top`：俯视摄像头（对应原 "俯视角"）

**重要提示：**
- `index_or_path` 需要根据实际摄像头设备ID调整
- 使用 `lerobot-find-cameras opencv` 命令查看摄像头ID
- 不要将两个同款摄像头连接在同一个 USB-HUB 上

## 2. 完整实现流程

### 2.1 步骤一：准备数据集目录

首先运行采集命令，创建数据集基础结构：
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=nn \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=nn \
    --display_data=true \
    --dataset.repo_id=formal1/kind1 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=24 \
    --dataset.reset_time_s=7
```

rm -rf /home/lyj/.cache/huggingface/lerobot/formal1/kind1

**注意：** 先采集 1 个 episode 来创建数据集目录结构。

### 2.2 步骤二：创建预定义任务列表

找到数据集目录并创建 `tasks.jsonl` 文件：

```bash
# 1. 进入数据集目录
cd ~/.cache/huggingface/lerobot/TommyZihao/lerobot_zihao_dataset_a/

# 2. 创建 meta 目录（如果不存在）
mkdir -p meta

# 3. 创建 tasks.jsonl 文件
cat > meta/tasks.jsonl << 'EOF'
{"task": "Grasp the half-filled transparent cup steadily from the table without spilling or crushing", "task_id": "1A", "category": "基础平稳抓取", "description": "平稳端起桌面上的半满透明水杯，全程不洒漏、不捏瘪"}
{"task": "Close the gripper to critical contact force, slowly lift the cup, keep liquid level horizontal", "task_id": "1B", "category": "基础平稳抓取", "description": "收紧夹爪至临界接触力，缓慢提升水杯，全程保持液面水平"}
{"task": "Approach the cup from the side, adjust the grasp angle and lift steadily", "task_id": "1C", "category": "基础平稳抓取", "description": "从侧面接近水杯，调整抓取角度后平稳端起"}
{"task": "Locate and grasp the transparent cup while avoiding visual distractors", "task_id": "2A", "category": "视觉鲁棒性对抗", "description": "绕过视觉干扰物，准确端起半满透明水杯"}
{"task": "Identify the transparent cup under strong sidelight and grasp it", "task_id": "2B", "category": "视觉鲁棒性对抗", "description": "在强侧光条件下，识别透明杯轮廓并抓取"}
{"task": "Accurately identify the transparent cup in complex background and complete grasping", "task_id": "2C", "category": "视觉鲁棒性对抗", "description": "在复杂背景中准确识别透明水杯并完成抓取"}
{"task": "Grasp the transparent cup with a small amount of liquid, maintain gentle stability", "task_id": "3A", "category": "力控适应性", "description": "抓取少量液体的透明水杯，保持轻柔稳定"}
{"task": "Grasp the transparent cup with a large amount of liquid, maintain stability without overflowing", "task_id": "3B", "category": "力控适应性", "description": "抓取大量液体的透明水杯，保持平稳不溢出"}
{"task": "Grasp the transparent cup with wet surface, prevent slipping", "task_id": "3C", "category": "力控适应性", "description": "抓取表面湿润的透明水杯，防止打滑"}
{"task": "Readjust grasping strategy after grasping slip", "task_id": "4A-1", "category": "失败引导与恢复", "description": "在抓取滑脱后重新调整抓取策略"}
{"task": "Adjust posture to restore horizontal when cup tilts", "task_id": "4A-2", "category": "失败引导与恢复", "description": "在杯身倾斜时调整姿态恢复水平"}
{"task": "Stop immediately and adjust cup when liquid spills", "task_id": "4A-3", "category": "失败引导与恢复", "description": "在液体洒漏时立即停止并调整杯身"}
{"task": "Move the cup from current position to left 20cm position", "task_id": "5A-1", "category": "任务变体与指令泛化", "description": "将水杯从当前位置移动到左侧20cm位置"}
{"task": "Move the cup from current position to right 20cm position", "task_id": "5A-2", "category": "任务变体与指令泛化", "description": "将水杯从当前位置移动到右侧20cm位置"}
{"task": "Move the cup from current position to front 30cm position", "task_id": "5A-3", "category": "任务变体与指令泛化", "description": "将水杯从当前位置移动到前方30cm位置"}
{"task": "Grasp the cup from front angle", "task_id": "5B-1", "category": "任务变体与指令泛化", "description": "从正面角度抓取水杯"}
{"task": "Grasp the cup from 30-degree side angle", "task_id": "5B-2", "category": "任务变体与指令泛化", "description": "从30度侧面角度抓取水杯"}
{"task": "Grasp the cup from 60-degree side angle", "task_id": "5B-3", "category": "任务变体与指令泛化", "description": "从60度侧面角度抓取水杯"}
{"task": "Pass the cup to me", "task_id": "5C-1", "category": "任务变体与指令泛化", "description": "把杯子递给我"}
{"task": "Place the cup on the left side of the table", "task_id": "5C-2", "category": "任务变体与指令泛化", "description": "将水杯放到桌子左边"}
{"task": "Carefully pick up the cup", "task_id": "5C-3", "category": "任务变体与指令泛化", "description": "请小心地拿起杯子"}
{"task": "Slowly lift the cup", "task_id": "5C-4", "category": "任务变体与指令泛化", "description": "慢慢端起水杯"}
{"task": "Move the cup to the right", "task_id": "5C-5", "category": "任务变体与指令泛化", "description": "把杯子移到右边"}


EOF
```

### 2.3 步骤三：使用预定义列表采集数据

修改采集命令，使用预定义任务列表：

```bash
# 采集任务0的数据
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=zihao_follower_arm \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, top: {type: opencv, index_or_path: 4, width: 1280, height: 720, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=zihao_leader_arm \
    --display_data=true \
    --dataset.repo_id=TommyZihao/nn_dataset \
    --dataset.num_episodes=40 \
    --dataset.single_task="Grasp the half-filled transparent cup steadily, lift it without spilling or deforming" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=15
```

**关键点：**
- `--dataset.single_task` 的值必须与 `tasks.jsonl` 中的某个任务描述完全一致
- 系统会自动匹配任务描述到对应的 `task_index`

### 2.4 步骤四：批量采集多个任务

为每个任务创建单独的采集脚本：

```bash
#!/bin/bash
# batch_collect.sh

# 任务0：基础抓取
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AAF2193061 \
    --robot.id=zihao_follower_arm \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, top: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AAF2194741 \
    --teleop.id=zihao_leader_arm \
    --display_data=true \
    --dataset.repo_id=TommyZihao/lerobot_zihao_dataset_a \
    --dataset.num_episodes=40 \
    --dataset.single_task="抓取桌上的半满透明塑料杯" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=2

# 任务1：侧面抓取
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AAF2193061 \
    --robot.id=zihao_follower_arm \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}, top: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 60, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AAF2194741 \
    --teleop.id=zihao_leader_arm \
    --display_data=true \
    --dataset.repo_id=TommyZihao/lerobot_zihao_dataset_a \
    --dataset.num_episodes=40 \
    --dataset.single_task="从侧面抓取桌上的半满透明塑料杯" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=2

# ... 为其他任务重复类似命令
```

## 3. 关键配置说明

### 3.1 摄像头命名规范

根据 LeRobot 最佳实践，摄像头命名建议使用：
- `top`：俯视摄像头（对应 "俯视角"）
- `side`：侧面摄像头（对应 "侧"）
- `gripper`：夹爪摄像头（对应 "夹爪"）

这些名称会自动映射为：
- `observation.images.top`
- `observation.images.side`
- `observation.images.gripper`

### 3.2 任务描述规范

根据附件中的建议：
- **清晰具体**：描述机器人要执行的具体动作和涉及的对象
- **简洁性**：25-50 字符为宜
- **避免模糊**：不要使用 "task1"、"demo2" 等无意义名称
- **一致性**：相同任务使用完全相同的描述

### 3.3 数据集结构验证

采集完成后，验证数据集结构：
```bash
# 查看数据集结构
tree ~/.cache/huggingface/lerobot/TommyZihao/lerobot_zihao_dataset_a/

# 验证 tasks.jsonl 内容
cat ~/.cache/huggingface/lerobot/TommyZihao/lerobot_zihao_dataset_a/meta/tasks.jsonl

# 查看数据集信息
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('TommyZihao/lerobot_zihao_dataset_a')
print(f'数据集大小: {len(dataset)}')
print(f'特征: {list(dataset.features.keys())}')
"
```

## 4. 常见问题解决

### 4.1 摄像头索引问题

**问题**：摄像头索引不匹配
**解决**：
```bash
# 查看可用摄像头
lerobot-find-cameras opencv

# 根据输出调整 index_or_path 参数
# 如果摄像头ID为 0, 1, 2，则配置为：
--robot.cameras="{ gripper: {type: opencv, index_or_path: 0, ...}, side: {type: opencv, index_or_path: 1, ...}, top: {type: opencv, index_or_path: 2, ...}}"
```

### 4.2 任务匹配问题

**问题**：`single_task` 与 `tasks.jsonl` 不匹配
**解决**：
```bash
# 确保 single_task 的值与 tasks.jsonl 中的 task 字段完全一致
# 包括空格、标点符号等

# 查看当前数据集的任务列表
python -c "
import json
with open('/path/to/meta/tasks.jsonl', 'r') as f:
    for line in f:
        task = json.loads(line)
        print(f'Index {task[\"task_index\"]}: {task[\"task\"]}')
"
```

### 4.3 数据集路径问题

**问题**：找不到数据集目录
**解决**：
```bash
# 查找数据集路径
find ~ -name "lerobot_zihao_dataset_a" -type d 2>/dev/null

# 或者使用 HuggingFace 缓存默认路径
ls -la ~/.cache/huggingface/lerobot/
```

## 5. 完整工作流程总结

1. **准备阶段**：
   - 安装 LeRobot 和依赖
   - 校准机械臂和摄像头
   - 测试遥操作功能

2. **数据集初始化**：
   - 运行单 episode 采集创建数据集目录
   - 创建 `meta/tasks.jsonl` 预定义任务列表

3. **批量数据采集**：
   - 使用三摄像头配置
   - 按任务分组采集数据
   - 确保任务描述一致性

4. **质量验证**：
   - 检查数据集结构
   - 验证任务匹配
   - 查看视频质量

5. **模型训练**：
   - 使用 π0.5 预训练模型
   - 配置训练参数
   - 监控训练效果

## 6. 参考资源

- [LeRobotDataset v3.0 官方文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [摄像头配置文档](https://huggingface.co/docs/lerobot/cameras)
- [附件：LeRobot 预定义列表与批量标注完整流程](file:///C:/Users/Banana/AppData/Roaming/WPS%20灵犀/paste/LeRobot预定义列表与批量标注完整流程_20260421_124252_d78143cf.docx)