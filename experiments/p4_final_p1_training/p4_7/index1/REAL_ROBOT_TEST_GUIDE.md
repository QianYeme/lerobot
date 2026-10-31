# P4.7 真机测试指南

本轮只准备发布与测试工具；尚未进行任何真机运动。各repo实际revision见result/upload/upload_receipts.json；未见该receipt和下载验证PASS前，不将仓库候选名称当作已发布。

## 1. 下载与环境

在连接机械臂的Linux控制主机运行。使用现有项目Python环境，依赖应与服务器记录的Python3.12.3、torch2.8.0、torchvision0.23.0、torchcodec0.7.0及requirements-server.txt核对；不要直接覆盖已有可用CUDA环境。私有库先用`hf auth login`登录有读权限账户，凭据不要写进脚本。模型repo包含匹配source/src、推理入口和离线backbone，不使用另一个环境中安装的策略源码。

依据receipt逐个下载，必须指定冻结revision：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="QYyyyyyyy/lerobot-p45-vanilla-act-s1", revision="替换为receipt中的commit", local_dir="outputs/deploy/P47/VACT-S1")'
```

也可以使用交付包的统一下载器（先在当前Python环境安装huggingface_hub、完成hf auth login），按receipt自动下载七个冻结revision并逐文件SHA核验：

```bash
python tools/p47_download.py --receipts=result/upload/upload_receipts.json --root=outputs/deploy/P47
```

在项目工作区运行时将tools替换成scripts、receipts替换为本轮index1/result/upload/upload_receipts.json。可以追加`--models L1 VACT-S1`只下载指定模型。下载器不会读取机器人标定或操作机械臂。

其余六模型同理，下载到各模型名目录。不要从result ZIP加载权重。各包checksums.json用于下载内容SHA核验；独立下载/加载预检结果见result/upload。


### 实际私有发布revision（已独立下载验SHA）

|模型|仓库|revision|
|---|---|---|
|L1|[QYyyyyyyy/lerobot-p4-l1-legacy-p1](https://huggingface.co/QYyyyyyyy/lerobot-p4-l1-legacy-p1)|fa5eb1d861c649f296e854da2fb16642f434329f|
|F3|[QYyyyyyyy/lerobot-p4-f3-fixed-p1](https://huggingface.co/QYyyyyyyy/lerobot-p4-f3-fixed-p1)|d768d3e0d0ae497e758fe11555d66ead34d57526|
|A1|[QYyyyyyyy/lerobot-p4-a1-act-matched](https://huggingface.co/QYyyyyyyy/lerobot-p4-a1-act-matched)|d1f89fc240fa6436d7a5ea4a0ddca1ece763f6c1|
|VACT-S1|[QYyyyyyyy/lerobot-p45-vanilla-act-s1](https://huggingface.co/QYyyyyyyy/lerobot-p45-vanilla-act-s1)|8203d2acfb03edfc1a76afce6990cb0942d14649|
|VDP-S1|[QYyyyyyyy/lerobot-p45-vanilla-diffusion-s1](https://huggingface.co/QYyyyyyyy/lerobot-p45-vanilla-diffusion-s1)|3f9102dd1c58ec6004a07b44784e730bf9b1716e|
|NG-L-GL|[QYyyyyyyy/lerobot-p45-newgl-legacy-gl](https://huggingface.co/QYyyyyyyy/lerobot-p45-newgl-legacy-gl)|a7c0deeaca4a700023f75db3317cd230a758bd3b|
|NG-F-GL|[QYyyyyyyy/lerobot-p45-newgl-fixed-gl](https://huggingface.co/QYyyyyyyy/lerobot-p45-newgl-fixed-gl)|dcf3af3239a0c1fa65bcab4d3d2eaa7c42bad286|

## 2. 现场设置

从臂SO101，`/dev/ttyACM0`、`robot.id=nn`；同一套nn标定，arm degrees，夹爪0–100；Top=4、gripper=2，640×480/30fps。主臂若用于复位：SO101 leader `/dev/ttyACM1`、id=nn；复位结束后关闭teleop，再运行策略。每次手动Home+Open，固定杯位、目标区、相机方向/曝光/白平衡、灯具档位与水量，记录照度或灯具设置。

所有ACT显式`n_action_steps=1`、`temporal_ensemble_coeff=0.01`是部署覆盖参数，可人工修改但比较时须一致并留档；不声称它们是stock ACT训练默认。`max_relative_target=5`始终在命令可调，不写死在模型内部。Diffusion不传ACT这两个参数，保留原生2帧观测、16预测、8执行、100去噪，实际延迟和超时记录在控制CSV；不能承诺30Hz闭环。

## 3. 按顺序执行

1. 未接机械臂时，先将下面命令的`--execute`换为`--dry-run`；这是合成零观测smoke，不是真实相机或机械臂验收。
2. 接入后现场确认相机/标定/方向。核心L1/F3/A1/VACT/VDP各做明亮半杯水2次smoke；失败就停止该模型并诊断，不混入正式统计。
3. NewGL先对同一Global的Local OFF/ON分别预检。Legacy必须60k Global+10k Local；Fixed必须100k Global+10k Local。Local仅在全局时序集成和反归一化之后修正夹爪一次，arm输出保持一致；不把修正值当真实力。各2次正常smoke，未过不扩展。
4. 前导有杯60次：5核心模型×2光照×2水量×3位置×1重复。按条件与位置平衡交替模型，不根据中途效果只给某模型增加样本。标准完整设计36次/模型，核心180次；NewGL两种预处理×OFF/ON四条件额外144次，共上限324，分阶段分日。
5. 无杯只用`--shadow-no-cup --water=absent`，程序读取观测/记录预测但没有send_action调用；不得执行无杯预测，也不计入任务成功率。2光照×3空场摆位×3重复/部署模型。

正式主终点为自主Grab→Lift→Hold→Transport→Place→Release且无洒水；记录各阶段、首次失败、空抓、停滞、重复闭合、目标误差和人为干预。Ctrl+C/Esc中断，右箭头正常结束；中断不能自动解释为失败或完整成功。人工协助单列，不计自主成功。无新的Stage0/指纹/STOP锁；同名试次自动归档旧目录。碰撞、掉杯、洒水、通信异常立即人工停止，急停/断电可触及。

`--light`、`--water`、`--position A/B/C`、`--repeat`与`--trial-id`标明环境和试次。配置文件记录模型source hash；repo revision从下载receipt关联。视频/CSV与逐次人工判定分开保存，不自动生成成功率。位置A/B/C对应现场固定三个杯位，先拍照并记录实际坐标，不沿用旧六位置定义。

### 无杯的 object-path 诊断

若要同时记录DINO全局/局部中心与confidence、patch匹配强度和argmax，使用交付包tools/p47_shadow.py（发布补充后各模型repo根目录也提供该文件）。它对实际tokenizer输出加只读hook，记录真实策略所用的预处理路径，不额外运行检测器，不调用send_action。无DINO模型明确标记object_branch_present=false；不能据静止输出断言“识别出无杯”。

```bash
python outputs/deploy/P47/L1/p47_shadow.py \
  --bundle=outputs/deploy/P47/L1 --output=outputs/p47/shadow/L1_bright_A_R1.jsonl \
  --light=bright --position=A --repeat=1 --seconds=10 \
  --robot.port=/dev/ttyACM0 --robot.id=nn --top=4 --gripper=2
```

## 4. 七个模型的直接测试命令

以下每条仅执行一轮明亮半杯水A位置smoke；现场监督后人工运行，不自动串联机械臂。测试其它条件时修改light/water/position/repeat与trial-id。NewGL默认下面启用Local；严格OFF对照删去`--local-on`并将试次ID的ON改为OFF。无杯改为shadow-no-cup和water=absent。

### L1

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/L1/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/L1" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=L1 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_L1_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/L1_$(date +%Y%m%d_%H%M%S).log"
```

### F3

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/F3/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/F3" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=F3 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_F3_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/F3_$(date +%Y%m%d_%H%M%S).log"
```

### A1

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/A1/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/A1" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=A1 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_A1_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/A1_$(date +%Y%m%d_%H%M%S).log"
```

### VACT-S1

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/VACT-S1/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/VACT-S1" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=VACT-S1 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_VACT-S1_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/VACT-S1_$(date +%Y%m%d_%H%M%S).log"
```

### VDP-S1

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/VDP-S1/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/VDP-S1" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=VDP-S1 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_VDP-S1_bright_half_A_R1 \
  --policy.device=cuda \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/VDP-S1_$(date +%Y%m%d_%H%M%S).log"
```

### NG-L-GL

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/NG-L-GL/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/NG-L-GL" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=NG-L-GL --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_NG-L-GL_bright_half_A_R1_ON \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --local-on \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/NG-L-GL_$(date +%Y%m%d_%H%M%S).log"
```

### NG-F-GL

```bash
mkdir -p outputs/p47/session_01
python outputs/deploy/P47/NG-F-GL/control_p47.py \
  --bundle="$PWD/outputs/deploy/P47/NG-F-GL" \
  --session-dir="$PWD/outputs/p47/session_01" \
  --execute --model=NG-F-GL --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P47_NG-F-GL_bright_half_A_R1_ON \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --local-on \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p47/session_01/NG-F-GL_$(date +%Y%m%d_%H%M%S).log"
```

## P4.9 Gate+ 补充

新增formal6 Legacy seed1与Fixed seed3两个80k候选，不替换历史L1/F3。新模型尚未真机测试。下载和两条直接推理指令见[补充指南](../../p4_9_unified_gate_init/index1/REAL_ROBOT_TEST_GUIDE.md)。
