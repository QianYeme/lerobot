# F3 r1 四模型真机对照测试指南

对应合同：`decision_F3_REAL_ROBOT_MULTI_MODEL_COMPARISON_r1.md`。
离线验收见`result/OFFLINE_ACCEPTANCE.md`：TF32原batch16参考通过，但batch1与batch16存在约0.0177数值差；同batch下新适配器与原F2.3实现完全一致。不得换参考或放宽容差来制造通过，目标设备仍须单独验证。
本次编写指南并实现统一入口，**没有启动机器人、采集试次或修改部署参数**。

> 当前完成模型发布与服务器离线检查，尚未完成新系统部署验证。
> 四模型使用新增的 **`src/lerobot/scripts/control_f3.py`**，不要再拼接旧
> `control_act_det.py`。最后一节给出真实CLI；发送动作前仍须通过第5节人工门。

## 1. 这次究竟测什么

| 模型 | 冻结模型来源 | 全局输入 | 局部修正 |
|---|---|---|---|
| M0 | 历史P1_ocfix2_s1，100k | RGB＋8D位置/load/curr | 无 |
| M1 | F2.1 A0 retry1，80k | RGB＋6D位置 | 无 |
| M2 | F2.1 A1，60k | RGB＋6D位置，另加单帧load token | 无 |
| M3 | **与M1完全相同的全局模型**＋F2.3 C1-load，10k | 全局仅RGB＋6D位置 | load/夹爪位置/nominal指令→小MLP |

主要比较M3 vs M2，辅助比较M2 vs M1、M3 vs M1；M0仅为历史参照。
M0不是纯ACT，不能将M0 vs M1的变化全部归因于load。

本轮不测W15、B1门控、C1-no-load、FiLM，不换checkpoint、不训练。
沿用熟悉的同一杯型、水量、场景、任务和杯位放置协议；不增加点位、
平移距离或空间泛化要求。不能照搬旧G1的81次/位置分层计划。

## 2. 从上次指南继承哪些参数

参数来源：

- `experiments/position/g1_real_robot_generalization/index1/result/REAL_ROBOT_TEST_GUIDE.md`，
  第3节主从预检、第9节推理示例。
- `docc/指南/lerobot_annotation_implementation_guide.md`，旧采集/推理示例。
- 本轮发布的`result/model_identity.json`，四组实际checkpoint配置。

这些是历史示例，不是“对任意新机器均已验证的最优参数”。

| 项目 | 本次准备采用的历史值/建议值 | 上机前检查 |
|---|---|---|
| 控制主机 | 机械臂/相机直接连接的本地Linux主机 | 不用云端SSH控制实机 |
| 从臂 | `so101_follower`、`/dev/ttyACM0`、`id=nn` | 串口和标定身份不能只按编号猜 |
| 主臂 | `so101_leader`、`/dev/ttyACM1`、`id=nn` | 只用于预检/安全人工复位 |
| 顶部相机 | 名称`top`，历史索引4 | 确认真的是桌面视角 |
| 腕部相机 | 名称`gripper`，历史索引2 | 确认真的是夹爪视角 |
| 相机输入 | 640×480、30 FPS | 保持训练语义/方向/裁剪一致 |
| 控制目标频率 | 30 Hz | 记录实际循环Hz、延迟和漏帧 |
| `n_action_steps` | 1 | 四组当前checkpoint均为1 |
| `chunk_size` | 100 | 四组当前checkpoint均为100，不修改 |
| `temporal_ensemble_coeff` | `0.01`，现场启用ensemble | 四组checkpoint均为null；**入口默认已改为0.01**，漏带不再退化为抖动原始动作，真机卡顿现场仍显式传0.01 |
| `max_relative_target` | 2 | 真机卡顿，现场统一收紧为2 |
| 单次时间上限 | 60秒 | 不是必须等满60秒；危险立即停 |
| 人工复位预留时间 | 约30秒 | 安全复位/检查未完成不得自动续跑 |
| task文本 | `Cup pick and place`（沿用旧示例） | 实际任务须与熟悉任务一致，不改场景目标 |
| M3残差上限 | ±4.832699459791184 | 原生夹爪位置单位，不能放大 |

真机首测观察到动作卡顿，**现场统一**将`temporal_ensemble_coeff`设为`0.01`、
`max_relative_target`收紧为`2`，四模型一致，对比仍公平；已重新冻结并过Stage0审核。
这两项不是某组单独调参，也不能看到结果后再继续改。

卡顿根因是入口旧默认`temporal_ensemble_coeff=None`会静默关闭ensemble，
每步只吐chunk第一个原始预测（`n_action_steps=1`），逐帧抖动；G1流畅正是因为
同一参数被真正开启。现已把入口默认改为`0.01`，并在`None`时打警告。

`max_relative_target=2`不是厘米、速度上限或防碰撞保证；不能因动作卡顿
就删除限幅。M3的±4.8327是**额外夹爪残差上限**，不能替代机器人安全限幅。

## 3. 下载模型与环境准备（无硬件动作）

所有命令在本地Linux部署主机执行。按实际位置修改项目路径，不要照抄
旧机器的`/home/lyj/...`绝对路径。以下不执行`git reset`或覆盖现有工作区。

```bash
cd ~/lerobot/lerobot-main
git status --short
git branch --show-current
git rev-parse HEAD

# 激活已准备好的环境，二选一，按实际环境执行：
# source .venv/bin/activate
# conda activate lerobot
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

hf auth login
mkdir -p models/f3_r1
hf download QYyyyyyyy/F3_R1_MULTI_MODEL_SO101 \
  --revision e903b0a1489815077f7a0ef881bec4ca007b6b18 \
  --local-dir models/f3_r1

export F3_BUNDLE="$PWD/models/f3_r1"
export TORCH_HOME="$F3_BUNDLE/torch_cache"
export PYTHONPATH="$F3_BUNDLE/source/src${PYTHONPATH:+:$PYTHONPATH}"

python "$F3_BUNDLE/source/scripts/verify_f3_bundle.py" \
  --bundle "$F3_BUNDLE" --device cuda
python "$F3_BUNDLE/source/scripts/verify_f3_multimodel_bundle.py" \
  --bundle "$F3_BUNDLE" --device cuda
```

仓库是私有的，使用本人有读取权限的账户认证；不要把token写进命令文件、
截图或日志。首次安装先参考包内`environment_versions.json`和源码依赖。
服务器验证环境Python3.12.3、torch2.8.0+cu128，不保证同一CUDA wheel适配
新系统。不要为了安装方便覆盖已可用的驱动/torch环境。

这两个脚本不打开串口/相机、不发送动作。结果仅为离线工程检查。
新GPU/CPU的TF32/FP32与算子数值差异仍需核对，不能放宽断言把错误掩盖过去。

### 3.1 包内模型路径

```bash
export M0_PATH="$F3_BUNDLE/models/M0"
export M1_PATH="$F3_BUNDLE/global"
export M2_PATH="$F3_BUNDLE/models/M2"
export M3_GLOBAL_PATH="$F3_BUNDLE/global"
export M3_LOCAL_PATH="$F3_BUNDLE/local/010000.pt"
```

M0使用自带标准processor JSON和safetensors；M1/M2/M3使用包内冻结
`normalization.json`，RGB使用ImageNet统计。全局原型文件路径需要重定位
到本机`assets/cup_prototype_bank_v1.pt`，不能沿用服务器绝对路径。

不必下载完整训练视频来推理。旧`control_act_det.py`依赖dataset metadata，
这不意味着可以拿任意旧formal1/formal4 metadata代替本轮处理规则。

## 4. 相机、标定、主从预检（仅供现场人员手动执行）

下面包括**可能使机器人运动的teleoperate命令**，不是离线校验。
仅在现场急停可用、操作者授权、运动范围清空后手动执行。本次编写指南
并未替你执行这些命令。

### 4.1 先识别设备

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM* /dev/video* 2>/dev/null || true
lerobot-find-cameras opencv
```

确认串口实际指向leader还是follower、两路相机画面与设备编号匹配。
旧编号仅作起点；不同系统编号可变。若用by-id/by-path稳定路径，先验证
配置入口支持并冻结实际字符串。不要同时开启多个程序占用同一串口/相机。

保存与`nn`及当前实物匹配的标定文件，记录存储路径和校验值。相同`id=nn`
不代表标定文件相同；不要让旧环境指南触发无必要的重新标定或改关节方向。
若硬件或标定与训练语义不兼容，停止并记录，不强行开始正式对照。
串口权限交由本机既有权限/设备管理方式处理，不复制旧指南的通配符全局chmod。

### 4.2 主臂手动控制从臂

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

只验证主从方向/画面并缓慢回到一致的Home + Open，不借此采集额外科研试次。
模型推理前彻底停止teleoperate、确认串口释放；不能让主臂和策略同时写从臂。
正式自主试次中不允许人工带动主臂“帮模型抓到杯子”。安全干预必须保留记录。

## 5. 统一入口已实现，现场Stage 0仍须完成

**代码接线已有离线验证；现场部署/安全审查不是已经通过的检查表。**

现有`control_act_det.py`的真实能力：读取dataset metadata和checkpoint的
标准processors，执行已有策略路径；目前CSV含target/sent各关节和夹爪实际值。
它没有M2独立load token接线、M3局部模式开关/全量遥测，不能只换policy.path。
新增`control_f3.py`及`f3_runtime.py`单独处理本轮路由，不改旧入口。
服务器首个真实Dev观察对照四组原管线，特征与无局部动作差均为0；
另通过假机器人CSV/视频/异常清理测试，但这些不等于实机安全验证。

开始发送动作前，应逐项核对以下已实现路径及现场条件：

1. **逐模型观察路由**：M0=8D；M1=6D；M2=6D＋单帧load；M3 global=6D、
   local=raw load＋raw gripper.pos＋native nominal gripper。
2. **特征顺序**：六位置为shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、
   wrist_roll、gripper的`.pos`；M0末两项为`gripper.load`、`gripper.curr`。
   load/current可以统一记录，但不得因记录而进入M1/M2/M3未授权输入。
3. **预处理**：匹配每组冻结统计与相机语义；禁止把局部load预归一化后重复
   传入适配器。局部nominal必须为反归一化后的实际全局候选夹爪指令。
4. **后处理顺序**：全局既定动作选择/ensemble与反归一化→M3局部修正→
   共同的机器人安全限幅→发送；其他模型经过同一安全限幅路径。
5. **三模式**：disabled不算局部模型并原样返回全局动作；shadow计算但不执行
   修正；regulated才执行。通过开关关闭后须恢复M1的同一步全局动作路径。
6. **机械臂保持**：对同一观察，在安全限幅前比较nominal/regulated的前五列
   完全不变；限幅后的比较也须经过同一个限幅函数。不能把实际不同试次的
   臂轨迹要求为相同，也不能把安全限幅造成的差异误判成局部改臂。
7. **限幅与符号**：验证夹爪方向；M3输出始终在±4.83269946内；不可把
   残差作为绝对夹爪位置发出，也不可给五个臂关节加同一残差。
8. **重置/清理**：每次trial开始调用policy.reset及存在的pre/postprocessor.reset，
   清空动作队列/ensemble；异常退出可靠停止I/O并落盘。
9. **日志与视频**：满足第8节，不用旧控制CSV代替完整F3遥测；相机首帧正常
   不能代替连续试次录像。
10. **共同配置**：填写并固定`deployment_config.json`，仍有PENDING则不进入
    shadow/正式试次。保存新系统git commit、运行参数、设备和checkpoint hash。

旧脚本的三个reset出现在每episode开始；Ctrl+C进入finally主要执行disconnect/
关闭CSV和键盘监听，**不保证再调用三个reset，也不等于硬件急停**。
重新启动进程也不会自动把机械臂回到相同Home。旧reset_time_s期间通常保持
上一目标而不是自动回Home，手动复位前要确认当前控制器所有权和电机状态安全。

新入口真实参数为`--model M0/M1/M2/M3`和`--mode disabled/shadow/regulated`，
不是旧脚本参数。`--prepare-only`和`--dry-run`都不连接机器人；只有显式
`--execute`、有效标定与人工Stage0报告匹配当前配置后才进入硬件路径。
M3 regulated还要求3个真实shadow记录及其人工审核。脚本不自动批准这些门。
本指南顺序需人工遵守；入口校验试次ID/禁止覆盖，不代替完整轮转安排的人工审核。

## 6. 现场冻结清单

在`result/deployment_config.json`的本次现场副本中填写，并保留开始前版本：

```text
code_commit / HF_revision / model+processor+normalization hashes
deployment_host / OS / Python / torch / CUDA / numerical precision settings
robot_type / robot_id / serial device identity / calibration path+hash
top_camera / gripper_camera / view screenshots / input resolution / camera FPS
target control Hz / measured latency+Hz / n_action_steps / temporal ensemble
common action limiting / max_relative_target / any existing smoothing
cup type / familiar fill state / task / familiar placement and start-pose photos
trial duration cap / reset procedure / emergency-stop method / operator
M3 local hash / input units / delta_max / shadow gate status
log root / video root / trial-ID convention / clock alignment
```

M1与M3共享同一全局文件，不能下载后分别编辑两份配置。
更改现场设备编号属于映射核验，改变视角/裁剪/分辨率/动作语义不只是改编号。
无法保持合同共同条件时，记录阻塞并停止，不能偷偷改其中一组。

## 7. 实机执行顺序：先shadow，再轮转

### 7.1 M3 shadow，仅观察、不执行残差

| 试次ID示例 | 模式 | 真实执行 |
|---|---|---|
| SHADOW_01 | M3 shadow | M1 global-only动作 |
| SHADOW_02 | M3 shadow | M1 global-only动作 |
| SHADOW_03 | M3 shadow | M1 global-only动作 |

确认全部无arm修改、NaN/Inf、越界、危险正负快速交替，near-bound不占主导，
释放时没有明显任务冲突，开关关闭路径正常，日志视频完整。
near-bound建议描述口径沿用F2.3：`abs(delta) >= 0.95*delta_max`，不是接触判据。
危险振荡/释放冲突不能只看一个平均数，必须结合逐步曲线和视频人工审核。

shadow审核PASS前禁止M3 regulated。若失败，记录具体flag并阻断M3，
不能通过改bound、ensemble、MLP、输入或补更多试次绕过。M0/M1/M2
仅在其自身安全门通过且仍符合合同的情况下继续。

### 7.2 正式试次固定轮转

| 轮次 | 固定顺序 |
|---|---|
| Round1 | R01_M0 → R01_M1 → R01_M2 → R01_M3 |
| Round2 | R02_M0 → R02_M1 → R02_M2 → R02_M3 |
| Round3 | R03_M0 → R03_M1 → R03_M2 → R03_M3 |

如果M3被shadow阻断，每轮保留M0→M1→M2，并记录M3为BLOCKED，不填失败
率的普通0、不补其他模型。不得看完结果后改顺序、换模型或增加次数。

每次只有一个控制进程：停止上次→确认停止→安全复位到同一Home + Open→
恢复熟悉杯位/水量→检查相机/硬件→重置状态→开始视频日志→启动对应模型。
不要让一个`num_episodes=3`连续跑完某模型，破坏合同轮转顺序。

历史60秒上限和30秒复位等待只是操作参考，不授权新添Stage A测试次数。
部署验证不另设重复效果试次，正式开始后不为修复结果补跑。无关硬件故障
导致的无效试次也保留，是否补做需先审核，不自动凑足“好看的3次”。

## 8. 每步日志和每次视频

建议在新目录下为每个trial建立独立文件夹，避免覆盖训练数据或旧实验：

```text
outputs/f3_r1/<session_id>/
  deployment_config_frozen.json
  model_identity.json
  SHADOW_01/
    command.txt
    control.csv
    terminal.log
    top.mp4
    gripper.mp4
    review.json
  ...
  R01_M0/
  ...
```

逐步至少记录：step/frame序号、单调时间、墙钟时间、state/action字段、
raw gripper.pos/load/curr（可用时）、
nominal五关节动作、nominal夹爪动作、预测delta、限幅delta、假设最终夹爪动作、
真实发送动作、机器人安全限幅前后值。trial/model/mode、身份与异常/停止
信息在同目录configuration.json/summary.json；实际Hz可从CSV相邻t_s计算，
不把目标dataset.fps当实际频率。CSV另有inference_s、processing_s、overrun。

M0/M1/M2没有local的字段标为NA/不适用，不能伪造局部运行记录。
M3 shadow要同时保存“假设会发送什么”和“实际仍发nominal什么”。
顶/腕视频由入口逐控制步写入。MP4使用目标30FPS作为播放时间基，若控制
循环慢于30Hz，必须用CSV的step→t_s映射还原真实事件时间，不能直接把
MP4播放秒数当保持时长；另拍视频也应记录时间对齐。不要只用tee替代录像。

测试后按合同整理trial_manifest、per_trial_metrics、model_summary、
m3_shadow_metrics/review、m3_regulator_metrics、safety_review、
mechanism_review、video_index、realtime_log_index等到本轮result。
没有采集就写NOT_RUN/BLOCKED，不填0冒充真实失败或PASS。

## 9. 人工判定与保持时间

每次立刻记录：抓持建立、拿起、可可靠测得的抬升高度、保持开始/结束/时长、
可见滑落、提前松爪、过度挤压/杯子变形、熟悉任务中适用的搬运、稳定放下、
正确释放、整链完成、紧急人工干预。拿起太低/保持太短要如实记录。

不从旧G1移植“稳定0.5秒才算Lift”等阈值作为新主指标。本轮按F3合同作
小样本描述对照，报告实际现象和可复核时间；看不清边界则记录区间/不确定，
不脑补高度、时长或接触帧。视频后审阶段标记只用于解释，不进入局部控制器。

shadow/regulated另记录mean/median/max abs(delta)、near-bound比例、
非零比例、符号变化率及主要事件的load/position/nominal/delta/final动作。
符号变化率须注明零附近处理口径，不据此发明新的在线门控。
M3不能修改臂动作，所以不能未经机制证据就把抬升/停留时长变化归功于力控。

## 10. 停止条件

出现危险过闭合、杯子明显压坏、快速夹爪振荡、反复异常收紧、调节器导致
提前释放、非有限动作、残差越界、局部代码改臂、碰撞/倾倒/泼水或硬件风险：
立即采用现场已验证的停止/急停方式，保存日志视频并记SAFETY_ABORT。

M3一旦出现调节器相关严重危险事件，停止其后所有regulated试次。
不调整参数后在本轮重试。紧急介入即使使试次无效，也保留为安全证据。
不能把Ctrl+C、Esc、右方向键或“进程已退出”当作物理急停保证。

## 11. 最后执行的测试指令（一次只运行一条）

### 11.1 共同参数：参考第一次G1真机测试

先确保GitHub det分支已更新，当前目录确实包含新入口。
`SESSION`是本次唯一会话目录，不得每换模型就换会话以绕过配置一致性检查。

```bash
cd ~/lerobot
conda activate lerobotnn
export PROJECT_ROOT="$PWD"
export F3_BUNDLE="$PROJECT_ROOT/outputs/deploy/F3_R1"
export TORCH_HOME="$F3_BUNDLE/torch_cache"
# 这里必须使用GitHub项目新入口，不能仍指向HF包的旧源码快照。
export PYTHONPATH="$PROJECT_ROOT/src"
export SESSION="$PROJECT_ROOT/outputs/f3_r1/session_01"
mkdir -p "$SESSION"
set -o pipefail

F3_COMMON=(
  --bundle "$F3_BUNDLE"
  --session-dir "$SESSION"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=nn
  --robot.max_relative_target=2
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
  --policy.device=cuda
  --policy.n_action_steps=1
  --policy.temporal_ensemble_coeff=0.01
  --dataset.single_task="Cup pick and place"
  --dataset.num_episodes=1
  --dataset.episode_time_s=60
  --dataset.reset_time_s=30
  --dataset.fps=30
)
python src/lerobot/scripts/control_f3.py --help
```

如已有标定文件不在默认目录，先核实其身份，再将共同参数加入
`--robot.calibration_dir=/实际包含nn.json的目录`，所有组相同。
不自动重新标定。30秒reset参数仅记录人工复位参考；单次调用不会自动回Home。
无需`--dataset.root`、`--dataset.repo_id`：新入口不借用旧训练数据metadata，
直接使用冻结包内processor/normalization；也不重新生成或上传数据集。
现场统一指定`temporal_ensemble_coeff=0.01`，继承前述共同设置。

### 11.2 先完成离线/现场审核，不能跳过

以下仅离线准备，不发送任何动作：

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --prepare-only
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --dry-run --model M0
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --dry-run --model M1
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --dry-run --model M2
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --dry-run --model M3 --mode shadow
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" --dry-run --model M3 --mode regulated
```

prepare会生成`deployment_config_frozen.json`、默认`passed:false`的
`stage0_review.json`与`shadow_review.json`，绑定共同参数、代码、模型和标定hash。
配置一旦变化，同一会话后续调用会拒绝运行；这不是授权换会话重新调参。

由现场人员完成第4–6节和干运行审核后，**人工**编辑
`$SESSION/stage0_review.json`：填写非空reviewer，确认deployment_fingerprint
保持不变，确实通过才将passed改true。不要仅为让命令通过而改true。
此时shadow_review仍保持false。本文不会给出自动修改审核结果的命令。

### 11.3 M3 shadow：3次，残差不执行

每条之间都先停止、复位、恢复杯位和水量。两次回车确认分别用于
连接前安全确认与首帧相机人工确认（输入非空内容即中止）；准备不充分不要确认。

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M3 --mode shadow --trial-id SHADOW_01 \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/SHADOW_01_terminal.log"

python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M3 --mode shadow --trial-id SHADOW_02 \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/SHADOW_02_terminal.log"

python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M3 --mode shadow --trial-id SHADOW_03 \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/SHADOW_03_terminal.log"
```

查看三次完整视频/CSV，按第7节审核。通过后人工填写shadow_review的reviewer、
passed=true、valid_trial_ids=["SHADOW_01","SHADOW_02","SHADOW_03"]，保留原
fingerprint。入口还会检查三条已有shadow记录、非空日志/视频和非中止状态；
这些文件检查不代替振荡、释放冲突等人工审核。
失败则保持false，记录原因并跳过M3正式命令，只做安全且授权的M0/M1/M2。

### 11.4 四模型正式命令：每轮M0→M1→M2→M3

先完成并审核shadow。Round1设置`ROUND=01`，依次运行以下四条；
Round2改为02、Round3改为03，再同顺序执行。不能把四条分别各跑3遍。
已有试次目录会拒绝覆盖，不删除目录绕过失败记录或重跑预算。

```bash
export ROUND=01
```

**M0：历史8D P1，100k**

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M0 --mode disabled --trial-id "R${ROUND}_M0" \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/R${ROUND}_M0_terminal.log"
```

**M1：干净6D A0，80k，无load/current反馈**

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M1 --mode disabled --trial-id "R${ROUND}_M1" \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/R${ROUND}_M1_terminal.log"
```

**M2：A1，60k，全局单帧load融合**

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M2 --mode disabled --trial-id "R${ROUND}_M2" \
  --stage0-report "$SESSION/stage0_review.json" \
  2>&1 | tee "$SESSION/R${ROUND}_M2_terminal.log"
```

**M3：同M1全局模型＋C1-load，10k，执行局部夹爪修正**

```bash
python src/lerobot/scripts/control_f3.py "${F3_COMMON[@]}" \
  --execute --model M3 --mode regulated --trial-id "R${ROUND}_M3" \
  --stage0-report "$SESSION/stage0_review.json" \
  --shadow-review "$SESSION/shadow_review.json" \
  2>&1 | tee "$SESSION/R${ROUND}_M3_terminal.log"
```

若M3被阻断，不运行最后一条。其他组仍须人工遵守固定轮转，不自动增加次数。
有桌面键盘监听时右方向键表示正常提前结束，Esc表示安全中止；Ctrl+C也记录
安全中止。M3 regulated发生异常/中止会留下M3_STOP.json并禁止后续M3干预。
无桌面键盘监听时不要假设方向键可用：正常可到60秒上限停止，任何风险仍立即
用现场急停；不能把Ctrl+C当作普通成功结束方式绕过记录。

启动后每次均会保留control.csv、两路MP4、首帧图、配置与summary.json。
summary的TIME_LIMIT/OPERATOR_END只表示程序结束方式，不自动代表试次有效或成功。
Ctrl+C/出错后disconnect可能卸力，夹持中的杯子及机械臂需由现场安全措施保护。
停止/断电的现场方法优先于所有软件按键。
完成后停止，交Builder/Red-Team审核；不自动扩次数、读取Final或声明统计显著。

## 12. 四模型独立复制版正式测试命令

以下每个代码块均包含完整参数和环境设置，不依赖第11节的F3_COMMON、
SESSION或ROUND变量。**必须先按第11.2节完成相同会话的准备、离线检查与人工Stage0审核；M3还必须完成三次shadow并通过人工审核。**
直接复制不等于跳过安全门；缺少审核时程序会拒绝启动。

默认项目目录为`~/lerobot`，模型包为`outputs/deploy/F3_R1`，
会话为`outputs/f3_r1/session_01`，与第11节一致。如果实际安装位置不同，
先统一修正准备步骤与下面四块的路径，不要使用另一会话来绕过检查。
相机索引4/2、从臂ACM0/id nn须现场确认；主臂ACM1仅用于手动预检/复位，
策略运行时不同时开启teleoperate。

参数参照此前G1指令：n_action_steps=1、640×480/30FPS、每次1段且最多60秒、
人工复位参考30秒；真机卡顿故现场统一`temporal_ensemble_coeff=0.01`、`max_relative_target=2`。
新入口从模型包读取归一化，不需要旧指令的dataset.repo_id/root。
不会自动回Home，两次回车确认后才可能发送动作（输入非空内容即中止）。

下面是**第1轮**，按M0→M1→M2→M3顺序，每块一次。第2轮仅将
trial-id和tee文件名中的R01统一改为R02，第3轮改为R03；
不要重复第1轮或删除已有记录。每轮间、每模型间均安全停止并人工复位。
不要在云服务器执行这些硬件命令，应在直连机械臂和相机的Linux主机上运行。

### 准备：先冻结配置 + 四模型离线干运行（无硬件；不过不能上真机）

① 冻结本次会话共同配置，② 对四模型各做一次离线前向，全部打印 `OFFLINE_PASS_NO_HARDWARE` 才算过。两步都不连接机械臂/相机、不发送动作；不先跑 ① 的话，下面 `--execute` 会直接拒绝启动。

```bash
cd ~/lerobot || exit 1
conda activate lerobotnn
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/outputs/deploy/F3_R1/torch_cache"
BUNDLE="$PWD/outputs/deploy/F3_R1"
SESSION="$PWD/outputs/f3_r1/session_01"
mkdir -p "$SESSION"
set -o pipefail

COMMON=(--bundle "$BUNDLE" --session-dir "$SESSION" --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn --robot.max_relative_target=2 \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30)

# ① 冻结本次会话共同配置（绑定 2/0.01/相机/标定 id/代码哈希；之后任何一项改动都会被 --execute 拒绝）
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --prepare-only

# ② 四模型离线干运行，全部应打印 OFFLINE_PASS_NO_HARDWARE
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --dry-run --model M0
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --dry-run --model M1
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --dry-run --model M2
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --dry-run --model M3 --mode shadow
python src/lerobot/scripts/control_f3.py "${COMMON[@]}" --dry-run --model M3 --mode regulated

```

③ 人工 Stage0 安全审核：这是**安全门，只能现场操作者做，不能跳过**。逐项确认全部满足后，才把 `passed` 改 `true`：

- 杯中约半杯水、桌面干燥；机械臂在 **Home + Open**；夹爪/底座/相机/线缆无碰撞风险，臂运动范围内无人无杂物；
- 你手能随时够到急停；两路相机画面：top=桌面全局、gripper=腕部视角（不黑屏/不过曝/不错位）；
- **先停掉 teleoperate**、确认串口释放（策略与主臂不能同时写从臂）。

```bash
# 现场确认上述安全门全部满足后执行：把 reviewer 改成你的名字，passed 改 true
python - <<'EOF'
import json
p='outputs/f3_r1/session_01/stage0_review.json'
r=json.load(open(p))
r['reviewer']='lyj'
r['passed']=True
json.dump(r,open(p,'w'),indent=2,ensure_ascii=False)
print('fingerprint 未变 =', r['deployment_fingerprint'])
EOF
```

`--execute` 会校验这个文件：`passed` 必须为 true、`reviewer` 非空、`deployment_fingerprint` 与冻结配置一致，否则拒绝启动。指纹变了就说明冻结配置被改过，须回退重审。

### M0：历史8D P1

```bash
cd ~/lerobot || exit 1
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/outputs/deploy/F3_R1/torch_cache"
mkdir -p outputs/f3_r1/session_01
set -o pipefail
python src/lerobot/scripts/control_f3.py \
  --bundle="$PWD/outputs/deploy/F3_R1" \
  --session-dir="$PWD/outputs/f3_r1/session_01" \
  --execute --model M0 --mode disabled --trial-id R01_M0 \
  --stage0-report="$PWD/outputs/f3_r1/session_01/stage0_review.json" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.max_relative_target=2 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee outputs/f3_r1/session_01/R01_M0_terminal.log
```

### M1：干净6D，无反馈

```bash
cd ~/lerobot || exit 1
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/outputs/deploy/F3_R1/torch_cache"
mkdir -p outputs/f3_r1/session_01
set -o pipefail
python src/lerobot/scripts/control_f3.py \
  --bundle="$PWD/outputs/deploy/F3_R1" \
  --session-dir="$PWD/outputs/f3_r1/session_01" \
  --execute --model M1 --mode disabled --trial-id R01_M1 \
  --stage0-report="$PWD/outputs/f3_r1/session_01/stage0_review.json" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=2 \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee outputs/f3_r1/session_01/R01_M1_terminal.log
```

### M2：全局单帧load

```bash
cd ~/lerobot || exit 1
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/outputs/deploy/F3_R1/torch_cache"
mkdir -p outputs/f3_r1/session_01
set -o pipefail
python src/lerobot/scripts/control_f3.py \
  --bundle="$PWD/outputs/deploy/F3_R1" \
  --session-dir="$PWD/outputs/f3_r1/session_01" \
  --execute --model M2 --mode disabled --trial-id R01_M2 \
  --stage0-report="$PWD/outputs/f3_r1/session_01/stage0_review.json" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=2 \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee outputs/f3_r1/session_01/R01_M2_terminal.log
```

### M3：局部load残差，必须先通过shadow

```bash
cd ~/lerobot || exit 1
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/outputs/deploy/F3_R1/torch_cache"
mkdir -p outputs/f3_r1/session_01
set -o pipefail
python src/lerobot/scripts/control_f3.py \
  --bundle="$PWD/outputs/deploy/F3_R1" \
  --session-dir="$PWD/outputs/f3_r1/session_01" \
  --execute --model M3 --mode regulated --trial-id R01_M3 \
  --stage0-report="$PWD/outputs/f3_r1/session_01/stage0_review.json" \
  --shadow-review="$PWD/outputs/f3_r1/session_01/shadow_review.json" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=2 \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee outputs/f3_r1/session_01/R01_M3_terminal.log
```

M3审核未通过则跳过M3，保留原因；不调参重试。软件退出不等于物理急停，安全处理以现场方案为准。
