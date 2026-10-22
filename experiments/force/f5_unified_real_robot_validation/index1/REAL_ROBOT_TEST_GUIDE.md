# F5：四模型统一真机测试指南

本指南按 `docc/指南/真机测试经验与代码指令参考.md` 的直接运行方式设计。
本轮只准备了代码、精确模型包和离线检查，**尚未运行任何真机试次，F5_COMPLETE=false**。
不需要Stage0、shadow、代码指纹审批、STOP解锁，也不自动重训或增加试次。

## 1. 本次测试哪四个模型

| 指令中的模型名 | 全局模型 | 局部分支 | 用途 |
|---|---|---|---|
| P1-Hist | 历史P1/M0，8D位置+load+current | 无 | 历史实用参考，非严格因果消融 |
| Strong-G | G-Load global seed3/80k | 无 | 最强全局load单路 |
| L-Load | 既有6D无load全局 | F4.1 local seed4/9k | 局部load单路 |
| New-GL | 与Strong-G完全相同 | F4.2 local seed3/7k | 主候选，双路load |

不测试Old-GL、旧G seed1、独立B0或其它变体。L-Load内部必须包含其无load全局组件，这不等于另加一个B0试次。
精确权重身份由部署包自动核验，不需要人工填写哈希或审批文件。

## 2. 把代码与部署包放到连接机械臂的系统

所有真机指令在连接了机械臂和摄像头的Linux电脑执行，**不是在云服务器执行**。
云服务器不能通过本指南直接访问本机 `/dev/ttyACM0`。
F5模型与配套文件发布仓库：`https://huggingface.co/QYyyyyyyy/F5_UNIFIED_SO101`。
项目代码使用GitHub的`det`分支；旧F3模型包不包含本轮四个冻结候选。

推荐从Hugging Face下载，不依赖云服务器继续开机：

```bash
mkdir -p outputs/deploy
hf download QYyyyyyyy/F5_UNIFIED_SO101 F5_deploy_bundle.tar.gz F5_project_files.tar.gz --local-dir outputs/deploy
# 先保存同名文件的本机修改，或在项目副本中解包。
tar -xzf outputs/deploy/F5_project_files.tar.gz -C .
tar -xzf outputs/deploy/F5_deploy_bundle.tar.gz -C outputs/deploy
```

`huggingface.co` 直连被重置时，用镜像重跑同一命令即可（仓库公开，无需登录）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download QYyyyyyyy/F5_UNIFIED_SO101 F5_deploy_bundle.tar.gz F5_project_files.tar.gz --local-dir outputs/deploy
```

镜像返回的 revision 应仍是 `4d680b2552c0257f3a9989bde45e54bedb98aa5c`；下载后必须照第2节末尾核对字节数与 SHA256，再解包。

若Hugging Face暂不可达，也可从项目根目录使用服务器备用路径（SSH密码交互输入，文档不保存密码）：

```bash
mkdir -p outputs/deploy
scp -P 47382 root@connect.westd.seetacloud.com:/root/autodl-tmp/F5_20261022/F5_project_files.tar.gz outputs/deploy/
scp -P 47382 root@connect.westd.seetacloud.com:/root/autodl-tmp/F5_20261022/F5_deploy_bundle.tar.gz outputs/deploy/
# 先保存本机同名文件的未提交修改；建议在项目副本中解包，避免覆盖个人改动。
tar -xzf outputs/deploy/F5_project_files.tar.gz -C .
tar -xzf outputs/deploy/F5_deploy_bundle.tar.gz -C outputs/deploy
```

模型包解压后为 `outputs/deploy/F5/`，包括三套全局权重、两套局部权重、normalizer、prototype和离线DINO/ResNet缓存。
Strong-G与New-GL共享同一份全局权重，不是两份不同训练结果。
模型包约962,473,497字节，SHA256：
`9187d7e8db452fa468c3d6930558fb1e6ba9d313bdd52a0a454cf3c6c6f0cca9`。

```bash
echo '9187d7e8db452fa468c3d6930558fb1e6ba9d313bdd52a0a454cf3c6c6f0cca9  outputs/deploy/F5_deploy_bundle.tar.gz' | sha256sum -c -
python -m pip install -e .
python src/lerobot/scripts/control_f5.py --help
```

使用已有LeRobot环境，不需要为本轮强制升级torch/CUDA。代码依赖项目已有的F3预处理模块、OpenCV、pynput、PyYAML；没有连接机器人时也能看help。
如本机项目过旧、缺 `src/lerobot/policies/act/f3_runtime.py`，先同步当前完整项目；不要用旧代码运行新模型包。
服务器GPU检查仅证明模型路由/状态/归一化/残差/重置等离线工程行为，不是这台真机的标定或安全证明。

## 3. 一次完成基本检查，不做额外shadow试次

- 从臂：`/dev/ttyACM0`、`so101_follower`、`robot.id=nn`；必须使用与训练设备一致的机械臂标定/关节语义。
  **现场先确认标定 id**：本机同时存在 `nn.json`（homing≈−1900、gripper range `[1526,2978]`）与 `nnn.json`（homing≈0–100、gripper range `[2043,3493]`），两者关节语义不同。代码在文件存在时会静默按 `--robot.id` 加载，不会报错；formal3 采数用 `nnn`，F3 r1 真机冻结配置用 `nn`。以本次实物当前标定为准，开工前统一改四条指令。
- 主臂只用于试次间手动复位：`so101_leader`、`/dev/ttyACM1`、`teleop.id=nn`。策略推理不需要主臂输入，也不连接主臂串口。
- top相机index4，gripper相机index2，均RGB640×480/30fps；腕部MJPG。用实际画面确认二者没有接反，视角保持和数据采集一致。
- 若相机索引/串口在本机不同，在**正式测试前**统一修改四条指令；不能看到某个模型失败后换视角。
- 启动时报 `TimeoutError: Timed out waiting for frame from camera OpenCVCamera(4/2)`：多半是上一次进程刚退出、摄像头还没释放或刚被重新枚举。等 2–3 秒直接重跑（同一试次会自动把旧目录移到 `_prev_`）；连续失败时先用 `python -c "import cv2;c=cv2.VideoCapture(4);print(c.isOpened(),c.read()[0]);c.release()"` 逐个确认索引能出帧，再查 USB 连接。
- 已有标定文件必须存在；脚本 `calibrate=False`，不会自动进入校准或运动归零。
- 通过以前能正常使用的主从臂手动控制，确认Home+Open和开闭方向。不要在方向不明时让策略试探。
- 策略输出的是既有原生动作坐标；local的正负是该坐标内的修正，不是力值。不能凭load推断牛顿。

试次间复位如需主臂，先结束策略释放串口，按已有采集方法慢控回Home+Open；两份控制程序不能同时占用从臂：

```bash
python src/lerobot/scripts/lerobot_teleoperate.py \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=nn
```

复位完停止teleop，再启动下一条策略指令。无需反复重启电脑、机械臂供电或做shadow审批。
断开程序可能释放力矩，停止前用安全方式支撑机械臂、杯子，避免坠落/洒水。保持物理急停可用。

## 4. 开测前固定三个点位与共同参数

在已验证的熟悉工作区内选A/B/C，**不要求扩大范围，不臆造新厘米坐标**。拍一张同时能看到三点与机械臂的照片，记录在本次session。
同杯型、同水位、同相机/桌面、同任务；所有模型从一致Home+Open开始。
点位不能在模型失败后移动，也不要把杯子临时往夹爪里推后当作自主成功。

建议在session内保存 `positions.md`：

```text
A：相对固定桌边/参考物的位置说明
B：同上
C：同上
杯子/水位：……
Home+Open复位方式：……
照片文件：positions.jpg
```

默认沿用参考指令：`n_action_steps=1`、`temporal_ensemble_coeff=0.01`、`max_relative_target=5`、30fps、每试次最多60秒。
**ensemble与relative限制都可在末尾命令行直接修改，不在内部写死。** 例如：

```text
--policy.temporal_ensemble_coeff=0.02
--robot.max_relative_target=3
```

脚本把实际值写入每个trial的configuration.json，并真正传给ACT和机器人。
先统一确定参数，再开始12次正式测试；不要只给某个模型调参。正式开始后改平滑/限幅/权重/杯位需另立实验，不拿失败结果现场调参救场。
`max_relative_target`约束每步目标与当前位置差，不是绝对速度上限或力控；relative限制仍可能改变原始模型动作。
`nominal`来自全局时序平滑和反归一化，local仅加一次，然后才走原机器人relative限制。`final_gripper`与真正送出的`sent.gripper.pos`可能不同。

## 5. 固定12次顺序

| 轮次 | 设置点位变量 | 按顺序复制末尾哪些模型指令 |
|---|---|---|
| A | `export F5_POSITION=A` | P1-Hist → Strong-G → L-Load → New-GL |
| B | `export F5_POSITION=B` | Strong-G → L-Load → New-GL → P1-Hist |
| C | `export F5_POSITION=C` | L-Load → New-GL → P1-Hist → Strong-G |

每条指令只运行一个试次；总计每模型3次、全轮12次。不要把dataset.num_episodes改为3，也不要自动重跑失败trial。
每次开始前人工复位Home+Open、放杯于本轮固定点，确认周围无遮挡/线缆/易损物；预留至少30秒手动复位时间。
`reset_time_s=30`记录的是复位时间约定，脚本**不会自动运动复位，也不会自动启动下一episode**。

## 6. 自主A、辅助B、结束与急停

每次启动默认是自主A阶段。完整动作应有：Approach → Geometry → Close → Lift → Hold → Place → Release。

- 右箭头或输入 `end` 并回车：提前正常结束，记OPERATOR_END；并不自动记成功。
- Ctrl+C或Esc：立即停止控制循环并收尾，记SAFETY_ABORT，不生成STOP锁。卡死或危险时以物理急停/断电优先。
- 自主已经明确失败、准备开始辅助时，**先输入 `assist` 并回车**。这意味着操作员确认A失败，后续都是B，不能计作自主成功。
- B可轻移杯子到可抓区域等，用于观察下游链条；不重置policy时序状态、不新增trial、不延长60秒预算。
- assist不会暂停运动。任何需要把手伸入机械臂/夹爪运动区域的辅助都不应直接做；先急停，安全优先。不要为保留“恢复”录像冒碰撞风险。
- 先人为动杯再补标记会污染自主区间，应在review里记录，无法分清时用UNKNOWN/OPERATOR_INVALID，不能倒填自主成功。

终端加tee仍支持 `assist`/`end` + Enter；桌面环境可用箭头/Esc，SSH/无桌面时箭头监听可能不可用，但Ctrl+C和上述行命令仍可用。
程序每次启动/结束均reset策略及适用的pre/postprocessor；A→B不reset。SIGKILL/断电无法保证Python finally执行，届时保留残缺证据，不擅自当作没做过试次。
重复运行同一试次**不会被阻止**：上一次的目录会自动改名为 `<TRIAL_ID>_prev_<时间戳>`，新一次写入原来的 `<TRIAL_ID>`。旧录像/CSV/summary 都保留在 `_prev_` 目录里，但不会计入汇总表；不需要为了重测手动删除文件。不要改动TRIAL_ID来规避12次正式预算；硬件中断是否补测应另请规划方决定。

**结果不得覆盖，必须可重复**：每次试次的 CSV、视频、summary、stage_review 都是独立证据，只允许归档到 `_prev_`，不允许删除、覆盖或用“最好的一次”替换先前记录。可重复性是本轮的硬要求——同一模型/点位按计划重复执行，每次单独判定；报告只能基于全部已尝试试次，不能只保留成功的那一次。

## 7. 每次保存什么，如何人工判定

例：`outputs/f5/session_01/F5_NewGL_A_01/`。

- `F5_NewGL_A_01.csv`：每步MODEL_ID/TRIAL_ID/POSITION/phase、六维state/nominal/requested/sent、load/current、nominal_gripper/delta_gripper/final_gripper。
- `F5_NewGL_A_01_top.mp4`、`..._gripper.mp4`：同ID录像；首帧jpg；手机录像也按同一ID命名。
- configuration.json、summary.json、assistance_event.json（仅确实按assist才出现）。
- stage_review.json：人工依据视频填写的阶段表，不会自动把时间结束当成功。
- L/New另存local_regulator_metrics.json：全程及A/B分别的mean/median/max绝对残差、近界比例、非零比例、符号变化率。
- session根目录trial_manifest.csv记录真实已尝试试次；终端tee保存同ID stdout日志。

`sent`是机器人接口返回的限幅后目标，不是电机寄存器回读或杯子实际运动；state是发本步动作前的观测。
不要求视频与CSV亚秒级对齐，只要求模型、试次、点位身份一致。视频为名义30fps，不冒称精确墙钟。

每次或整轮结束后打开对应stage_review.json，把A/B的S0–S6填为PASS/FAIL/UNKNOWN：

| 阶段 | 判定对象 |
|---|---|
| S0 Approach | 到杯子附近 |
| S1 Grasp Geometry | 杯子进入合理的双指包夹区域 |
| S2 Close | 夹爪实际充分闭合、尝试建立夹持，不只是发了目标 |
| S3 Lift | 杯子确实随夹爪离桌，不是手帮着托起 |
| S4 Hold | 预期保持段稳定、不明显滑落；看不清就UNKNOWN |
| S5 Place | 杯子返回/移动到规定放置区域 |
| S6 Release | 放置后合理松开 |

自主全链通过才将A.full_autonomous_success写PASS；用过assist则A固定FAIL，B.full_chain_after_assistance另填。
辅助后的“全下游恢复”不是要求重复走一遍已经结束的Approach：首次失败在S0/S1时检查B的S2–S6；失败在S2或以后时检查首次失败阶段到S6，均观察到PASS才记恢复PASS。没发生的B前序阶段保留UNKNOWN，不虚填PASS。
如没有B，B.phase_used保持false，B阶段UNKNOWN不代表B失败。
另填可见滑移、早松、过闭、杯体变形、急停、辅助类型/首次失败阶段，以及可靠可测时的高度/保持时长；看不清不猜数值。
first_autonomous_failure_stage填写S0–S6之一；visible_slip等可见现象用JSON的true/false或字符串"UNKNOWN"。
validity可用VALID/HARDWARE_INVALID/OPERATOR_INVALID/TASK_FAILURE；抓取失败不等于无效数据，也不能删除该trial。

```bash
python scripts/summarize_f5.py --session-dir "$PWD/outputs/f5/session_01"
```

导出阶段、x/3自主通过、辅助与分阶段残差CSV。n=3只作描述，不能写统计显著优胜。
汇总脚本不自动批准最终科学结论；F5_COMPLETE和Preferred模型仍需完整12次及证据审查。

## 8. 开始正式测试前的终端设置

切到项目根目录，激活已有LeRobot环境，确认你拿到的是F5包：

```bash
mkdir -p "$PWD/outputs/f5/session_01"
export F5_POSITION=A
```

下面每条都是完整独立命令，复制一条即运行一次。A轮按1→2→3→4；B轮先改`export F5_POSITION=B`再按2→3→4→1；C轮改C再按3→4→1→2。
四条里的ensemble和relative参数都可以直接人工修改，但必须开测前统一修改。

## 9. 最后：四个模型的完整运行指令

### 1）P1-Hist

```bash
python src/lerobot/scripts/control_f5.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/f5/session_01" \
  --execute --model=P1-Hist --position="${F5_POSITION:?先export F5_POSITION=A或B或C}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/f5/session_01/F5_P1Hist_${F5_POSITION}_01.stdout.log"
```

### 2）Strong-G

```bash
python src/lerobot/scripts/control_f5.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/f5/session_01" \
  --execute --model=Strong-G --position="${F5_POSITION:?先export F5_POSITION=A或B或C}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/f5/session_01/F5_StrongG_${F5_POSITION}_01.stdout.log"
```

### 3）L-Load

```bash
python src/lerobot/scripts/control_f5.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/f5/session_01" \
  --execute --model=L-Load --position="${F5_POSITION:?先export F5_POSITION=A或B或C}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/f5/session_01/F5_LLoad_${F5_POSITION}_01.stdout.log"
```

### 4）New-GL

```bash
python src/lerobot/scripts/control_f5.py \
  --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/f5/session_01" \
  --execute --model=New-GL --position="${F5_POSITION:?先export F5_POSITION=A或B或C}" \
  --policy.device=cuda --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Cup pick and place" --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=30 --dataset.fps=30 \
  2>&1 | tee "$PWD/outputs/f5/session_01/F5_NewGL_${F5_POSITION}_01.stdout.log"
```
