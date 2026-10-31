# P4.9 formal6 Gate+ 真机测试补充指南

本轮两模型均为80k：Legacy seed1 Gate+ 配对历史L1；Fixed seed3 Gate+ 配对历史F3。只改变可学习Gate初始值0→+.001，部署导出训练结束后的实际Gate，不重置。没有新的真机结果，不把离线MAE当作抓取成功。Legacy与Fixed的seed不同，不作预处理单变量胜负比较。

沿用P4.7已验证的独立运行包。包内role仍为L1/F3以兼容8D状态与预处理分支，但新模型身份由仓库、job、checkpoint SHA和trial ID区分，绝非覆盖原L1/F3。Legacy维持DINO双重标准化；Fixed维持原始Top单次标准化，勿交换脚本/权重。

## 准备与记录

在连接机械臂的Linux主机运行，服务器不连接真机。相机gripper=2、top=4，640×480/30fps；从臂/dev/ttyACM0、robot.id=nn，use_degrees=true沿用之前的nn标定；主臂如用于手动复位，为/dev/ttyACM1、teleop.id=nn。自主推理无需连接主臂。每次人工摆回起始姿态并张开夹爪，不新增Home数值/容差门。

先在原项目环境登录HF（私有仓库），再下载；不在脚本或日志写token：

```bash
hf auth login
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('QYyyyyyyy/lerobot-p49-legacy-s1-gateplus-80k', local_dir='outputs/deploy/P49/L1_GATEPLUS')
snapshot_download('QYyyyyyyy/lerobot-p49-fixed-s3-gateplus-80k', local_dir='outputs/deploy/P49/F3_GATEPLUS')
PY
```

正式上传revision以release/result/upload/upload_receipts.json为准，复测可在snapshot_download中指定该revision。包内包含权重、配置、Train90统计/原型、源码、DINO依赖与运行入口；无需下载训练视频。沿用已有项目依赖，不强行升级环境。

与对应历史Gate0交替测试，同杯位/光照/水量和手动初始姿态。先少量有人监督测试，再依decision规划配对试次。重点记录：对准、闭爪、抬升、保持、搬运、释放及首个失败阶段；干预另记，不算自主成功。可提前Ctrl+C停止，截断待机不等于整次失败；记录停止原因，保留CSV/summary和可选视频。

不设额外指纹、shadow、文件锁或Home审核门。发生碰撞、挤压杯子、掉杯、洒水或失控立即停止。n_action_steps、temporal_ensemble_coeff、max_relative_target均在命令行显式给出，可人工调整；配对比较时两模型保持相同值。本指南不新增真实测试结果。

## 可直接复制的推理指令

每次改position/repeat/trial-id区分试次；默认A、bright、half。末尾两条分别运行，勿同时控制同一机械臂。

### L1_GATEPLUS（80k）

```bash
mkdir -p outputs/p49/session_01
python outputs/deploy/P49/L1_GATEPLUS/control_p47.py \
  --bundle="$PWD/outputs/deploy/P49/L1_GATEPLUS" \
  --session-dir="$PWD/outputs/p49/session_01" \
  --execute --model=L1 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P49_L1_GATEPLUS_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p49/session_01/L1_GATEPLUS_$(date +%Y%m%d_%H%M%S).log"
```

### F3_GATEPLUS（80k）

```bash
mkdir -p outputs/p49/session_01
python outputs/deploy/P49/F3_GATEPLUS/control_p47.py \
  --bundle="$PWD/outputs/deploy/P49/F3_GATEPLUS" \
  --session-dir="$PWD/outputs/p49/session_01" \
  --execute --model=F3 --position=A --repeat=1 \
  --light=bright --water=half --trial-id=P49_F3_GATEPLUS_bright_half_A_R1 \
  --policy.device=cuda \
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.use_degrees=true --robot.max_relative_target=5 \
  --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  2>&1 | tee "outputs/p49/session_01/F3_GATEPLUS_$(date +%Y%m%d_%H%M%S).log"
```
