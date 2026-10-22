# F5 真机前本机预检记录（2026-10-22）

范围：只做 `REAL_ROBOT_TEST_GUIDE.md` 的真机前步骤。**未连接机械臂、未连接两路相机、未执行任何真机试次**；F5_COMPLETE=false，正式试次 0/12。

## 1. 机器与网络

| 项 | 值 |
|---|---|
| 主机 | lyj-ROG-Strix-G614JV（RTX 4060 Laptop 8GB） |
| Python | 3.12.13（conda `lerobotnn`） |
| torch | 2.10.0+cu128，`cuda.is_available()=True` |
| HF 直连 | `huggingface.co` TLS 被重置（curl exit 35） |
| 实际下载通路 | `HF_ENDPOINT=https://hf-mirror.com` |

## 2. 下载与完整性

- HF 仓库 `QYyyyyyyy/F5_UNIFIED_SO101` 为公开仓库，镜像返回 revision `4d680b2552c0257f3a9989bde45e54bedb98aa5c`，与 `hf_publish_receipt.json` 记录一致。
- `F5_deploy_bundle.tar.gz`：962,473,497 字节，SHA256 `9187d7e8db452fa468c3d6930558fb1e6ba9d313bdd52a0a454cf3c6c6f0cca9`，与发布回执一致。
- `F5_project_files.tar.gz`：16,076 字节，SHA256 `8489f180f69f9695df8bd5ff9154f0a08bfc9671a564969b1692c03f3190a74d`，校验通过。
- 代码包内 5 个文件（`control_f5.py`、`f5_runtime.py`、`f5_protocol.py`、`summarize_f5.py`、`REAL_ROBOT_TEST_GUIDE.md`）与当前 `det` 仓库版本**逐字节相同**，未解包覆盖。
- 解包位置：`outputs/deploy/F5/`（993MB，gitignore 内），含 `models/{P1-Hist,Strong-G,B0-6D}`、`local/{L-Load,New-GL}.pt`、`assets/cup_prototype_bank_v1.pt`、`normalization.json`、`torch_cache`。

## 3. 代码与身份核验

- `python src/lerobot/scripts/control_f5.py --help`：exit 0。
- `pytest tests/scripts/test_f5_control.py test_f3_control_cli.py test_f3_control_protocol.py`：**20 passed**（与服务器记录条数一致）。注：`lerobotnn` 环境原本没有 pytest，本轮为跑回归安装了 pytest 9.1.1。
- `verify_bundle()` 对四个模型逐一核验 `model_identity.json`/`manifest.json` 哈希：**全部通过**。
  - P1-Hist → `models/P1-Hist`
  - Strong-G → `models/Strong-G`
  - L-Load → `models/B0-6D` + `local/L-Load.pt`
  - New-GL → `models/Strong-G` + `local/New-GL.pt`
- `BOUND = 4.832699459791184` 与身份文件一致。

## 4. 四模型离线 dry-run

命令形态（`--dry-run`，合成零观测，无机器人）：

```bash
python src/lerobot/scripts/control_f5.py --bundle="$PWD/outputs/deploy/F5" \
  --session-dir="$PWD/outputs/f5/preflight" --dry-run --model=<模型> --position=A \
  --policy.device=cuda --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01
```

| 模型 | exit | 状态 | 六维动作有限 | 夹爪 |
|---|---|---|---|---|
| P1-Hist | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 9.233 |
| Strong-G | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 19.662 |
| L-Load | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 18.458 |
| New-GL | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 14.833 |

- L-Load 相对同 global 的 `M1 disabled` F3 夹具差 −4.828825，New-GL 相对 Strong-G 差 −4.829018，均在 ±4.832699 界内；合成零观测下局部修正贴近边界，**只证明接线与限界，不是行为证据**。
- 五次非夹爪关节在不同进程间最大差 ≤0.008（`cudnn.benchmark=True` 的算法选择差异），不是权重差异。

## 5. 无法在本机完成的部分

- 无 `/dev/ttyACM*`：机械臂未连接，`connect(calibrate=False)`、Home+Open、开闭方向、夹爪低速检查均**未做**。
- 仅有 `/dev/video0`、`/dev/video1`，且 `udevadm` 显示为**同一台** UVC 摄像头（Sonix USB2.0 HD UVC WebCam）的两个节点；指南要求的 top=4 / gripper=2 两路相机均无法核验。
- 因此本机不是接好机械臂的现场主机；第 5–7 节的 12 次正式试次未开始。

## 6. 现场必须人工决定的一项

`--robot.id` 存在二义性，代码不会报错，只会在 `nn.json` 存在时静默加载：

| 文件 | SHA256 | 主要差异 |
|---|---|---|
| `~/.cache/huggingface/lerobot/calibration/robots/so_follower/nn.json` | `c6f9487c…` | 各关节 homing_offset 约 −1900 量级；gripper range `[1526,2978]` |
| `…/nnn.json` | `f38d337b…` | homing_offset 约 0–100 量级；gripper range `[2043,3493]` |

两套标定的关节语义不同，`wrist_roll`、`gripper` 尤其明显。历史记录：formal3 采数用 `nnn`；F3 r1 真机冻结配置用 `nn`（`outputs/f3_r1/session_preflight/deployment_config_frozen.json`）。**开工前必须在现场确认实物当前标定对应哪一份，再统一四条指令；本轮未擅自改动指南默认值。**
