# R5 真机前本机预检记录（2026-10-23）

范围：只做 `REAL_ROBOT_TEST_GUIDE.md` 第 2–3 节的真机前离线步骤。**未连接机械臂、未连接两路相机、未执行任何真机试次**；R5_COMPLETE=false，正式试次 0/15。

## 1. 机器与网络

| 项 | 值 |
|---|---|
| Python | 3.12.13（conda `lerobotnn`） |
| torch | 2.10.0+cu128 |
| GPU | 开始时不可用（内核 7.0.0-38 缺 nvidia 模块包），后修复：`apt install linux-modules-nvidia-580-open-7.0.0-38-generic` 后 `nvidia-smi` 正常（RTX 4060 Laptop, 580.178.04, 8188 MiB），`torch.cuda.is_available()=True` |
| `huggingface.co` | TLS 被重置（curl exit 35；`hf download` 报 `[Errno 104] Connection reset by peer`），加 token 无效 |
| `hf-mirror.com` | 仓库 API 与普通文件可用（`checksums.json` 走 resolve-cache 约 25 s），**LFS 大文件不可用**：`resolve` 无响应、整档 GET 在 257 s 被重置 |
| 实际下载通路 | `https://aifasthub.com/<repo>/resolve/main/<file>`（同一 HF revision，LFS 直连，约 0.5–0.8 MB/s） |

## 2. 下载与完整性

- 仓库 `QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101`，revision `6990fb2e82c39a571671fd22f7f45979f9d9ade6`，与 `hf_publish_receipt.json` 一致。
- 按指南第 2 节的校验脚本逐档核对，**两个归档均 OK**：

| 文件 | 字节 | SHA256 | 结论 |
|---|---|---|---|
| `R5_deploy_bundle.tar.gz` | 962,550,207 | `33b083a403098b46abb61f60045e38569d2fc25f4826539873d63552b67d9ba5` | 与 checksums/receipt 一致 |
| `R5_project_files.tar.gz` | 929,487 | `ac0d1dbb9df1dd370fa8c097fabc3c7a78b727eb52dbbda1a97b17efc815a90f` | 与 checksums/receipt 一致 |
| `checksums.json` | 1,326 | `609c48c3051d14da6e4facab7104af22ab32dc9aa16b0a823a9b0d718aec25b7` | 与 receipt 一致 |

- 解包位置 `outputs/deploy/R5/`（993 MB，gitignore 内）：`models/`、`local/`、`normalization/`、`assets/cup_prototype_bank_v1.pt`、`torch_cache/`、`manifest.json`（221 个文件摘要）、`model_identity.json`。
- 离线视觉缓存齐全，`torch_cache/hub/checkpoints/` 含 `dinov2_vits14_pretrain.pth`（88,283,115 B）与 `resnet18-f37072fd.pth`（46,830,571 B）；五个 dry-run 均在无网络假设下加载成功。

## 3. 代码身份

- 代码包解到独立目录 `outputs/deploy/R5_project_files_extract/`，**未覆盖仓库**。
- 与当前 `det` HEAD 逐文件比对：`src/lerobot/utils/r5_protocol.py`、`policies/act/r5_runtime.py`、`scripts/control_r5.py`、`utils/f5_protocol.py`、`scripts/build_r5_bundle.py`、`scripts/smoke_r5_runtime.py`、`scripts/summarize_r5.py` **全部逐字节相同**。
- 其余存在差异的文件（`control_f5.py`、`control_f3.py`、`policies/act_det/*`、部分 `scripts/eval_*.py`）经 `diff --strip-trailing-cr` 核对**内容一致，仅 CRLF 与 LF 的差异**；因此不需要启用备用代码包。

## 4. 身份与离线回归

- `python src/lerobot/scripts/control_r5.py --help`：exit 0，可选模型 M1–M5。
- `pytest`（R5/F5/R4/F3 相关）：**30 passed**。
- `verify_bundle(outputs/deploy/R5, model)` 对五个模型全部通过：

| 模型 | 标签 | global | local | normalizer | step/seed |
|---|---|---|---|---|---|
| M1 | Strong-G-36-S3 | G36 `e122e0e6…` | — | `normalization/train36.json` | 80000 / 3 |
| M2 | Strong-G-60-S3 | G60 `998026c8…` | — | `normalization/full60.json` | 80000 / 3 |
| M3 | New-GL-36 | G36 `e122e0e6…` | `local/New-GL-36.pt` `695d6fa8…` | `normalization/train36.json` | 80000 / 3 |
| M4 | New-GL-60 | G60 `998026c8…` | `local/New-GL-60.pt` `a71541b5…` | `normalization/full60.json` | 80000 / 3 |
| M5 | G60-BadSeed-S4 | BAD `1277d98f…` | — | `normalization/full60.json` | 80000 / 4 |

- `BOUND=4.832699459791184`，M3/M4 的 `local` 均带同一界限；`verify_bundle` 同时核对了 normalizer 与 checkpoint 内 `f2_1_normalization.json` 逐字节相同。

## 5. 五个模型 dry-run（CPU + CUDA）

先因 GPU 驱动缺失用 `--policy.device=cpu` 跑通；修复驱动后再用 `--policy.device=cuda` 复跑。这是**合成零观测的代码/smoke 检查，不是行为证据**。两条路径命令形态相同，仅 `--policy.device` 不同：

```bash
python src/lerobot/scripts/control_r5.py --bundle="$PWD/outputs/deploy/R5" \
  --session-dir="$PWD/outputs/r5/preflight[_cuda]" --dry-run --model=<M#> --position=A \
  --policy.device=[cpu|cuda] --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5 --dataset.fps=30 --dataset.episode_time_s=60 --dataset.reset_time_s=30
```

### CPU（`outputs/r5/preflight/`）

| 模型 | exit | 状态 | 六维有限 | 夹爪 |
|---|---|---|---|---|
| M1 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 19.665 |
| M2 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 16.241 |
| M3 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 14.835 |
| M4 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 11.419 |
| M5 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 18.633 |

- `offline_M1.json` 重复执行一次，六维动作**逐位相同**（CPU 无算法选择抖动）。

### CUDA（`outputs/r5/preflight_cuda/`，驱动修复后）

| 模型 | exit | 状态 | 六维有限 | 夹爪 | CUDA−CPU max\|Δ\| |
|---|---|---|---|---|---|
| M1 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 19.663 | 5.28e-3 |
| M2 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 16.245 | 1.26e-2 |
| M3 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 14.834 | 5.28e-3 |
| M4 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 11.423 | 1.26e-2 |
| M5 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | 是 | 18.650 | 3.12e-2 |

- CUDA 与 CPU 的六维最大差在 5e-3~3e-2 量级，是 TF32/GPU 数值 vs CPU float32 的预期差异，不是权重差异。
- **局部调节器隔离（CUDA 下同样精确成立）**：M3−M1、M4−M2 的五个臂维度差**精确 0.0**，仅夹爪差 −4.829162 / −4.822172，均在 ±4.832699 界内，说明 local 只在去归一化后、相对限幅前作用于夹爪，符合 `INVALID_LOCAL_ISOLATION` 守卫。
- 五个模型动作两两不同；M5（BadSeed-S4）与 G60 系差异最大，符合其独立 seed。

## 6. 未完成 / 现场前置条件

- **GPU 已修复并验证**：内核 7.0.0-38 缺 `linux-modules-nvidia-580-open-7.0.0-38-generic`（HWE 内核升级时 meta 包停在 7.0.0-30），`apt install` 补上后 `modprobe nvidia` 成功、`nvidia-smi` 正常、CUDA dry-run 已通过。不再阻塞。
- **硬件**：无 `/dev/ttyACM*`；`/dev/video0`、`/dev/video1` 是笔记本自带同一 UVC 摄像头。top=index4 / gripper=index2、`robot.id=nn` 与实物标定、夹爪开合方向、Home+Open 复位方式均**未核验**。
- 现场仍需按指南第 3 节：确认标定 id（`nn` 与 `nnn` 是不同关节语义文件，能加载不等于匹配）、用首帧图确认两路相机没接反、主从臂串口不要同时占用。

## 7. 结论

离线部分（下载、哈希、解包、身份核验、代码一致性、五模型 CPU 与 CUDA 加载/推理链路、local 隔离与界限）在本机全部通过；**全部真机项未验证**（无机械臂、无目标相机）。按指南，下一步是在连好设备的现场主机上完成第 3 节检查后，从 A 轮 M1 开始执行 15 个冻结试次。
