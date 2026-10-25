# R5.10 真机前本机离线预检记录（2026-10-25）

范围：`REAL_ROBOT_TEST_GUIDE.md` 第 2 节的下载与离线核对。**未连接机械臂、未连接两路相机、未执行任何真机试次**；R5_10_COMPLETE=false，正式试次 0/9（+可选 M4、定向确认）。

## 1. 网络与下载通路

| 项 | 值 |
|---|---|
| `huggingface.co` 直连 | 仍 TLS 被重置（curl exit 35），`hf download` 报 `Errno 104` |
| `hf download` 走 `HF_ENDPOINT=https://aifasthub.com` | 小文件可用，但**整仓下载会触发一致性校验错误**：aifasthub 把文件大小错报成 104857600（100MB），导致 `.gitattributes`/`checksums.json` 校验失败 |
| 实际通路 | 直接 `curl -L` 到 `aifasthub.com/.../resolve/main/<path>`（按递归树清单逐个下载），与 R5 时一致，全部成功 |

## 2. 下载与完整性

- 仓库 `QYyyyyyyy/R5_10_TRANSFER_SO101`。按递归树清单下载 250 个文件（`models/**`、`normalization/**`、`assets/**`、`torch_cache/**`、`model_identity.json`、`manifest.json`、`checksums.json`、`bundle_receipt.json`），逐个核对字节数，**250/250 一致**。
- 用 `checksums.json` 全量核对：覆盖 258 项，**249 项字节数与 SHA256 全部一致、0 不一致**；9 项"缺失"是文档/代码归档（`decision_*.md`、`README.md`、`REAL_ROBOT_TEST_GUIDE.md`、`R5_10_project_files.tar.gz`、`planned_trials.csv`、`runtime_smoke.json`、`home_*` 等），这些本应使用 GitHub `det` 代码、无需下载，符合指南"HF 只下模型和推理资产"。
- 四模型权重各 ~300MB：`models/M1..M4/model.safetensors` 齐全；`torch_cache/hub/checkpoints/` 含 `dinov2_vits14_pretrain.pth`（88MB）与 `resnet18-f37072fd.pth`（46MB）及 dinov2 源码树。
- 解包位置 `outputs/deploy/R5_10/`（1.3GB，gitignore 内），无 tarball，直接文件。

## 3. 身份核验

`verify_bundle(outputs/deploy/R5_10, model)` 四模型全过（核对 `clip_recipe`/`seed`/`source_steps`、global-only 无 local、Full60 normalizer、manifest 哈希、normalizer 与 checkpoint `f2_1_normalization.json` 逐字节一致）：

| 模型 | 标签 | clip | seed | source_steps | global SHA256（前 12） |
|---|---|---|---|---|---|
| M1 | C10-S3-80k | clip10 | 3 | [80000] | `4f9e6a057496` |
| M2 | C5-S2-80k | clip5 | 2 | [80000] | `1b37ee520945` |
| M3 | C5-S2-SWA_60_80 | clip5 | 2 | [60000,80000] | `fbffb937bbbd` |
| M4 | C5-S3-SWA_60_80_optional | clip5 | 3 | [60000,80000] | `8da087003aef` |

均为 6D 关节位置 state + 独立单帧 load、6D 位置动作、Full60 归一化、无 local 调节器，与 `model_identity.json`/`bundle_receipt.json` 一致。

## 4. 四模型 CUDA dry-run

`control_r5_10.py` 入口设置 `cudnn.benchmark=False`、`cudnn.deterministic=True`、TF32 保持 True。合成零观测，`--policy.device=cuda`：

| 模型 | exit | 状态 | 六维动作 |
|---|---|---|---|
| M1 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | -6.254, 23.202, 60.280, -69.635, -91.404, 18.628 |
| M2 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | -5.240, -24.301, 71.642, -36.540, -93.985, 16.489 |
| M3 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | -4.291, -20.706, 70.994, -39.512, -94.498, 16.480 |
| M4 | 0 | SYNTHETIC_CODE_SMOKE_ONLY | -3.221, 29.272, 55.453, -70.960, -97.144, 23.392 |

- 四模型动作全部有限、两两不同；M2/M3 同为 clip5 seed2（M3 为 M2 的 60k+80k SWA），动作相近符合预期。这是合成零观测 smoke，不是行为证据。

## 5. 未完成 / 现场前置条件

- GPU 正常（RTX 4060 Laptop 580.178.04，`torch.cuda.is_available()=True`），CUDA 路径已验证。
- 无 `/dev/ttyACM*`；`/dev/video0/1` 为笔记本自带 UVC。top=4 / gripper=2、`robot.id=nn` 标定、夹爪开合方向、人工复位方式均**未核验**。
- 指南 §3 明确本轮为"每次人工复位、无 Home 数值/容差检查"，启动前操作者手动统一起始点；`initial_state.json` 的 `valid` 为 null，试次有效性由现场人工填写。
- 现场仍需按 §4 确认相机没接反、标定 id 匹配实物、主从臂串口不双占。

## 6. 结论

离线下载与身份核验全部通过；四模型 CUDA 加载/推理链路正常。全部真机项未验证。按指南，现场人工复位到统一起始点后，从 A 轮 M1 开始跑 9 次主比较（+可选 M4、定向确认 ≤3 次）。
