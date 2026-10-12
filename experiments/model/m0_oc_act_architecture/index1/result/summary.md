# M0 OC-ACT Architecture Build — Result

状态：`ARCHITECTURE_READY = TRUE`

本轮只完成结构工程与 smoke 验证；没有正式训练，没有读取 dev/final-test，没有真机执行。验证使用 Formal4 `kind_merged_nomaster` episode 0 的真实 top/wrist 首帧、8D state 和 6D action，prototype 是明确标记的 `NOT_FOR_FORMAL_TRAINING` synthetic bank。

## Gate 结论

| Gate | 结果 | 关键证据 |
|---|---:|---|
| M0-G0 baseline 构建 | PASS | 原 ACT 构建、真实帧 forward 正常 |
| M0-G1 OC-ACT 构建 | PASS | 真实 pretrained `dinov2_vits14` 已下载并接通 |
| M0-G2 shape | PASS | top `[1,3,480,640]`；DINO `[1,256,384]`；raw tokens `[1,5,387]`；action `[1,4,6]` |
| M0-G3 DINO freeze | PASS | 22,056,576 个 frozen 参数；真实 backward 后 grad 全为 None |
| M0-G4 trainable gradient | PASS | adapter、cross-attention、gate、ACT action head 均为非零梯度 |
| M0-G5 baseline equivalence | PASS | 相同 ACT 权重、输入、eval 状态下 `alpha=0` max abs diff = `0.0` |
| M0-G6 object sensitivity | PASS | zero/shuffle/spatial-shift hidden max diff 均大于 0 |
| M0-G7 checkpoint | PASS | OC-ACT config、DINO、prototype buffer、adapter、attention、gate 完整往返，diff=`0.0` |
| M0-G8 baseline regression | PASS | baseline forward/loss/backward/save/reload 独立通过 |
| M0-G9 resource smoke | PASS | 参数、checkpoint、batch-1 latency、peak VRAM 已记录于 `metrics.json` |

## 关键解释

- `gate_init=0` 使模型初始行为严格等于原 ACT，同时 gate 在第一次 backward 已有非零梯度；adapter/cross-attention 在 gate 为 0 时理论上梯度为 0，因此 M0 按 decision 另用 `gate=0.2` 做连通性梯度 smoke。
- DINO 运行在 `torch.no_grad()` 且全部参数 `requires_grad=False`；action loss 只更新 adapter、cross-attention、gate 和原 ACT 路径。
- smoke loss 只证明计算图可运行，不构成模型效果证据。
- 当前 smoke prototype 绝不可用于正式训练。下一科研轮必须是独立的 `P0 Prototype Calibration`，由 train split 或独立 calibration reference 生成正式 prototype bank。

详细数值见 `validation.json` 与 `metrics.json`；结构说明见 `architecture.md`。
