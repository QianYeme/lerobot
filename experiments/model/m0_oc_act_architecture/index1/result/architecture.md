# M0 OC-ACT Architecture

```text
Top RGB ─┬─> original shared ResNet ─────────────────────────────┐
         └─> frozen pretrained DINOv2-S (16×16 patches)         │
              -> frozen cup prototype cosine similarity         │
              -> global + 2×2 local soft pooling                │
              -> [DINO semantic, x, y, confidence] × 5          │
              -> trainable adapter + token-slot embedding       │
                                                               ▼
Wrist RGB -> original shared ResNet ──────> ACT observation tokens
8D NOMASTER state ────────────────────────>          │ Q
                                                      │ K,V = object tokens
                                             object cross-attention
                                                      │
                                   h' = h + scalar_alpha * response
                                                      │
                                     original ACT encoder/decoder/head
                                                      │
                                               6D action chunk
```

实现边界：DINO 只读取 top camera，固定 resize 到 224×224 并做 ImageNet normalization；原 ACT 的 top/wrist ResNet、encoder、decoder 和 action head 均保留。baseline config 不构造任何 OC 模块。M0 只实现 pre-encoder 注入，不包含 Force、detector、SAM 或正式 prototype calibration。

Object token 固定为 5 个：全局、左上、右上、左下、右下。每个 raw token 是 `384D semantic + normalized x + normalized y + confidence = 387D`；adapter 映射到 ACT `dim_model`。token-slot embedding 使区域身份在 attention 中可辨识；显式 x/y 仍来自 similarity-weighted soft center。

Prototype bank 是 registered buffer，不接受 action gradient。M0 smoke bank 明确标记 `NOT_FOR_FORMAL_TRAINING`；正式训练前必须由独立 P0 decision 生成 train-only/reference-calibration prototype。
