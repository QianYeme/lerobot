# Diagram brief

## User goal
- Audience: thesis/paper reader (robotics + imitation learning), two-column manuscript
- Output: editable `.drawio` plus: PNG preview + SVG (300 dpi intent)
- Must communicate: the V2/V3 ACTDet training architecture — shared dual-view backbone, top-view FPN → {FCOS detection branch (training-only), Mask Decoder (training-only, Innovation 1), Detection-Feature Fusion (train+inference)}, wrist view original F4, merged tokens into the ACT Transformer (CVAE) → action L1 loss
- Must not do: modify data-collection scripts; invent modules/dims not in the code; mix V4–V6 injections into this V2/V3 figure

## Source inventory
| Source | Role (content/structure/style/layout/asset) | Notes |
|---|---|---|
| README_MVF_ACT.md §2.1 | content (stage order, training-only markers, Innovation 1) | authoritative semantics |
| src/lerobot/policies/act_det/detection/{fpn,fcos,fusion,mask_decoder}.py | content (dimensions: F2/F3/F4, P2–P4, 1×1 Conv, token count, loss names) | verified dims |
| .drawio-tmp/mvfact-v3/v23-architecture.spec.yaml (figure A) | structure (same 24-edge semantic graph, same legend semantics) | figure B must be content-equivalent |
| academic-figures-drawer skill contract | style (semantic palette, font ladder, dash grammar) | normative for figure B |

## Requirement traceability
| Requirement | Diagram evidence (panel/cell/edge) | Status |
|---|---|---|
| Dual-view input | top_cam, wrist_cam (input color) | done |
| Augmentation top-view only | aug node (dashed border = training only) | done |
| Shared ResNet18 F2/F3/F4 | backbone container + f2/f3/f4 | done |
| Top-view FPN P2–P4 | fpn node with per-level spatial sizes | done |
| Detection branch training-only | det container (dashed) + fcos_head → det_loss (dotted) ← cvat_gt (GT dashed) | done |
| Mask branch training-only, Innovation 1 | mask container (dashed) + mask_dec → mask_loss (dotted) ← sam2_gt (GT dashed); accent fill | done |
| Fusion train+inference | fusion (accent, solid container) → enhanced_f4 → conv_flatten | done |
| Wrist view original F4 | f4 → wrist_f4 → conv_flatten | done |
| ACT Transformer (CVAE) | act container + enc/dec/action_head, robot_state input | done |
| Action L1 loss | action_head → action_loss (dotted) ← action_gt (GT dashed) | done |
| Lifecycle legend | legend node: solid/dotted/dashed-arrow/dashed-border/accent | done |

## Semantic model
- Input: top-view RGB (3,480,640), wrist-view RGB (3,480,640), robot state (6/9-dim), GT: CVAT bbox (XML), SAM2 mask (NPZ offline), action chunk (B,100,6)
- Stages: augmentation (top only) → shared ResNet18 (F2 128×60×80, F3 256×30×40, F4 512×15×20) → FPN (P2/P3/P4 at 60×80 / 30×40 / 15×20) → FCOS head + Mask Decoder + Det-Feature Fusion (P2–P4 spatial attention enhancing F4); wrist F4 used raw
- Proposed contribution: detection branch (FCOS head, training-only supervision) + Det-Feature Fusion (train+inference) + Mask guidance = Innovation 1 (training-only)
- Output: 6-dim action chunk per step; L1 loss vs action GT
- Training-only path: aug, det container, mask container, all three loss nodes + their GT sources

## Open assumptions
- V2/V3 scope: V2 = detection branch, V3 = + mask guidance; V4–V6 injection paths are deliberately excluded (separate figure)
- Edge style mapping: solid = train+inference data flow, dotted = gradient/loss flow, gray dashed = GT annotation feed
- Accent marks all proposed blocks (fcos_head, fusion, mask_dec); "Innovation 1" = Mask Branch (legend states it)
