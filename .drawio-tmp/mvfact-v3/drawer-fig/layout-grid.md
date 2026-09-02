# Layout grid

- Canvas: 1760 × 1200, origin top-left, 10 px grid
- Coordinate origin: page top-left; all coordinates absolute

## Containers (dashed, fillColor=none)
| id | x | y | w | h | title (fs) |
|---|---|---|---|---|---|
| grp_backbone | 512 | 4 | 216 | 348 | Shared ResNet18 (21) |
| grp_det | 744 | 32 | 720 | 144 | Detection Branch (training only) (21) |
| grp_mask | 1064 | 416 | 488 | 272 | Mask Branch (training only) (21) |
| grp_act | 1432 | 736 | 272 | 440 | ACT Transformer (CVAE) (20) |

## Vertices (x, y, w, h; parent)
| id | x | y | w | h | parent | fill |
|---|---|---|---|---|---|---|
| top_cam | 32 | 104 | 152 | 64 | 1 | input |
| wrist_cam | 32 | 380 | 152 | 64 | 1 | input |
| action_gt | 48 | 1096 | 152 | 64 | 1 | input |
| aug | 240 | 96 | 232 | 56 | 1 | transform (dashed) |
| f2 | 544 | 52 | 152 | 56 | grp_backbone | standard |
| f3 | 544 | 140 | 152 | 56 | grp_backbone | standard |
| f4 | 544 | 228 | 152 | 56 | grp_backbone | standard |
| fpn | 760 | 264 | 264 | 80 | 1 | standard |
| fcos_head | 760 | 72 | 264 | 80 | grp_det | accent |
| det_loss | 1088 | 72 | 168 | 64 | grp_det | head |
| cvat_gt | 1288 | 72 | 152 | 64 | grp_det | input |
| fusion | 1120 | 200 | 288 | 96 | 1 | accent |
| enhanced_f4 | 1480 | 216 | 160 | 68 | 1 | transform |
| mask_dec | 1088 | 456 | 264 | 80 | grp_mask | accent |
| mask_loss | 1376 | 456 | 160 | 64 | grp_mask | head |
| sam2_gt | 1376 | 600 | 160 | 64 | grp_mask | input |
| conv_flatten | 1452 | 744 | 216 | 72 | grp_act | transform |
| robot_state | 1248 | 880 | 160 | 56 | 1 | input |
| enc | 1452 | 864 | 216 | 80 | grp_act | standard |
| dec | 1452 | 992 | 216 | 72 | grp_act | standard |
| action_head | 1452 | 1112 | 216 | 56 | grp_act | output |
| action_loss | 1240 | 1112 | 168 | 56 | 1 | head |
| wrist_f4 | 560 | 512 | 152 | 56 | 1 | standard |
| legend | 32 | 32 | 440 | 56 | 1 | text |

## Baselines / columns
- x=32 input rail (top_cam, wrist_cam); x=48 action_gt (16 px offset breaks the column cluster → avoids spacing WARN; invisible at print scale)
- x=544 backbone column (f2/f3/f4, 32 px gaps); wrist_f4 at x=560 (take-off lane)
- x=760/1088/1288 det row (64/32 px gaps); x=1452 ACT stack (48 px gaps); x=1376 mask loss/GT column
- Vertical corridors: enhanced drop at x=1560 (8 px right of mask container); wrist lane y=808; mask lane y=496

## Edges (exact ports; all edges carry source/target refs + explicit sourcePoint/targetPoint + exit/entry fractions)
| id | from → to | ports (exit → entry) | style |
|---|---|---|---|
| e01 | top_cam→aug | (184,136)→(240,124) | solid |
| e02 | wrist_cam→f2 | (184,412)→(544,100) | solid |
| e03 | aug→f2 | (472,124)→(544,80) | solid |
| e04 | f2→fpn | (696,80)→(760,282) | solid |
| e05 | f3→fpn | (696,168)→(760,304) | solid |
| e06 | f4→fpn | (696,264)→(760,326) | solid |
| e07 | f4→fusion | (696,236)→(1120,210) | solid |
| e08 | f4→wrist_f4 | (620,284)→(636,512) | solid |
| e09 | fpn→fcos_head | (892,264)→(892,152) | solid |
| e10 | fcos_head→det_loss | (1024,112)→(1088,104) | dotted |
| e11 | cvat_gt→det_loss | (1288,104)→(1256,104) | GT dashed, label "GT" |
| e12 | fpn→fusion | (1024,304)→(1120,250) | solid |
| e13 | fpn→mask_dec | (892,344)→(1088,496) via (892,496) | solid |
| e14 | mask_dec→mask_loss | (1352,496)→(1376,496) | dotted |
| e15 | sam2_gt→mask_loss | (1456,600)→(1456,520) | GT dashed, label "GT" |
| e16 | fusion→enhanced_f4 | (1408,248)→(1480,250) | solid |
| e17 | enhanced_f4→conv_flatten | (1560,284)→(1560,744) | solid |
| e18 | wrist_f4→conv_flatten | (712,540)→(1452,780) via (712,808),(1400,808),(1400,780) | solid |
| e19 | conv_flatten→enc | (1560,816)→(1560,864) | solid |
| e20 | robot_state→enc | (1408,908)→(1452,908) | solid |
| e21 | enc→dec | (1560,944)→(1560,992) | solid |
| e22 | dec→action_head | (1560,1064)→(1560,1112) | solid |
| e23 | action_head→action_loss | (1452,1140)→(1408,1140) | dotted, label "actions" |
| e24 | action_gt→action_loss | (200,1128)→(1240,1140) | GT dashed, label "GT" |

- Forbidden crossing zones: container titles, node bodies, legend block
- Drawing order: containers → vertices → edges → legend
