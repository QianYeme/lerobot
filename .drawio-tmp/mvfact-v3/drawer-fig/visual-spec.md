# Visual specification

## Global style
- Canvas/aspect: landscape, 1760 × 1200 px (within skill band 1600–2200 × 850–1200)
- Font: Arial throughout
- Grid/margins: 10 px grid; 24–48 px outer margins (content bounds ≈ 32–1704 × 4–1176)
- Corner radius / stroke: arcSize 14–16 / 2 px normal, 2.5 px accent
- Arrow grammar: solid #263238 = data (train+inference); dotted `1 4` = gradient/loss flow; gray dashed `6 4` #6B7280 = GT annotation; all arrowheads classic filled

## Semantic palette
| Meaning | Fill | Stroke | Used in |
|---|---|---|---|
| Input/raw signal/context | #E8F2F5 | #58727D | top_cam, wrist_cam, action_gt, cvat_gt, sam2_gt, robot_state |
| Existing/standard | #EAF0F6 | #63758A | f2, f3, f4, fpn, wrist_f4, enc, dec |
| Feature/tensor transform | #EDE9F4 | #7B6A9A | aug (dashed border), enhanced_f4, conv_flatten |
| Training/task/output head | #F4EEDC | #9A7B3F | det_loss, mask_loss, action_loss |
| Proposed contribution (accent) | #F1D7D4 | #B44948 | fcos_head, fusion, mask_dec |
| Output/decision | #E5F1E3 | #5A8A55 | action_head |
| Group container | none | #63758A dashed `8 8` | 4 containers |

Deviation note: skill default colors group borders with the auxiliary gray (#6B7280); using #63758A instead keeps vertex stroke count ≤ 6 (validator WARN threshold). Edge colors (#263238 / #6B7280) are not counted as vertex strokes.

## Typography
- Container titles: 21 px bold (grp_act 20 px — deviation, see below)
- Module/body labels: 16 px (line 1 = name, line 2 = dims/ops)
- Legend: 13 px; edge labels (GT, actions): 12 px, white halo
- Hierarchy gate: 21/13 ≈ 1.62 ≥ 1.5 ✓
- Paper-scale note: 1760 px canvas at 700 pt print width → 16 px ≈ 6.4 pt effective. Recorded deviation: figure B favors the skill-native composition over figure A's minPt=7 print gate; acceptable for the A/B skill comparison, would re-balance for final submission.

## Composition notes
- Overview: left column = dual inputs + GT sources; backbone column center-left; FPN hub with 4-way fan (up → FCOS, right → fusion, down → mask, left ← F2–F4); ACT stack far right; legend top-left
- Detail: fusion sits right of FPN at mid height (f4 feeds it above FPN top edge); mask branch bottom-middle with stacked loss + GT; wrist lane routed along y=808 bottom corridor
- Legend: top-left single text node, 2 lines, 13 px
- Forbidden crossing zones: container titles, node bodies, legend block
- Accepted exception: e07 (f4→fusion) × e09 (fpn→fcos) mid-air crossing at ≈(892,223) — proven geometrically unavoidable (entry above fpn top requires port outside fusion left edge; entry below fpn bottom crosses fpn body). Same crossing exists in figure A. See defect-log.md.
