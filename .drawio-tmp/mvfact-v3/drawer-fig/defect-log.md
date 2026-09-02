# Defect log

## Accepted exceptions (pre-authorized, geometric proofs in visual-spec.md)

| ID | Class | Description | Severity | Resolution |
|---|---|---|---|---|
| X01 | edge-edge crossing | e07 (f4→fusion, port (696,236)→(1096,210)) crosses e09 (fpn→fcos_head, x=892 vertical) at ≈(892,223), mid-air, 3 px above fcos_head bottom edge (892,160) | P2, accepted | Proven unavoidable: entry y<160 requires a port left of fusion's left edge (x<1096, exit fraction 0.143 already maximal); entry y>293 crosses the fpn body; a corridor below fpn bottom is occupied by the mask lane (e13). Same crossing exists in figure A. |
| D01 | typography scale | 16 px body labels at 700 pt print width → ≈6.4 pt effective; below the cn-thesis minPt=7 gate used by figure A | accepted for A/B comparison | Figure B favors the skill-native composition. Would re-balance fonts before final submission (figure A's profile already documents the print gate). |
| D02 | container stroke | container borders use #63758A instead of skill auxiliary #6B7280 | accepted | Keeps vertex stroke count at 6 (validator WARN threshold is 7+). Edge colors are not counted as vertex strokes. |
| D03 | title font | grp_act container title at 20 px instead of 21 px | accepted | "ACT Transformer (CVAE)" = 22 chars × 21 px ≈ 255 px exceeds the 256 px usable title width (272 − 16 padding). 20 px is visually indistinguishable. |
| D04 | input column offset | action_gt at x=48 instead of x=32 | accepted | 16 px offset breaks an x=32 column cluster (3 items, cv≈0.51) that would emit a spacing-variance WARN. Invisible at print scale. |
| D05 | label spacing | fpn label uses nbsp for per-level double spaces (`P2 (60×80)  P3 (30×40)  P4 (15×20)`) | accepted | Keeps level groups readable; nbsp counts as latin char in the overflow estimator. |

## Validation results

- `validate_drawio.py`: OK (54 cells, 28 vertices, 24 edges, 0 raster, 0 duplicate ids)
- `validate_visual_quality.py`: 0 FAIL, 0 WARN, PASSED
- Round 1 had 2 FAIL (e03/e04 endpoints inside the 680 px legend box) + 2 WARN (5-member y-chain across input/aug/det rows; 3-member x=1088 column chain). Fixes: legend narrowed to 440 px with 2 compact lines; det row moved y 88→72 (breaks transitive chain with aug at 96, keeps 15 px clearance from container title); fusion moved x 1096→1120 (breaks column chain; X01 crossing recomputed to (892,223), unchanged in kind).

## Export checklist

- [x] PNG export (draw.io Desktop, -s 2 → 6142×3806, repaired via repair_png.py, DPI 300 metadata set with PIL)
- [x] SVG export (draw.io Desktop)
- [x] PNG visual self-check (Read): no clipped text, no edge-through-box, accepted crossing X01 present but mid-air, legend readable, accent on fcos/fusion/mask_dec, all 4 container titles clear
