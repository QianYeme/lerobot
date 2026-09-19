"""Phase-E val10 summary over full-run (100k) chunk captures, four models.

Reuses the phase-C metric definitions (summarize_run / relative_to_reference) with the
C50_{ACT,DET,I0,I1}_val10_s{STEP} naming and prints the pairwise §8 ratio table.

Usage: summarize_phaseE_val10.py --root outputs/act_sequence_phaseE_2026-09-19
"""
import argparse
import json
import re
from pathlib import Path

from summarize_phaseC_dev8 import relative_to_reference, summarize_run

RUN_PATTERN = re.compile(r"C50_(ACT|DET|I0|I1)_val10_s(\d+)$")
COLUMNS = ["h0_arm_raw", "h0_gripper_raw", "conflict_h1_5_arm", "conflict_h1_5_gripper",
           "stage_pre_arm", "stage_grasp_arm", "stage_post_arm", "delay_grasp_frames",
           "delay_release_frames", "step_ratio_arm", "step_ratio_gripper", "queue1_step2_arm"]
PAIRS = [("I1", "I0"), ("I1", "DET"), ("I1", "ACT"), ("I0", "DET"), ("I0", "ACT"), ("DET", "ACT")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for run_dir in sorted(args.root.iterdir()):
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if match and (run_dir / "results.json").exists() and (run_dir / "evaluation.done").exists():
            runs[(match.group(1), match.group(2))] = summarize_run(run_dir)
    print("| run | " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * (len(COLUMNS) + 1))
    for (model, step), run in sorted(runs.items()):
        cells = [(f"{run['aggregate'][key]:.4f}" if run["aggregate"].get(key) is not None else "-")
                 for key in COLUMNS]
        print(f"| {model} s{step} | " + " | ".join(cells) + " |")
    print("\n### 相对比值（<1 表示优于参照；验收线：h0 ≤1.05，冲突或关键阶段 ≤0.90）")
    steps = sorted({step for _, step in runs})
    for model, base in PAIRS:
        for step in steps:
            if (model, step) in runs and (base, step) in runs:
                ratios = relative_to_reference(runs[(model, step)], runs[(base, step)])
                text = "  ".join(f"{key}={value:.3f}" for key, value in ratios.items() if value is not None)
                print(f"- {model}/{base} s{step}: {text}")
    (args.root / "screen_summary.json").write_text(
        json.dumps({f"{model}_s{step}": run for (model, step), run in runs.items()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
