"""Apply the preregistered train48 screening gates to the h0-aux candidates.

Reads h0_bias_*.json from the diagnose step and compares each candidate
(W=3, W=10, 2k) against the preroll-2k baseline on the same train48 episodes:

- S1 (primary): bias_deg_abs_mean_arm <= 0.70 x baseline (the fixed h0 bias
  must be substantially reduced on the training set itself);
- S2 (companion): strictly more cosine+ episodes, strictly more pan-correct
  episodes, and a norm ratio median inside [0.5, 2.0].

A candidate passes screening only with S1 AND S2. Failure means immediate
stop: no extended training, no dev12 consumption. This is diagnostic, not a
promotion gate.
"""
import argparse
import json
from pathlib import Path

CANDIDATES = ("H0AUX_W03_002000", "H0AUX_W10_002000")
BASELINE = "PREROLL_BASE_002000"
BIAS_FRACTION = 0.70


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()

    root = args.campaign / "diagnose_train48"
    baseline = json.loads((root / f"h0_bias_{BASELINE}.json").read_text())
    base = baseline["metrics"]

    report = {"baseline": {"metrics": base}, "candidates": {}}
    for tag in CANDIDATES:
        data = json.loads((root / f"h0_bias_{tag}.json").read_text())
        m = data["metrics"]
        s1 = m["bias_deg_abs_mean_arm"] <= BIAS_FRACTION * base["bias_deg_abs_mean_arm"]
        s2 = (m["cosine_positive_count"] > base["cosine_positive_count"]
              and m["pan_direction_count"] > base["pan_direction_count"]
              and 0.5 <= m["norm_ratio_median"] <= 2.0)
        report["candidates"][tag] = {
            "metrics": m,
            "s1_bias_reduced_70pct": bool(s1),
            "s2_direction_pan_ratio_improved": bool(s2),
            "pass_screening": bool(s1 and s2),
        }

    out_json = root / "screening.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# h0-aux train48 2k screening (preregistered)", "",
             "| tag | cosine+ | pan | ratio med | bias abs mean arm | S1 | S2 | verdict |",
             "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tag, item in report["candidates"].items():
        m = item["metrics"]
        lines.append(f"| {tag} | {m['cosine_positive_count']}/48 | {m['pan_direction_count']}/48 | "
                     f"{m['norm_ratio_median']:.3f} | {m['bias_deg_abs_mean_arm']:.2f} | "
                     f"{'PASS' if item['s1_bias_reduced_70pct'] else 'FAIL'} | "
                     f"{'PASS' if item['s2_direction_pan_ratio_improved'] else 'FAIL'} | "
                     f"{'PASS' if item['pass_screening'] else 'FAIL'} |")
    b = base
    lines.append(f"| {BASELINE} (ref) | {b['cosine_positive_count']}/48 | {b['pan_direction_count']}/48 | "
                 f"{b['norm_ratio_median']:.3f} | {b['bias_deg_abs_mean_arm']:.2f} | - | - | reference |")
    lines += ["",
              f"- S1: bias_deg_abs_mean_arm <= {BIAS_FRACTION:.0%} of baseline",
              "- S2: strictly more cosine+/pan-correct episodes and ratio median in [0.5, 2.0]",
              "- Any candidate failing both gates stops immediately (no extended training, no dev12)."]
    (root / "screening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
