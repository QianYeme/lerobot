#!/usr/bin/env python
"""Render the frozen T0.2 transplant evidence into report files."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/position/t0_2_token_transplant/index1/result"
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    controls = json.loads((root / "controls.json").read_text(encoding="utf-8"))
    results = json.loads((root / "transplant_results.json").read_text(encoding="utf-8"))
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    variants = ["whole", "local_only", "global_only"]
    colors = {"whole": "#2563eb", "local_only": "#0f766e", "global_only": "#d97706"}

    def axis_plot(axis: str, target_key: str, action_dim: int, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        rows = [r for r in results if r["group"] == axis]
        x = np.asarray([r[target_key] for r in rows])
        for variant in variants:
            y = np.asarray([r["variants"][variant]["action_delta_h0"][action_dim] for r in rows])
            ax.scatter(x, y, s=24, alpha=0.78, label=variant, color=colors[variant])
        ax.axhline(0, color="black", lw=0.7); ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel(f"donor - base token coordinate ({target_key[-1]})")
        ax.set_ylabel(f"h0 action delta dim{action_dim}")
        ax.set_title(f"T0.2 {axis} directionality")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)

    axis_plot("horizontal", "target_delta_x", 0, "horizontal_directionality.png")
    axis_plot("longitudinal", "target_delta_y", 2, "longitudinal_directionality.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [[metrics[v][a]["mean_abs_arm"] for v in variants] for a in ("horizontal", "longitudinal")]
    x = np.arange(2); width = 0.24
    for i, v in enumerate(variants): ax.bar(x + (i - 1) * width, [vals[j][i] for j in range(2)], width, label=v, color=colors[v])
    ax.set_xticks(x, ["horizontal", "longitudinal"]); ax.set_ylabel("mean |h0 arm action delta|")
    ax.set_title("Whole vs local-only vs global-only"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures / "whole_vs_local_vs_global.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = ["same_position", "horizontal", "longitudinal", "random_unmatched"]
    ax.bar(np.arange(4) - 0.15, [metrics["whole"][g]["mean_abs_arm"] for g in groups], 0.3, label="whole", color=colors["whole"])
    ax.bar(np.arange(4) + 0.15, [metrics["local_only"][g]["mean_abs_arm"] for g in groups], 0.3, label="local-only", color=colors["local_only"])
    ax.set_xticks(np.arange(4), ["same", "horizontal", "longitudinal", "random"]); ax.set_ylabel("mean |arm action delta|")
    ax.set_title("Large-position vs controls"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures / "large_vs_same_position.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    mat = np.asarray([[np.mean([abs(r["variants"][v]["action_delta_h0"][d]) for r in results if r["group"] in ("horizontal", "longitudinal")]) for d in range(6)] for v in variants])
    im = ax.imshow(mat, aspect="auto", cmap="magma"); ax.set_xticks(range(6), [f"dim{i}" for i in range(6)]); ax.set_yticks(range(3), variants)
    ax.set_title("Per-action-dimension mean absolute response"); fig.colorbar(im, ax=ax, label="mean |delta|"); fig.tight_layout(); fig.savefig(figures / "per_action_dimension.png", dpi=180); plt.close(fig)

    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    validation["checkpoint_frozen"] = True
    (root / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    text = """# T0.2 Token Transplant Directionality

## Scope and validity

This is a Dev-only diagnostic on frozen `P1_ocfix2_s1/100000`; it does not retrain, read Final Test, or operate the robot. The same base Top/Wrist RGB and 8D state are used for each pair; only object tokens are transplanted. Pair selection was frozen before counterfactual action inference: confidence >= 0.20, pre-release timing fraction <= 0.75, state-standardised nearest matching with time difference <= 0.20, and 20 deterministic pairs per group (seed 17). No phase labels exist in Formal4, so the timing rule is explicitly recorded rather than presented as human phase ground truth.

## Result

The outcome is **PARTIAL (one-axis mechanism evidence)**. Longitudinal transplant is directionally consistent for whole/local/global tokens (0.90/0.85/0.90), exceeds the 0.70 reference, and the large-position response is larger than same-position control (whole arm delta 0.0691 vs 0.0183). Random-unmatched longitudinal control is only 0.15. Horizontal directionality does not pass (whole/local/global: 0.20/0.00/0.20); therefore the model cannot be claimed to have reliable two-axis steering.

Whole-token and global-only responses are larger than local-only for large longitudinal pairs, while local-only still preserves the longitudinal sign. Gripper response is reported separately; the dominant longitudinal effect is not a gripper-only change.

## Fixed interpretation

The frozen P1 OC-ACT representation contains evidence of causal longitudinal object-conditioned steering, but the horizontal channel remains unreliable and is confounded by weak spatial sensitivity in T0.1. This is not a PASS for a complete spatial mechanism. Do not enter Confirmation or Final Test from this result; the next research decision should target a single horizontal-steering intervention or an improved position representation.

See `metrics.json`, `controls.json`, `pair_manifest.json`, `transplant_results.json`, and the PNG figures in `figures/` for raw evidence.
"""
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "proposal.md").write_text("""# Proposal after T0.2\n\nKeep the frozen P1 checkpoint as a longitudinal-mechanism reference. Before any Confirmation, design one new Discovery intervention for horizontal steering only (explicit x representation or a horizontal contrastive/object-token loss), with a new decision contract and fresh Dev evidence. Do not tune gate, prototypes, tau, coordinate scale, or local-token order inside T0.2.\n""", encoding="utf-8")


if __name__ == "__main__":
    main()
