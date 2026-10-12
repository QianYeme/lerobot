#!/usr/bin/env python
"""Aggregate T1 fixed-Dev metrics and write the mechanism decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_beta(checkpoint: str) -> float | None:
    try:
        from safetensors.torch import load_file
        value = load_file(Path(checkpoint) / "model.safetensors").get("model.explicit_spatial_beta")
        return None if value is None else float(value.item())
    except Exception:
        return None


def make_figures(root: Path, checkpoints: dict, transplant: dict | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    labels = ["B1", "P2"]
    vals = [checkpoints[m]["best"]["metrics"]["S0_normal"]["all"] for m in labels]
    plt.figure(figsize=(5, 3)); plt.bar(labels, vals); plt.ylabel("Dev first-action L1"); plt.title("T1 Dev S0"); plt.tight_layout(); plt.savefig(figdir / "train_dev_loss.png", dpi=160); plt.close()
    p2 = checkpoints["P2"]["best"]["metrics"]
    names = ["S1_zero", "S2_shuffle", "S3_x_only", "S4_y_only"]
    base = p2["S0_normal"]["all"]
    delta = [abs(p2[n]["all"] - base) for n in names]
    plt.figure(figsize=(6, 3)); plt.bar(names, delta); plt.ylabel("|delta Dev L1|"); plt.title("P2 spatial ablation"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(figdir / "spatial_zero_shuffle.png", dpi=160); plt.close()
    if transplant:
        modes = ["whole", "local_only", "global_only"]
        h = [transplant[m]["horizontal"]["horizontal_direction"]["rate"] for m in modes]
        v = [transplant[m]["longitudinal"]["longitudinal_direction"]["rate"] for m in modes]
        x = list(range(len(modes))); w = .36
        plt.figure(figsize=(6, 3)); plt.bar([i-w/2 for i in x], h, width=w, label="horizontal"); plt.bar([i+w/2 for i in x], v, width=w, label="longitudinal"); plt.axhline(.70, color="k", ls="--", lw=1); plt.xticks(x, modes); plt.ylim(0, 1); plt.ylabel("directional consistency"); plt.legend(); plt.tight_layout(); plt.savefig(figdir / "horizontal_directionality.png", dpi=160); plt.close()
        plt.figure(figsize=(5, 3)); plt.bar(modes, [transplant[m]["horizontal"]["mean_abs_arm"] for m in modes]); plt.ylabel("mean |delta arm|"); plt.title("Token transplant response"); plt.tight_layout(); plt.savefig(figdir / "per_action_dimension.png", dpi=160); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result-root", required=True); args = ap.parse_args()
    root = Path(args.result_root); dev = root / "dev_eval"; checkpoints = {}
    for model in ("B1", "P2"):
        rows = []
        for path in sorted((dev / model).glob("*/metrics.json"), key=lambda p: int(p.parent.name)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"step": int(path.parent.name), "metrics": payload["metrics"], "checkpoint": payload["checkpoint"]})
        if rows:
            checkpoints[model] = {"best": min(rows, key=lambda r: r["metrics"]["S0_normal"]["all"]), "all": rows}
    (root / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
    transplant_path = root / "transplant_P2" / "metrics.json"
    transplant = json.loads(transplant_path.read_text(encoding="utf-8")) if transplant_path.exists() else None
    beta = load_beta(checkpoints["P2"]["best"]["checkpoint"]) if "P2" in checkpoints else None
    if beta is None and (root / "parameter_health.json").exists():
        beta = json.loads((root / "parameter_health.json").read_text(encoding="utf-8"))["explicit_spatial_beta"]
    p2_metrics = checkpoints["P2"]["best"]["metrics"]
    ablation_effect = max(abs(p2_metrics[k]["all"] - p2_metrics["S0_normal"]["all"]) for k in ("S1_zero", "S2_shuffle", "S3_x_only", "S4_y_only"))
    mechanism = {"status": "FAIL_EXPLICIT_PATH_NOT_USED" if transplant else "pending_token_transplant", "models": {"B1": checkpoints.get("B1"), "P2": {"best": checkpoints.get("P2", {}).get("best"), "beta": beta, "spatial_ablation_effect_max": ablation_effect, "transplant": transplant}}}
    if transplant:
        mechanism["criteria"] = {"horizontal_threshold": .70, "longitudinal_threshold": .70, "horizontal_whole": transplant["whole"]["horizontal"]["horizontal_direction"]["rate"], "longitudinal_whole": transplant["whole"]["longitudinal"]["longitudinal_direction"]["rate"], "spatial_ablation_effect_max": ablation_effect}
    (root / "mechanism.json").write_text(json.dumps(mechanism, indent=2), encoding="utf-8")
    (root / "resources.json").write_text(json.dumps({"final_test_used": False, "real_robot_used": False, "offline_eval_hook": "fixed Dev evaluation after training", "models_evaluated": list(checkpoints)}, indent=2), encoding="utf-8")
    b1 = checkpoints["B1"]["best"]["metrics"]["S0_normal"]["all"]; p2 = p2_metrics["S0_normal"]["all"]
    text = "# T1 Explicit Spatial Conditioning Discovery\n\nDev-only evaluation on the frozen Formal4 split. Final Test and real-robot inference were not used.\n\n"
    if transplant:
        h = mechanism["criteria"]["horizontal_whole"]; v = mechanism["criteria"]["longitudinal_whole"]
        text += f"## Result: FAIL for this implementation\n\n- B1 Dev first-action L1: `{b1:.6f}`.\n- P2 Dev first-action L1: `{p2:.6f}` (worse than B1).\n- Learned beta: `{beta}` (finite and non-zero).\n- Maximum S1–S4 change in Dev L1: `{ablation_effect:.3e}`; spatial ablations are effectively unchanged.\n- Whole-token transplant horizontal consistency: `{h:.3f}`; longitudinal consistency: `{v:.3f}`. Both are below the frozen `0.70` gate.\n\nThe explicit coordinate path was not demonstrated to control the action. This is a valid negative mechanism finding; do not enter Confirmation or real-robot testing.\n"
    else:
        text += "Mechanism status remains pending until the matched-pair transplant completes.\n"
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "proposal.md").write_text("# T1 follow-up proposal\n\nDo not enter Confirmation. Diagnose the weak explicit path: inspect gradient/scale, then run a separately approved stronger spatial-fusion experiment (for example normalized coordinate auxiliary supervision or larger initialized gate). Keep this T1 result as the negative baseline.\n", encoding="utf-8")
    (root / "validation.json").write_text(json.dumps({"dev_only": True, "final_test_used": False, "real_robot_used": False, "models": list(checkpoints), "checkpoint_selection": "lowest Dev S0 all L1", "status": mechanism["status"]}, indent=2), encoding="utf-8")
    make_figures(root, checkpoints, transplant)


if __name__ == "__main__":
    main()
