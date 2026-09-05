#!/usr/bin/env python
"""Plot training loss curves from one or more `<output_dir>/metrics.csv` files.

The training script (`lerobot_train.py`) appends a row to `metrics.csv` every
`log_freq` steps. This script reads those CSVs and renders a small-multiples
figure: one subplot per loss component, one line per model.

Usage:
    python scripts/plot_loss_curves.py \
        --model E4 outputs/train/2026-09-04/12-57-40_act/metrics.csv \
        --model E5 outputs/train/2026-09-04/12-57-58_act_det/metrics.csv \
        --model E6 outputs/train/2026-09-04/12-58-39_act_det/metrics.csv \
        --out loss_curves.png
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# Loss components to plot, in display order. Only components present in at least
# one CSV are shown (act has no det_*/mask columns; act_det has det_*; E6 adds mask).
LOSS_COLS = [
    "loss",
    "l1_loss",
    "det_cls_loss",
    "det_reg_loss",
    "det_ctr_loss",
    "mask_loss",
    "kld_loss",
]

# Colorblind-safe categorical palette (Okabe-Ito). Assigned to models in the order
# they are given on the command line; never cycled beyond this fixed list.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        required=True,
        help="Label and metrics.csv path for one model. Repeat for each model.",
    )
    p.add_argument("--out", default="loss_curves.png", help="Output PNG path.")
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window applied to each curve (1 = no smoothing).",
    )
    p.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only plot steps >= this value (useful to zoom into the resume range).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labels = [m[0] for m in args.model]
    paths = [m[1] for m in args.model]

    frames = {}
    for label, path in zip(labels, paths):
        df = pd.read_csv(path)
        if args.min_step is not None:
            df = df[df["steps"] >= args.min_step]
        # Resume re-does steps since the last saved checkpoint, so a CSV appended
        # across an interruption can contain duplicate steps. Keep the last value
        # per step (the most recent run) and sort to a monotonic x-axis.
        df = df.drop_duplicates(subset=["steps"], keep="last").sort_values("steps")
        frames[label] = df

    # Which loss components actually have data in at least one model?
    present = [c for c in LOSS_COLS if any(df[c].notna().any() for df in frames.values())]

    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.9 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    colors = {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(labels)}

    for ax, col in zip(axes, present):
        n_series = 0
        for label, df in frames.items():
            if not df[col].notna().any():
                continue
            s = df[[ "steps", col]].dropna()
            if args.smooth and args.smooth > 1:
                s = s.set_index("steps")[col].rolling(args.smooth, min_periods=1).mean().reset_index()
            ax.plot(s["steps"], s[col], color=colors[label], label=label, linewidth=1.6)
            n_series += 1

        ax.set_ylabel(col)
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.6, linestyle="--")
        ax.tick_params(axis="both", labelsize=9)
        if n_series >= 2:
            ax.legend(frameon=False, fontsize=9, loc="upper right")
        else:
            # Single series: the y-label already names it, no legend box needed.
            ax.set_title(col, fontsize=10, loc="left")

    axes[-1].set_xlabel("training step")
    fig.suptitle("Training loss curves", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Console summary: final logged value of each component per model.
    print("\nFinal logged values:")
    for col in present:
        for label, df in frames.items():
            if df[col].notna().any():
                last = df[col].dropna().iloc[-1]
                print(f"  {label:>6} {col:>14}: {last:.4f}")


if __name__ == "__main__":
    main()
