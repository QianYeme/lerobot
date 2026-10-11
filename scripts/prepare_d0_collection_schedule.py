#!/usr/bin/env python3
"""Generate the frozen 5-round by 12-stratum D0 collection schedule."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


ROWS = ("front", "middle", "back")
COLS = ("left_outer", "left_inner", "right_inner", "right_outer")
FIELDNAMES = (
    "episode",
    "round",
    "order_in_round",
    "stratum",
    "row",
    "column",
    "actual_cx_px",
    "actual_cy_px",
    "status",
    "reach",
    "grasp",
    "lift",
    "hold",
    "place",
    "release",
    "drop",
    "spill",
    "closing_start",
    "closing_end",
    "lift_start",
    "hold_start",
    "hold_end",
    "lower_start",
    "place_contact",
    "release_start",
    "release_end",
    "notes",
)


def build_schedule(seed: int) -> list[dict]:
    rng = random.Random(seed)
    strata = [(row, column, f"{row}.{column}") for row in ROWS for column in COLS]
    records = []
    episode = 0
    for round_index in range(1, 6):
        order = strata.copy()
        rng.shuffle(order)
        for order_index, (row, column, stratum) in enumerate(order, start=1):
            record = {name: "" for name in FIELDNAMES}
            record.update(
                episode=episode,
                round=round_index,
                order_in_round=order_index,
                stratum=stratum,
                row=row,
                column=column,
            )
            records.append(record)
            episode += 1
    return records


def validate(records: list[dict]) -> None:
    if len(records) != 60 or [row["episode"] for row in records] != list(range(60)):
        raise ValueError("Schedule must contain episodes 0..59 exactly once")
    counts = {}
    for row in records:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    if set(counts.values()) != {5} or len(counts) != 12:
        raise ValueError("Each of 12 strata must occur exactly five times")
    for round_index in range(1, 6):
        round_rows = [row for row in records if row["round"] == round_index]
        if len(round_rows) != 12 or len({row["stratum"] for row in round_rows}) != 12:
            raise ValueError(f"Round {round_index} must cover all 12 strata once")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "collection_schedule.csv"
    json_path = args.output / "collection_schedule.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite an existing frozen schedule")
    records = build_schedule(args.seed)
    validate(records)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps({"seed": args.seed, "episodes": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(records), "csv": str(csv_path), "json": str(json_path)}))


if __name__ == "__main__":
    main()
