#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/F2_1_r1_20261019
test -f "$base/results/historical_metric_reproduction.json"
test -f "$base/results/smoke.json"
for variant in A0 A1 A2; do
    test ! -e "$base/results/$variant"
    ! screen -ls | grep -q "F21_$variant"
done
for variant in A0 A1 A2; do
    screen -dmS "F21_$variant" bash "$base/train_one.sh" "$variant"
done
screen -ls
