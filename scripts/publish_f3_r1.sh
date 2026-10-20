#!/usr/bin/env bash
set -euo pipefail
task_root=/root/autodl-tmp/F3_r1_transfer_20261020
trap 'code=$?; printf "%s\n" "$code" > "$task_root/publish.exit"' EXIT
/root/miniconda3/bin/python -u "$task_root/publish_f3_r1_bundle.py" --mode prepare
/root/miniconda3/bin/python -u "$task_root/publish_f3_r1_bundle.py" --mode upload
