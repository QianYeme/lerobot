#!/usr/bin/env bash
set -euo pipefail
job_dir="$1"
trap 'printf "%s\n" "$?" > "$job_dir/exit_code.txt"' EXIT
/root/miniconda3/bin/python -m venv "$job_dir/.venv"
"$job_dir/.venv/bin/python" -m pip install --disable-pip-version-check numpy==1.26.4 scipy==1.13.1 scikit-learn==1.4.2 threadpoolctl==3.6.0 joblib==1.4.2
"$job_dir/.venv/bin/python" "$job_dir/fit_f1p_remote.py" "$job_dir"
tar -czf "$job_dir/remote_output.tar.gz" -C "$job_dir/output" .
