#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
CONFIG="${1:-configs/baseline.yaml}"
"$PYTHON" src/train.py --config "$CONFIG"
# train.py already exports predictions; regenerate and verify all reports.
RESULTS="$("$PYTHON" -c 'import sys; sys.path.insert(0,"src"); import utils; print(utils.load_config(sys.argv[1])["output_dir"])' "$CONFIG")"
"$PYTHON" src/evaluate.py --config "$CONFIG" --results "$RESULTS"
