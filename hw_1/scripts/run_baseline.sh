#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
CONFIG="${1:-configs/baseline.yaml}"
"$PYTHON" src/train.py --config "$CONFIG"
"$PYTHON" src/evaluate.py --config "$CONFIG"
