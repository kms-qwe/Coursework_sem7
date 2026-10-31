#!/usr/bin/env bash
# весь пайплайн, логи в runs/logs/
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u -I"
mkdir -p runs/logs

[ -f data/cache/train_256.npz ] || $PY src/prepare_data.py

if [ ! -f runs/models/base.pth ]; then
  caffeinate -i $PY src/train_base.py 2>&1 | tee runs/logs/train_base.log
fi
caffeinate -i $PY src/prune_baseline.py 2>&1 | tee -a runs/logs/prune_baseline.log
caffeinate -i $PY src/benchmark.py 2>&1 | tee runs/logs/benchmark.log
$PY src/report.py
