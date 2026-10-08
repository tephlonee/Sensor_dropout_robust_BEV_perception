#!/usr/bin/env bash
# Train the three strategies and evaluate them. Extra args go to every training run,
# e.g.  bash run_week2.sh --base 16 --epochs 20      (faster, for CPU)
set -e
cd "$(dirname "$0")"
for cfg in baseline dropout aware; do
    echo "=== training $cfg ==="
    python step3_train.py --config "$cfg" "$@"
done
echo "=== evaluating ==="
python step4_evaluate.py
