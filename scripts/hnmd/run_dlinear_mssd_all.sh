#!/usr/bin/env bash
set -euo pipefail

python run_experiments.py \
  --models DLinear \
  --datasets all \
  --losses mssd \
  --mssd_mode tuned \
  --seeds 2024 \
  --itr 1 \
  "$@"
