#!/usr/bin/env bash
set -euo pipefail

python run_experiments.py \
  --models MLP DLinear SOFTS iTransformer \
  --datasets all \
  --losses mssd mse tildeq \
  --seeds 2024 \
  --itr 3 \
  "$@"
