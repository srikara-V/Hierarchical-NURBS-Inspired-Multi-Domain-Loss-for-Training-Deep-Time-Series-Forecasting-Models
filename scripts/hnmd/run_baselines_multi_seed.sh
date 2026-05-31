#!/usr/bin/env bash
set -euo pipefail

python run_experiments.py \
  --models MLP DLinear iTransformer \
  --datasets all \
  --losses mse tildeq \
  --seeds 2024 \
  --itr 3 \
  "$@"
