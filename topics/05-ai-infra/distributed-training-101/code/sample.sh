#!/usr/bin/env bash
# 推理（读 ../data/ckpt.pt）。改下面变量，或：
#   ./sample.sh --prompt "HAMLET:" --tokens 400 --samples 5
set -euo pipefail
cd "$(dirname "$0")"
export DDP_DATA_DIR="${DDP_DATA_DIR:-$(cd .. && pwd)/data}"

# ========== 改这里 ==========
DEVICE=auto                 # auto | cpu | cuda | mps
PROMPT=$'ROMEO:\n'
TOKENS=500
TEMPERATURE=0.8
TOP_K=200
SAMPLES=3
SEED=1337
# ============================

cmd=(
  python3 sample.py
  --device "$DEVICE"
  --prompt "$PROMPT"
  --tokens "$TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --samples "$SAMPLES"
  --seed "$SEED"
)

echo "+ ${cmd[*]}" "$@"
exec "${cmd[@]}" "$@"
