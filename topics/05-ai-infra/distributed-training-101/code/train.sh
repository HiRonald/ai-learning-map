#!/usr/bin/env bash
# 训练。改下面变量，或在命令行追加覆盖：
#   ./train.sh
#   ./train.sh --steps 200 --batch 32
set -euo pipefail
cd "$(dirname "$0")"
export DDP_DATA_DIR="${DDP_DATA_DIR:-$(cd .. && pwd)/data}"

# ========== 改这里 ==========
PRESET=shakespeare          # demo | shakespeare
DEVICE=auto                 # auto | cpu | cuda | mps
NPROC=1                     # >1 启用 DDP
RESUME=0                    # 1 = 从 ckpt.pt 续训
COMPILE=0                   # 1 = torch.compile（Mac 上建议关）

STEPS=                      # 空=用预设（shakespeare 5000）
BATCH=
LR=
MIN_LR=
WARMUP=
N_LAYER=
N_HEAD=
N_EMBD=
BLOCK_SIZE=
DROPOUT=
EVAL_INTERVAL=
EVAL_ITERS=
LOG_INTERVAL=
GRAD_ACCUM=

PROMPT=$'ROMEO:\n'          # 训完后的采样前缀
TOKENS=300
TEMPERATURE=0.8
TOP_K=200
# ============================

cmd=(
  python3 train.py
  --preset "$PRESET"
  --device "$DEVICE"
  --nproc "$NPROC"
  --prompt "$PROMPT"
  --tokens "$TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
)
if [[ "$RESUME" == "1" ]]; then cmd+=(--resume); fi
if [[ "$COMPILE" == "1" ]]; then cmd+=(--compile); fi

add() {
  local flag=$1 val=$2
  if [[ -n "$val" ]]; then cmd+=("$flag" "$val"); fi
}
add --steps "$STEPS"
add --batch "$BATCH"
add --lr "$LR"
add --min-lr "$MIN_LR"
add --warmup "$WARMUP"
add --n-layer "$N_LAYER"
add --n-head "$N_HEAD"
add --n-embd "$N_EMBD"
add --block-size "$BLOCK_SIZE"
add --dropout "$DROPOUT"
add --eval-interval "$EVAL_INTERVAL"
add --eval-iters "$EVAL_ITERS"
add --log-interval "$LOG_INTERVAL"
add --grad-accum "$GRAD_ACCUM"

echo "+ ${cmd[*]}" "$@"
exec "${cmd[@]}" "$@"
