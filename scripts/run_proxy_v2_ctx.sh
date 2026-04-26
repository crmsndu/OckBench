#!/usr/bin/env bash
# Like run_proxy_v2.sh but uses --max-context-window (for models with a
# combined input+output cap rather than a separate output cap).
# Usage: run_proxy_v2_ctx.sh <model_id> <short_name> <max_context_window>
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${1:?model required}"
SHORT="${2:?short name required}"
MAX_CTX="${3:?max_context_window required}"

mkdir -p cache logs/proxy-v2

for task in math coding science; do
  case "$task" in
    math)    ds=data/OckBench_Math_Selected.jsonl;    name=OckBench_math_selected ;;
    coding)  ds=data/OckBench_Coding_Selected.jsonl;  name=OckBench_coding_selected ;;
    science) ds=data/OckBench_Science_Selected.jsonl; name=OckBench_science_selected ;;
  esac
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] $MODEL :: $task (max_ctx=$MAX_CTX, c=1)"
  echo "================================================================"
  python main.py \
    --provider chat_completion \
    --model "$MODEL" \
    --api-key "$OCKBENCH_API_KEY" \
    --base-url https://proxy.example/v1 \
    --task "$task" \
    --dataset-path "$ds" \
    --dataset-name "$name" \
    --max-context-window "$MAX_CTX" \
    --concurrency 1 \
    --temperature 0 \
    --cache "cache/${SHORT}-${task}-selected.jsonl" \
    --log-dir logs/proxy-v2 \
    || echo "[$(date +%H:%M:%S)] $MODEL $task FAILED (continuing)"
done
echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] $MODEL :: ALL DONE"
echo "================================================================"
