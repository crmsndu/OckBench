#!/usr/bin/env bash
# Run OckBench Selected subsets for one model served via chat-completion proxy.
# Usage: run_proxy_v2.sh <model_id> <short_name> <max_output_tokens>
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${1:?model required}"
SHORT="${2:?short name required}"
MAX_OUT="${3:?max_output_tokens required}"

mkdir -p cache logs/proxy-v2

for task in math coding science; do
  case "$task" in
    math)    ds=data/OckBench_Math_Selected.jsonl;    name=OckBench_math_selected ;;
    coding)  ds=data/OckBench_Coding_Selected.jsonl;  name=OckBench_coding_selected ;;
    science) ds=data/OckBench_Science_Selected.jsonl; name=OckBench_science_selected ;;
  esac
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] $MODEL :: $task (max_out=$MAX_OUT, c=1)"
  echo "================================================================"
  python main.py \
    --provider chat_completion \
    --model "$MODEL" \
    --api-key "$OCKBENCH_API_KEY" \
    --base-url https://proxy.example/v1 \
    --task "$task" \
    --dataset-path "$ds" \
    --dataset-name "$name" \
    --max-output-tokens "$MAX_OUT" \
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
