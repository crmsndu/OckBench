#!/usr/bin/env bash
# Run OckBench Selected subsets for one DeepSeek v4 variant.
# Usage: run_deepseek_v4.sh <model>
#   e.g. run_deepseek_v4.sh deepseek-v4-flash
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${1:?model required}"
mkdir -p cache logs/deepseek-v4

for task in math coding science; do
  case "$task" in
    math)    ds=data/OckBench_Math_Selected.jsonl;    name=OckBench_math_selected ;;
    coding)  ds=data/OckBench_Coding_Selected.jsonl;  name=OckBench_coding_selected ;;
    science) ds=data/OckBench_Science_Selected.jsonl; name=OckBench_science_selected ;;
  esac
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] $MODEL :: $task"
  echo "================================================================"
  python main.py \
    --provider chat_completion \
    --model "$MODEL" \
    --api-key "$DEEPSEEK_API_KEY" \
    --base-url https://api.deepseek.com \
    --task "$task" \
    --dataset-path "$ds" \
    --dataset-name "$name" \
    --max-output-tokens 384000 \
    --concurrency 5 \
    --enable-thinking true \
    --reasoning-effort max \
    --cache "cache/${MODEL}-${task}-selected.jsonl" \
    --log-dir logs/deepseek-v4 \
    || echo "[$(date +%H:%M:%S)] $MODEL $task FAILED (continuing)"
done

echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] $MODEL :: ALL DONE"
echo "================================================================"
