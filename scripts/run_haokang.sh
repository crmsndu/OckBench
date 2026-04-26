#!/usr/bin/env bash
# OckBench runner for Haokang.
#
# Assumes a single OpenAI-compatible chat-completions endpoint:
#   export OCKBENCH_API_KEY="..."
#   export OCKBENCH_BASE_URL="https://.../v1"
#
# Runs the OckBench Selected subsets (100 math + 60 coding + 40 science) for
# every (model, effort) entry in the RUNS table below.
#
# Cache files under cache/ are per-(tag, task), so Ctrl-C and rerun resumes.
#
# ONE THING TO EDIT: the RUNS table below. The model strings are guesses —
# replace them with whatever IDs your proxy actually exposes.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

[[ -f .venv/bin/activate ]] && source .venv/bin/activate

: "${OCKBENCH_API_KEY:?Set OCKBENCH_API_KEY before running}"
: "${OCKBENCH_BASE_URL:?Set OCKBENCH_BASE_URL before running (e.g. https://.../v1)}"

mkdir -p cache logs/haokang results

# ---------------------------------------------------------------------------
# EDIT THIS TABLE. Format:  "<model_id>  <cache_tag>  <reasoning_effort_or_->"
# Use "-" in the third column when you don't want --reasoning-effort sent.
# ---------------------------------------------------------------------------
RUNS=(
  "gpt-5.5            gpt55-none        none"
  "gpt-5.5            gpt55-minimal     minimal"
  "gpt-5.5            gpt55-low         low"
  "gpt-5.5            gpt55-medium      medium"
  "gpt-5.5            gpt55-high        high"
  "gpt-5.5            gpt55-xhigh       xhigh"
  "claude-opus-4-7    opus47            -"
  "deepseek-v4-flash  dsv4-flash        -"
  "deepseek-v4-pro    dsv4-pro          -"
)

for line in "${RUNS[@]}"; do
  read -r model tag effort <<<"$line"
  for task in math coding science; do
    case "$task" in
      math)    ds=data/OckBench_Math_Selected.jsonl;    name=OckBench_math_selected ;;
      coding)  ds=data/OckBench_Coding_Selected.jsonl;  name=OckBench_coding_selected ;;
      science) ds=data/OckBench_Science_Selected.jsonl; name=OckBench_science_selected ;;
    esac
    cache_file="cache/${tag}-${task}-selected.jsonl"
    log_file="logs/haokang/${tag}-${task}.log"

    effort_args=()
    [[ "$effort" != "-" ]] && effort_args=(--reasoning-effort "$effort")

    echo
    echo "=================================================================="
    echo "[$(date +%H:%M:%S)] $model :: $task ${effort_args:+effort=$effort}"
    echo "  cache: $cache_file"
    echo "=================================================================="
    python main.py \
      --provider chat_completion \
      --model "$model" \
      --api-key "$OCKBENCH_API_KEY" \
      --base-url "$OCKBENCH_BASE_URL" \
      --task "$task" \
      --dataset-path "$ds" \
      --dataset-name "$name" \
      --max-output-tokens 128000 \
      --concurrency 10 \
      --timeout 600 \
      --cache "$cache_file" \
      --log-dir logs/haokang \
      "${effort_args[@]}" 2>&1 | tee "$log_file" \
      || echo "[$(date +%H:%M:%S)] $model $task FAILED (continuing)"
  done
done

echo
echo "=================================================================="
echo "[$(date +%H:%M:%S)] All done. Result files:"
echo "=================================================================="
ls -1 results/OckBench_*_selected_* 2>/dev/null | tail -40 || echo "(no result files found)"
