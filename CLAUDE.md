# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OckBench is an efficiency-aware benchmarking suite for LLMs that evaluates both accuracy and token usage. The key innovation is the **OckScore** metric:

```
OckScore = Accuracy - 10 * log(AvgTokens / 10000 + 1)
```

Three task types are supported: **math**, **coding**, and **science**, each with 200/200/100 problems in `data/`. Problem counts after dedup: math=200, coding=974, science=198.

A curated **Selected** subset (200 problems: 100 math + 60 coding + 40 science) lives in `data/OckBench_{Math,Coding,Science}_Selected.jsonl`. These were chosen by filtering to medium difficulty (10% < accuracy < 90%) and picking problems with the highest cross-model output token variance.

## Setup

```bash
uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -r requirements.txt
```

Always activate the virtual environment before running any commands:

```bash
source .venv/bin/activate
```

API keys via environment variables: `OPENAI_API_KEY` or `API_KEY` for chat_completion, `GEMINI_API_KEY` for Gemini, `ANTHROPIC_API_KEY` for Anthropic. `API_BASE_URL` sets the base URL.

## Running Benchmarks

```bash
# OpenAI
python main.py --model gpt-4o --task math --max-output-tokens 8192

# Local model via vLLM
python main.py --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1 --max-context-window 40960 --task math

# OpenRouter
python main.py --model openai/gpt-4o-mini --base-url https://openrouter.ai/api/v1 --api-key $KEY --task math --max-output-tokens 8192

# Using YAML config
python main.py --config configs/openai.yaml

# Resume interrupted run
python main.py ... --cache cache/run.jsonl
```

`--max-output-tokens` sets a fixed output budget; `--max-context-window` dynamically computes budget as `context_window - prompt_tokens`.

Results are saved to `results/` as JSON with per-problem details and aggregate summary statistics.

## LLM-based Re-evaluation

The regex-based math evaluator significantly underestimates accuracy (e.g. Opus 4.6: regex 32.5% vs LLM judge 76%). Always use LLM judge for math results.

```bash
python scripts/llm_eval.py results/OckBench_math_gpt-5.2_*.json
python scripts/llm_eval.py results/*.json --model gpt-4o --concurrency 10

# Using local model (e.g. Qwen3-4B via vLLM)
python scripts/llm_eval.py results/*.json --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://localhost:8000/v1 --concurrency 20
```

LLM eval outputs are saved to `results/llm_eval/` as both `*_llm_eval.json` (detailed) and `*_llm_rescored.json` (original format with updated scores).

## Architecture

**Data flow**: `main.py` → `src/utils/parser.py` → `src/core/runner.py` (async, semaphore-based concurrency) → model client + evaluator per problem → `results/` JSON.

**Key components:**

- `src/core/runner.py` — `BenchmarkRunner`: orchestrates loading, calling, evaluating, caching
- `src/core/schemas.py` — Pydantic models (`BenchmarkConfig`, `ProblemResult`, `EvalResult`, etc.)
- `src/core/config.py` — Config loading with priority: CLI args > YAML file > task presets > env vars
- `src/models/` — Provider clients (`OpenAIClient`, `GeminiClient`) extending `BaseModelClient`; adding a new provider means subclassing `BaseModelClient`
- `src/evaluators/` — `MathEvaluator` (regex on boxed/XML/text patterns), `CodeEvaluator` (subprocess execution with timeout), `ScienceEvaluator` (A-D letter extraction); all return `EvalResult`
- `src/utils/token_counter.py` — Uses `tiktoken` when available, else char-based estimate with 1.7× multiplier for non-OpenAI models

**Token tracking** distinguishes: `prompt_tokens`, `answer_tokens`, `reasoning_tokens` (for extended thinking models), `output_tokens` (answer + reasoning), `total_tokens`.

**Retry logic**: exponential backoff (2^attempt, max 30s); non-retryable errors include auth failures, context length exceeded, and content policy violations.

## OpenRouter Support

For models served via OpenRouter (`--base-url` containing "openrouter"), thinking/reasoning is enabled differently than local or native APIs:
- `--enable-thinking true` sends `{"reasoning": {"enabled": true}}` in `extra_body`
- `--reasoning-effort` (for non-reasoning models) sends `{"reasoning": {"effort": "<level>"}}` in `extra_body`

This is handled automatically in `src/models/openai_api.py`.

## Plotting

`scripts/plot/bubble_plot_selected.py` generates accuracy-vs-token bubble plots from `results/model_summary_selected.csv`. Closed-source models use solid-edge circles (sized by tier), open-source models use dashed-edge circles (sized by active params).

## Known Issues

- **Silent empty responses**: API calls (especially via Azure proxy) can return empty `model_response` with `error=None` and all token counts at 0. The cache loader (`_load_cache`) treats these as successful, so `--cache` resume won't retry them. Workaround: merge results from a separate run to fill in the gaps (see `results/merged/`).

## Directory Layout

- `results/` — own benchmark results
- `results/merged/` — merged results (own + peer, own prioritized, peer fills empty responses)
- `results/llm_eval/` — LLM judge re-evaluation outputs
- `peer_results/` — symlink to `/home/zheng/projects/results/` (colleague's results)
- `results/plots/` — generated visualizations (bubble plots, etc.)
- `results/model_summary_selected.csv` — per-model accuracy and avg token summary on the Selected subset
- `scripts/token_variance.py` — computes per-problem output token variance across models, selects high-variance problems
- `scripts/plot/` — plotting scripts
- `cache/` — JSONL cache files for resuming interrupted runs
