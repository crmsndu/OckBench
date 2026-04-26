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
uv pip install -e .
```

Always activate the virtual environment before running any commands:

```bash
source .venv/bin/activate
```

The `chat_completion` provider requires explicit `--api-key` and `--base-url` (no env var auto-detection). Other providers (`gemini`, `anthropic`) read their API keys from env vars (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

## Running Benchmarks

```bash
# OpenAI
python main.py --model gpt-4o --api-key $OPENAI_API_KEY --base-url https://api.openai.com/v1 \
    --task math --max-output-tokens 8192

# Local model via vLLM
python main.py --model Qwen/Qwen3-4B --api-key dummy --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task math

# OpenRouter
python main.py --model openai/gpt-4o-mini --api-key $OPENROUTER_API_KEY \
    --base-url https://openrouter.ai/api/v1 --task math --max-output-tokens 8192

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

## DeepSeek Direct API

When `--base-url` contains `deepseek.com` (`https://api.deepseek.com`):
- `--enable-thinking true|false` sends `{"thinking": {"type": "enabled|disabled"}}` in `extra_body`. Thinking is on by default server-side.
- `--reasoning-effort <level>` sends `reasoning_effort` in `extra_body` (supported levels per docs: `high` default, `max` for complex agents).
- Per DeepSeek docs, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` are silently ignored in thinking mode. The client strips them from the request when thinking is on so we don't send misleading parameters.
- Env var for API key: `DEEPSEEK_API_KEY` (not auto-detected under `chat_completion` — pass explicitly via `--api-key`).
- Pricing note: **HTTP 402 "Insufficient Balance" is not in the non-retryable list**, so an exhausted account retries 3× per problem before erroring out. Monitor balance at `GET /user/balance` with a Bearer header.

## Chat-completion endpoint quirks

Some runs route through a third-party OpenAI-compatible chat-completions endpoint (base URL and key supplied via `--base-url` / `--api-key`). Quirks to know:

- `max_total_tokens` (input + output combined) is hard-capped per model at the server's `max_model_len`. Caps observed so far: gemma-4-31b-it = 65,536; qwen3.6-27b / qwen3.6-35b-a3b = 131,072; kimi-k2.6 / glm-5.1 reject `max_tokens>=200K` cleanly, accept 262,144 silently and return empty choices (probe only — real work at 200K is fine).
- Prefer `--max-context-window N` over `--max-output-tokens N` when the endpoint enforces a combined cap. OckBench will compute `output_budget = N − input_tokens − 256` dynamically.
- Reasoning models stream `reasoning_content` deltas separately from `content`; our client only reads `delta.content`, so the reasoning text is automatically dropped from the answer.
- Reasoning models can spend the entire context budget inside `reasoning_content` on hard items and emit zero content deltas. Since April 2026 the client surfaces these as `empty_response_length_finish` / `empty_response_reasoning_only` errors (see Known Issues).
- Sporadic **504 Gateway Timeout** (raw HTML body from an nginx front-end) appears during brief backend instability. The client retries 3× with 1s/2s backoff which is too short for a real outage; affected rows land as errored and are refilled on `--cache` resume.

## Known Issues

- **Silent empty responses (historical)**: Before commit `55bddf0`, streaming calls could return `model_response=""` with `error=None` and the cache loader would treat them as successful, so `--cache` resume never retried them. The dominant cause on reasoning models routed through third-party chat-completion proxies was **budget-exhausted thinking**: the full context budget spent emitting `reasoning_content` with zero `content` deltas. The client now sets `response.error` to one of:
  - `empty_response_length_finish` — `finish_reason=length` with `text=""`
  - `empty_response_reasoning_only` — `reasoning_tokens>0` with `text=""` and a non-length finish
  - `empty_response_no_stream` — stream closed without any chunks
  Rows with these errors are filtered out of `completed_ids` on cache load, so `--cache` resume retries them automatically. Pre-existing silent-empty rows in older caches still need manual cleanup (drop rows where `error is None and model_response == ""`) before resume will pick them up.

- **Negative `answer_tokens` in token accounting**: On some proxy-routed models the `usage` object reports `reasoning_tokens > completion_tokens` (not the OpenAI subset convention). Our client computes `answer_tokens = completion_tokens − reasoning_tokens`, which can go negative. Total / output token counts remain correct; only the answer/reasoning split is distorted.

- **Anthropic `thinking_tokens` are estimated**: The direct Anthropic Messages API's `message_delta.usage` only reports total `output_tokens`, not a thinking/text split. The client accumulates `thinking_delta` text and estimates `reasoning_tokens = len(thinking_text) // 4`, then back-solves `answer_tokens = output_tokens − reasoning_tokens`. This is approximate.

## Helper Scripts

- `scripts/run_deepseek_v4.sh <model>` — runs one DeepSeek v4 variant across math/coding/science on the Selected subset (thinking on, `reasoning_effort=max`, `max_output_tokens=384000`, `c=5`, caching enabled).
- `scripts/run_proxy_v2.sh <model> <short_name> <max_output_tokens>` — runs one proxy-routed model across math/coding/science on the Selected subset at `c=1` with `--max-output-tokens`. Requires `OCKBENCH_API_KEY` and `OCKBENCH_BASE_URL` exported.
- `scripts/run_proxy_v2_ctx.sh <model> <short_name> <max_context_window>` — same but with `--max-context-window` for models whose deployment enforces a combined input+output cap.

## Plotting

`scripts/plot/bubble_plot_selected.py` generates accuracy-vs-token bubble plots from `results/model_summary_selected.csv`. Closed-source models use solid-edge circles (sized by tier), open-source models use dashed-edge circles (sized by active params).

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
