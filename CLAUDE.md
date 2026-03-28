# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OckBench is an efficiency-aware benchmarking suite for LLMs that evaluates both accuracy and token usage. The key innovation is the **OckScore** metric:

```
OckScore = Accuracy - 10 * log(AvgTokens / 10000 + 1)
```

Three task types are supported: **math**, **coding**, and **science**, each with 200/200/100 problems in `data/`.

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

API keys via environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `API_KEY` / `API_BASE_URL` for generic providers.

## Running Benchmarks

```bash
# OpenAI provider
python main.py --provider openai --model gpt-4o --task math --max-output-tokens 8192

# Local model via vLLM
python main.py --provider generic --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1 --max-context-window 40960 --task math

# Using YAML config
python main.py --config configs/openai.yaml

# Resume interrupted run
python main.py ... --cache cache/run.jsonl
```

`--max-output-tokens` sets a fixed output budget; `--max-context-window` dynamically computes budget as `context_window - prompt_tokens`.

Results are saved to `results/` as JSON with per-problem details and aggregate summary statistics.

## LLM-based Re-evaluation

```bash
python scripts/llm_eval.py results/OckBench_math_gpt-5.2_*.json
python scripts/llm_eval.py results/*.json --model gpt-4o --concurrency 10
```

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
