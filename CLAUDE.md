# CLAUDE.md

High-level guidance for Claude Code working in this repository. For project motivation, design rationale, and the experiment log, see `PROJECT.md`.

## Project

OckBench is an efficiency-aware LLM benchmark. It scores models by both accuracy and token usage via OckScore:

```
OckScore = Accuracy − 10 · log(AvgTokens / 10000 + 1)
```

Three tasks (math / coding / science) with full pools in `data/OckBench_{math,coding,science}.jsonl` and a 200-problem Selected subset in `data/OckBench_{Math,Coding,Science}_Selected.jsonl`.

## Setup

```bash
uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -e .
```

Always activate the venv before running anything.

## Running benchmarks

```bash
python main.py --provider chat_completion --model <id> \
    --api-key <key> --base-url <url> \
    --task math --max-output-tokens 8192 \
    --cache cache/<run>.jsonl
```

- `chat_completion` requires explicit `--api-key` and `--base-url` (no env-var auto-detection). `gemini` and `anthropic` providers read `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`.
- `--max-output-tokens` vs `--max-context-window` are mutually exclusive; prefer the latter when the endpoint caps input+output combined.
- `--cache` makes runs resumable: re-run the same command to pick up where it left off.
- YAML presets live in `configs/`.

## Rules

- **Always use the LLM judge for math accuracy.** Regex extraction underestimates it severely (Opus 4.6: 32.5% regex vs 76% LLM judge). Run `scripts/llm_eval.py <results.json>` after every math run; treat `*_llm_rescored.json` as authoritative.
- **Never commit API keys, base URLs with internal hostnames, or anything under `results/` / `cache/` / `logs/`** (all gitignored; keep it that way).

## Architecture

`main.py` → `src/utils/parser.py` → `src/core/runner.py` (async, semaphore-bounded) → `src/models/<provider>.py` client + `src/evaluators/<task>.py` → JSON in `results/`.

- `src/core/runner.py` — `BenchmarkRunner`
- `src/core/schemas.py` — Pydantic models (`BenchmarkConfig`, `ModelResponse`, `EvaluationResult`, `ExperimentResult`, `TokenUsage`)
- `src/core/config.py` — CLI > YAML > task preset priority
- `src/models/{openai_api,openai_responses_api,anthropic_api,gemini_api}.py` — all extend `BaseModelClient`
- `src/evaluators/{math,code,science}_eval.py` — return `EvalResult`
- `src/utils/{parser,prompt_formatter,token_counter,logger}.py`
- `src/loaders/base.py` — JSONL / MBPP loaders

Adding a provider: subclass `BaseModelClient`, implement `_call_api`, add a branch in `BenchmarkRunner._create_client`, extend the `provider` literal in `BenchmarkConfig`.

## Directory layout

- `data/` — datasets (full pools + Selected subset)
- `results/` — own benchmark JSONs (gitignored)
- `results/merged/` — own + peer merged, peer fills empty responses
- `results/llm_eval/` — LLM-judge re-eval outputs
- `results/plots/`, `results/model_summary_selected.csv` — summary artifacts
- `peer_results/` — symlink to a colleague's `results/` directory
- `cache/` — JSONL resume caches (gitignored)
- `configs/` — YAML presets
- `scripts/` — helpers (`llm_eval.py`, `token_variance.py`, `merge_results.py`, `plot/`, per-provider run wrappers)
