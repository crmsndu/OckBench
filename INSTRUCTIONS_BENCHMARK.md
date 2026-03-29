# OckBench Benchmark Instructions

## Setup

```bash
cd /path/to/OckBench
uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 1. Claude Models

Requires `ANTHROPIC_API_KEY`.

| Model | Max Output Tokens |
|-------|-------------------|
| claude-opus-4-6 | 128,000 |
| claude-sonnet-4-6 | 64,000 |
| claude-haiku-4-5-20251001 | 64,000 |

```bash
export ANTHROPIC_API_KEY="your-key"

# All tasks (math, coding, science)
./scripts/run_claude_benchmark.sh

# Specific task
./scripts/run_claude_benchmark.sh math
```

**After the math benchmarks finish, send me all result files before continuing with coding and science.**

## 2. OpenAI Models

Requires `OPENAI_API_KEY`.

| Model | Max Output Tokens | Reasoning Efforts |
|-------|-------------------|-------------------|
| gpt-4.1 | 32,000 | none |
| gpt-5 | 128,000 | low, medium, high |
| gpt-5-mini | 128,000 | low, medium, high |
| gpt-5.4 | 128,000 | none, low, medium, high, xhigh |
| gpt-5.4-mini | 128,000 | none, low, medium, high, xhigh |

```bash
export OPENAI_API_KEY="your-key"

# All models, all efforts, all tasks
./scripts/run_openai_benchmark.sh

# Specific task or model
./scripts/run_openai_benchmark.sh --tasks math
./scripts/run_openai_benchmark.sh --models gpt-5.4
```

## 3. Open-Source Models (SGLang)

No API key needed. Models are served locally via SGLang.

| Model | TP (default) | Context Window |
|-------|-------------|----------------|
| moonshotai/Kimi-K2.5 | 8 | 262,144 |
| zai-org/GLM-5 | 8 | 202,752 |
| MiniMaxAI/MiniMax-M2.5 | 8 | 196,608 |
| Qwen/Qwen3.5-397B-A17B | 8 | 262,144 |
| Qwen/Qwen3.5-122B-A10B | 4 | 262,144 |
| Qwen/Qwen3.5-35B-A3B | 2 | 262,144 |
| Qwen/Qwen3.5-27B | 4 | 262,144 |
| Qwen/Qwen3.5-9B | 2 | 262,144 |
| Qwen/Qwen3.5-4B | 1 | 262,144 |

The script automatically starts/stops SGLang for each model.

```bash
# All models, all tasks
./scripts/run_opensource_benchmark.sh

# Specific models or tasks
./scripts/run_opensource_benchmark.sh --models "Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
./scripts/run_opensource_benchmark.sh --tasks math

# Override tensor parallelism
./scripts/run_opensource_benchmark.sh --tp 4
```

The default TP values may not suit your GPU setup. Edit the `get_tp()` function in the script to adjust per-model TP. Logs are saved to `logs/`.

## Notes

- All scripts use `--cache` for incremental saving. If a run is interrupted, re-run the same command to resume.
- Results are saved to `results/` as JSON files.
- After all runs complete, send me all files in `results/`.
