# Experiment Log

Reverse chronological. Each entry records what changed, why, and any notable outcomes.

### 2026-04-24 — DeepSeek-v4 + proxy-routed open-weight wave

**New runs on Selected subset** (100 math + 60 coding + 40 science):

- **DeepSeek direct API**: `deepseek-v4-flash` and `deepseek-v4-pro` at `https://api.deepseek.com`, thinking on, `reasoning_effort=max`, `max_output_tokens=384000`, `concurrency=5`.
- **Third-party chat-completion proxy** at `concurrency=1`:
  - `google/gemma-4-31b-it` (`max_ctx=65536`)
  - `qwen/qwen3-5-397b-a17b` (first pass `max_out=81920`, rerun at `max_ctx=262144`)
  - `qwen/qwen3.6-35b-a3b`, `qwen3.6-27b` (`max_ctx=131072`, endpoint-capped)
  - `moonshotai/kimi-k2.6` (first pass `max_out=131072`, rerun at `max_ctx=262144`)
  - `zai-org/glm-5.1` (first pass `max_out=131072`, rerun at `max_ctx=202752`)
  - `minimaxai/minimax-m2.7` (`max_out=131072`)

**Code changes** (`55bddf0`):
- `openai_api.py`: direct-DeepSeek branch sending `{"thinking":{"type":"enabled|disabled"}}` in `extra_body` and stripping `temperature`/`top_p`/`presence_penalty`/`frequency_penalty` when thinking is on (docs say these are silently ignored in thinking mode).
- `openai_api.py`: when the stream ends with empty text, set `response.error` to one of `empty_response_length_finish` / `empty_response_reasoning_only` / `empty_response_no_stream`. This turns previously-silent empty rows (which the cache loader treated as successful and never retried) into retryable errors.
- `schemas.py` + `runner.py`: `finish_reason` field added to `EvaluationResult` and propagated from `ModelResponse` for both success and error paths.
- `anthropic_api.py`: captured `thinking_delta` text and estimated `reasoning_tokens = len(thinking_text) / 4` so the answer/reasoning token split is populated for Claude models with `thinking: adaptive`.
- Three run wrappers: `scripts/run_deepseek_v4.sh`, `scripts/run_proxy_v2.sh`, `scripts/run_proxy_v2_ctx.sh`.

**Empty-response investigation** (diagnosis):

The proxy produced silent empty responses at rates up to ~50 % on the hardest math items for several reasoning models. Inspecting the caches revealed three cache signatures — all the same underlying event (**budget-exhausted thinking: the full context budget spent on `reasoning_content`, not a single `content` delta emitted**) expressed differently in how the proxy serialized `usage`:

| Mode | Signature | Cause |
|---|---|---|
| A | `reasoning_tokens > 0, answer_tokens = 0, text=""` | OpenAI-style subset accounting: all of `completion_tokens` went to reasoning |
| B | all-zero usage, `text=""` | stream closed without ever emitting usage (observed on DeepSeek on a handful of hard AMO items) |
| C | `answer_tokens > 0` (even `>0`), `text=""` | proxy reports `reasoning_tokens > completion_tokens`, so `answer = completion − reasoning` went negative or mis-positive |

Client fix surfaces all three as `empty_response_*` errors so `--cache` resume retries them.

**Proxy caveats discovered:**
- `max_total_tokens` (input + output combined) is hard-capped per model. Observed caps: gemma-4-31b-it = 65,536; qwen3.6-27b / qwen3.6-35b-a3b = 131,072; qwen3-5-397b-a17b silently accepts >256K but returns empty on trivial prompts.
- Per-model rate limits on this key: 100 rpm / 100K tpm. Not hit under `concurrency=1` runs.
- `reasoning_content` is emitted as a separate delta field (not `content`); our streaming client already reads only `delta.content`, so reasoning text is correctly dropped from the answer.
- Sporadic **504 Gateway Timeout** from an nginx front-end (raw HTML, not LiteLLM JSON), concentrated on `qwen3.6-35b-a3b` in two short bursts totaling 11 rows. Transient — same prompts succeed on replay. Our client retries 3× with backoff 1s/2s/nil, which is too short for a backend outage; rows land as errored and are refilled by cache-resume.

**DeepSeek balance incident:** account balance exhausted mid-run (HTTP 402 "Insufficient Balance", which is not in the client's non-retryable list). Every subsequent request burned 3 retries before erroring. `ds-flash` finished math (99 valid) before the balance ran out; `ds-pro` got 59 valid math before hitting it; coding and science for both are entirely 402-errored and await a topup + `--cache` resume.

### 2026-04-22 — Codebase cleanup and provider consolidation

**Changes:**
- Merged `openai` + `generic` providers into single `chat_completion` provider
- `chat_completion` now requires explicit `--api-key` and `--base-url` (no env var auto-detection)
- Replaced `setup.py` + `requirements.txt` with `pyproject.toml`
- Fixed circular imports by slimming all `__init__.py` files
- Fixed double retry stacking in OpenAI client (SDK max_retries=0)
- Fixed httpx timeout semantics for streaming (read timeout = per-chunk gap)
- Stripped redundant docstrings across all source files (-1500 lines)
- Added ruff linting, all checks pass

**Motivation:** Repo was unprofessional — two redundant providers, circular imports, bloated docstrings, no proper packaging. Cleaned up for open-source release.

### 2026-04-01 — DeepSeek-V3.2 evaluation (with and without thinking)

**Models:** DeepSeek-V3.2, DeepSeek-V3.2 (thinking mode)
**Tasks:** math, coding, science (full dataset)
**Notes:** Tested both standard and thinking modes via DeepSeek API. Thinking mode significantly improves accuracy (15.5% → 35.5% overall on Selected) but at 26x more tokens.

### 2026-03-31 — Qwen3.5 family + Gemini 3.1 Flash Lite + MiniMax-M2.7

**Models:** Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Qwen3.5-9B, Gemini-3.1-Flash-Lite, MiniMax-M2.7
**Tasks:** math, coding, science (full dataset + Selected subset for Qwen models)
**Infrastructure:** Qwen models served locally via SGLang on local GPUs; Gemini via GCP; MiniMax via their API
**Notes:** Qwen3.5 series evaluated across 4 model sizes to study the Overthinking Tax — larger models are both more accurate and more token-efficient, confirming the scaling hypothesis.

### 2026-03-30 — Qwen-235B + Selected subset runs

**Models:** Qwen-235B (via third-party chat-completion proxy), Qwen3.5-35B-A3B, Qwen3.5-9B
**Tasks:** math, coding, science
**Notes:** First runs on the third-party chat-completion proxy. Qwen-235B selected as the baseline open-weight large model. Selected subset runs started for Qwen3.5-35B and 9B.

### 2026-03-29 — Initial large-scale evaluation wave

**Models:** GPT-5.4 (all reasoning efforts), Claude Opus 4.6, Claude Sonnet 4.6, Kimi-K2.5, GLM-5, MiniMax-M2.7
**Tasks:** math, coding, science (full dataset)
**Infrastructure:** OpenAI via direct API; Claude via Azure proxy; Kimi, GLM, MiniMax via their native APIs
**Notes:** First complete evaluation across all three domains. Claude models ran through Azure proxy which caused silent empty responses — required merge with peer results to fill gaps. GPT-5.4 emerged as the OckScore leader.

### 2026-03-28 — Benchmark framework and initial testing

**Changes:** Initial commit of OckBench benchmark suite — core runner, evaluators, model clients, data loaders
**Models:** Claude Sonnet 4.6 (mini coding test)
**Notes:** Framework validated on small coding subset before scaling to full evaluation.

### Pre-2026-03-28 — Dataset construction and item selection

**Changes:**
- Curated candidate pools: math (GSM8K, AIME, OlympiadBench, MATH500, HLE-math, AMO-Bench), coding (MBPP, LiveCodeBench), science (ScienceQA, MMLU-STEM, GPQA-Diamond)
- Ran pilot evaluations across a diverse model set to collect per-problem accuracy and token statistics
- Applied Differentiation Filter: difficulty banding (10% < acc < 90%) + top-k by token variance
- Produced final Selected subset: 200 problems (100 math + 60 coding + 40 science)
- `scripts/token_variance.py` implements the selection pipeline
