# OckBench: Project Overview

## Motivation

Current LLM benchmarks (HELM, Chatbot Arena, LM-Eval) evaluate models almost exclusively on output quality. They ignore a critical deployment dimension: **how many tokens does the model spend to reach the answer?**

This matters because:
- Models with similar accuracy can differ by **5x** in token consumption on the same problem
- Every output token costs latency, compute, and money — reasoning tokens especially so
- Smaller models often generate *more* tokens than larger ones to reach worse answers (the "Overthinking Tax"), making them paradoxically more expensive to deploy
- Open-source models are closing the accuracy gap with proprietary models but remain **far less token-efficient**

OckBench fills this gap by jointly measuring accuracy and token efficiency, introducing a unified metric that makes the trade-off visible and quantifiable.

## Core Concepts

### Per-Token Intelligence

A model's quality should be judged not just by *whether* it gets the right answer, but by *how concisely* it reasons to get there. Per-Token Intelligence captures this: the ability to solve complex problems correctly with minimal token expenditure.

### OckScore

The unified evaluation metric that operationalizes Per-Token Intelligence:

```
S_ock = Accuracy - lambda * log(T / C)
```

Where:
- `Accuracy` = percentage of problems solved correctly (0-100)
- `T` = average output tokens per problem
- `C` = 10,000 (normalization constant)
- `lambda` = 10 (penalty coefficient)

Design rationale:
- **Accuracy-first**: correctness is the base term; efficiency is a logarithmic penalty. Correct-and-verbose always outranks wrong-and-concise.
- **Logarithmic scaling**: output lengths vary by orders of magnitude (500 to 50,000+ tokens). A log penalty compresses this range, penalizing unnecessary verbosity without punishing legitimately hard problems that require long reasoning.
- **Calibration**: lambda=10 and C=10,000 were chosen so that the ranking aligns with model-generation ordering within families (GPT-5.4 > GPT-5.2, Gemini-3 > Gemini-2.5).

### Differentiation Filter

Random sampling from existing datasets is uninformative for measuring efficiency — easy problems all get short answers, impossible problems all get long wrong ones. OckBench applies a two-stage filter to select problems where efficiency differences are most visible:

1. **Difficulty Banding**: Keep only problems with 10% < average accuracy < 90% across a diverse model set. This removes trivial and intractable items, focusing on the "reasoning frontier."
2. **Maximizing Token Variance**: From the banded pool, select the top-k problems with the highest cross-model variance in output token count. High variance means multiple valid reasoning paths exist — some efficient, some verbose — directly exposing the efficiency gap.

Result: 200 problems (100 math + 60 coding + 40 science), proportional to each domain's candidate pool size.

## Benchmark Design

### Domains

| Domain | Source Datasets | Problem Count (Full / Selected) | Evaluation Method |
|--------|----------------|--------------------------------|-------------------|
| **Math** | GSM8K, AIME 2024/2025, OlympiadBench, MATH500, Humanity's Last Exam (math), AMO-Bench | 200 / 100 | Answer extraction (regex + LLM judge) |
| **Coding** | MBPP, LiveCodeBench | 974 / 60 | Functional execution against test cases |
| **Science** | ScienceQA, MMLU-STEM, GPQA-Diamond | 198 / 40 | Multiple-choice letter extraction (A/B/C/D) |

### Evaluation Protocol

- **Single-shot prompts** with greedy decoding (temperature=0) for reproducibility
- **Efficiency metric**: total output tokens (reasoning + answer tokens combined), as reported by each provider's API
- **Token counting**: uses each model's native token counts from API responses. Cross-tokenizer divergence is <10% (validated in paper Appendix A.6), so output token count is a fair cross-model comparison metric.

### Token Tracking

The system distinguishes five token categories:
- `prompt_tokens` — input tokens
- `reasoning_tokens` — thinking/reasoning tokens (o1/o3, Claude thinking, Gemini thinking)
- `answer_tokens` — visible output tokens
- `output_tokens` = reasoning_tokens + answer_tokens
- `total_tokens` = prompt_tokens + output_tokens

## Evaluation Pipeline

### Evaluators

Each domain has a dedicated evaluator that extracts the model's answer from free-form text and checks correctness:

**Math** (`MathEvaluator`): Tries extraction patterns in order of specificity — `\boxed{}`, `<answer>` tags, `#### marker`, "the answer is" phrases, and finally the last number in the response. Normalization handles commas, decimals, and string/numeric type coercion. Because regex extraction significantly underestimates accuracy on free-form math responses, an LLM-judge post-hoc re-evaluation step (`scripts/llm_eval.py`) is essential for final results.

**Coding** (`CodeEvaluator`): Extracts Python code from `<solution>` tags, markdown code blocks, or bare function definitions. Executes the extracted code in a sandboxed subprocess with a timeout, running it against the problem's test cases. Reports pass/fail per test case.

**Science** (`ScienceEvaluator`): Extracts a single letter (A/B/C/D) from `<answer>` tags, "the answer is" patterns, boxed notation, or standalone letters. Takes the last match to handle models that deliberate before concluding.

All evaluators return a standardized `EvalResult` with: `is_correct`, `extracted_answer`, `extraction_method`, and optional test-case details.

### LLM-Judge Re-evaluation

The regex-based math evaluator is a known weak point — it cannot handle equivalent expressions, different notations, or answers embedded in prose. `scripts/llm_eval.py` re-evaluates results using an LLM judge (default: gpt-4o-mini) that reads the full model response and ground truth, producing a binary correct/incorrect judgment. This is the authoritative accuracy number for math results.

### Caching and Fault Tolerance

The `--cache <path>` flag enables incremental saving: each completed problem is appended to a JSONL file as it finishes. On restart, completed problems (those without errors) are skipped. This makes long runs resilient to interruptions — particularly important for expensive API calls or SLURM job time limits.

### Concurrency Model

`BenchmarkRunner` uses `asyncio.Semaphore` to bound concurrent API requests (configurable via `--concurrency`). All problems are dispatched as async tasks and gathered. Progress is tracked via `tqdm`.

## Architecture

### Directory Structure

```
main.py                          CLI entry point
src/
  core/
    schemas.py                   Pydantic models (BenchmarkConfig, EvaluationResult, etc.)
    config.py                    Config loading: CLI args > YAML > task presets
    runner.py                    BenchmarkRunner: async orchestration with semaphore concurrency
  models/
    base.py                      BaseModelClient: retry logic, error classification
    openai_api.py                Chat completions (OpenAI, vLLM, SGLang, OpenRouter)
    anthropic_api.py             Anthropic Messages API with streaming SSE
    openai_responses_api.py      OpenAI Responses API (/v1/responses) with streaming
    gemini_api.py                Google Gemini via google-genai SDK
  evaluators/
    math_eval.py                 Regex-based answer extraction (boxed, XML, text patterns)
    code_eval.py                 Subprocess execution with timeout, test case validation
    science_eval.py              A/B/C/D letter extraction
  loaders/
    base.py                      JSONL and MBPP format loaders
  utils/
    parser.py                    Argument parsing and task presets
    prompt_formatter.py          Task-specific prompt templates
    token_counter.py             tiktoken-based estimation for dynamic context budgeting
    logger.py                    Logging setup and filename generation
scripts/
  llm_eval.py                   LLM-judge post-hoc re-evaluation (critical for math)
  token_variance.py             Compute cross-model token variance, select high-variance problems
  merge_results.py              Merge own + peer results (fill empty responses)
  plot/                         Visualization scripts (bubble plots, token distributions)
data/
  OckBench_{math,coding,science}.jsonl           Full datasets
  OckBench_{Math,Coding,Science}_Selected.jsonl  Curated 200-problem subset
```

### Data Flow

```
CLI args / YAML config
  -> BenchmarkConfig (validated by Pydantic)
  -> BenchmarkRunner
       -> DataLoader (JSONL -> List[Problem])
       -> ModelClient.generate() (async, semaphore-bounded concurrency)
       -> Evaluator.evaluate() (per-problem)
       -> EvaluationResult (cached incrementally to JSONL)
  -> ExperimentResult (JSON with per-problem detail + aggregate summary)
```

### Provider Model

All providers subclass `BaseModelClient` and implement a single method: `_call_api(prompt, temperature, max_output_tokens, **kwargs) -> ModelResponse`. The base class owns retry logic (exponential backoff, non-retryable error detection via HTTP status codes) and latency measurement.

| Provider | Class | Transport | Streaming |
|----------|-------|-----------|-----------|
| `chat_completion` | `OpenAIClient` | OpenAI Python SDK (`AsyncOpenAI`) | Yes, via SDK stream |
| `openai-responses` | `OpenAIResponsesClient` | Raw `httpx` SSE to `/v1/responses` | Yes, manual SSE parsing |
| `anthropic` | `AnthropicClient` | Raw `httpx` SSE to `/v1/messages` | Yes, manual SSE parsing |
| `gemini` | `GeminiClient` | `google-genai` SDK (sync, run in executor) | No |

The `chat_completion` provider is the workhorse — it covers OpenAI, vLLM, SGLang, LMDeploy, and OpenRouter through the OpenAI-compatible chat completions API. It requires explicit `--api-key` and `--base-url` (no env var auto-detection). Other providers read API keys from their standard env vars (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

Each provider normalizes token usage into a standardized `TokenUsage` object, separating `reasoning_tokens` from `answer_tokens`. This is provider-specific: OpenAI reports `completion_tokens_details.reasoning_tokens`, Gemini reports `thoughts_token_count`, and Anthropic currently does not separate thinking tokens from output.

### Configuration System

Configuration follows a strict priority chain: **CLI args > YAML config file > task presets**.

Task presets (`math`, `coding`, `science`) set sensible defaults for `dataset_path`, `evaluator_type`, and task-specific parameters (e.g., `execution_timeout` for coding). YAML configs add reproducibility for full experiment setups. CLI args override everything.

Validation happens at the `BenchmarkConfig` Pydantic model level:
- `--max-output-tokens` and `--max-context-window` are mutually exclusive (exactly one required)
- `chat_completion` provider requires both `--api-key` and `--base-url`

### Adding a New Provider

1. Create `src/models/new_provider.py` with a class extending `BaseModelClient`
2. Implement `_call_api()` returning `ModelResponse` with populated `TokenUsage`
3. Add the provider name to the `Literal` type in `BenchmarkConfig.provider`
4. Add a branch in `BenchmarkRunner._create_client()`
5. Add env var mapping in `config.py._apply_env_vars()` if applicable

### Adding a New Evaluator

1. Create `src/evaluators/new_eval.py` with an `evaluate()` method returning `EvalResult`
2. Add a branch in `get_evaluator()` in `math_eval.py`
3. Add a prompt template in `prompt_formatter.py`
4. Add the evaluator type to the CLI choices in `parser.py`

## Known Limitations

- **Regex math evaluation underestimates accuracy** (e.g., Opus 4.6: regex 32.5% vs LLM judge 76%). Always use `scripts/llm_eval.py` for math results.
- **Silent empty responses**: some API proxies return empty responses with no error. The cache loader treats these as completed, so `--cache` resume won't retry them. Workaround: merge results from a separate run.
- **Tokenizer divergence**: output token counts come from each provider's native tokenizer. Cross-tokenizer divergence is bounded at ~10% (paper Appendix A.6), but this is a known approximation.

## Research Directions

- Expand model coverage as new models release (continuous leaderboard updates)
- Investigate training-time interventions for token efficiency (RL with token penalties, model interpolation)
- Study the relationship between model scale, reasoning density, and the Overthinking Tax
- Extend to additional domains (agentic tasks, multi-turn reasoning)

---

## Experiment Log

Reverse chronological. Each entry records what changed, why, and any notable outcomes.

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
**Infrastructure:** Qwen models served locally via SGLang on GPUs; Gemini via GCP; MiniMax via their API
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
