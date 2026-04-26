# OckBench Benchmark Instructions (Collaborator Runs)

Thanks for the help. Below is everything you need to run OckBench on the
three models you have NVIDIA API access to:

| Model             | Variants                                         |
|-------------------|--------------------------------------------------|
| `gpt-5.5`         | reasoning efforts: `none, minimal, low, medium, high, xhigh` |
| `claude-opus-4-7` | single run                                       |
| `deepseek-v4`     | `deepseek-v4-flash`, `deepseek-v4-pro`           |

Each run evaluates the **Selected subset** (100 math + 60 coding + 40 science =
200 problems). No full-dataset runs needed this round.

## 1. Setup

Clone the repo and create the env (one time):

```bash
cd /path/to/OckBench
uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -e .
```

## 2. Configure the Nvidia API

I'm assuming your Nvidia API is a single OpenAI-compatible endpoint (chat
completions format) that serves all three model families under one base URL
and one API key. Export:

```bash
export OCKBENCH_API_KEY="your-company-key"
export OCKBENCH_BASE_URL="https://your.company.api/v1"   # whatever your endpoint is
```

## 3. Run

One script, runs everything:

```bash
./scripts/run_haokang.sh
```

**Before the first run, open `scripts/run_haokang.sh` and edit the `RUNS`
table at the top** so the model IDs match what your NVIDIA proxy actually
exposes. My guesses are `gpt-5.5`, `claude-opus-4-7`, `deepseek-v4-flash`,
`deepseek-v4-pro`; your proxy may use `anthropic/claude-opus-4-7`,
`deepseek/deepseek-v4-pro`, dated suffixes, etc. Only the first column of the
table matters for the API call; the second column is just a cache/log tag.

What the script does per run:

- `--provider chat_completion` (OpenAI-compatible chat completions)
- `--max-output-tokens 128000`, `--concurrency 10`
- Selected subset datasets under `data/OckBench_{Math,Coding,Science}_Selected.jsonl`
- For GPT-5.5: passes `--reasoning-effort <level>`. Claude and DeepSeek use default config.
- Each run gets its own `--cache cache/<tag>-<task>-selected.jsonl`, so **Ctrl-C
  and rerun** picks up where it left off.

Total runs:
`GPT-5.5: 6 efforts × 3 tasks = 18` + `Claude: 3` + `DeepSeek: 2 × 3 = 6` = **27 runs**.

## 4. Things that might go wrong (and what to do)

**Reasoning effort `none` or `minimal` rejected by the proxy.** If you see
400-ish errors for those levels on GPT-5.5, delete those lines from the
`RUNS` table and rerun the script. The cache keeps the efforts that already
succeeded, so they won't repeat.

**Extended thinking on Claude / DeepSeek.** The script doesn't set
`--enable-thinking`, so it runs with whatever the proxy's default is. If your
proxy requires an explicit toggle and you know what it expects, add
`--enable-thinking true` (or false) to the corresponding `run_one` call. If in
doubt, ask me — don't guess.

**Empty responses.** Sometimes a provider-proxy combination returns an empty
string with no error. OckBench now surfaces these as errors in the cache so
they get retried on the next run. If you still see rows with `"error": null`
and empty `model_response`, let me know which models — we'll filter them
server-side later.

## 5. What to send me

After everything finishes (or whenever you want to dump a partial batch), send me:

- `results/OckBench_math_selected_*.json`
- `results/OckBench_coding_selected_*.json`
- `results/OckBench_science_selected_*.json`
- `logs/haokang/*.log` (optional, only if something looked weird)

The `results/` JSONs contain per-problem detail + aggregate summary. That's
what I need to merge into the leaderboard. `cache/` files are just for resume
and don't need to be shipped.

## 6. Quick sanity check before the long run

If you want to confirm the plumbing works before committing to 27 runs,
comment out all but one line in the `RUNS` table (e.g. keep only
`gpt-5.5 ... medium`) and run `./scripts/run_haokang.sh`. That's 3 runs
(~15-30 min depending on rate limits). When it finishes there should be
three JSONs under `results/` and the printed summary at the end of each run
should show a non-zero accuracy. If those look right, uncomment the rest of
the `RUNS` table and rerun — the cache keeps the sanity-check runs.

Ping me with any errors — especially around model IDs or auth, those are the
most likely snags.
