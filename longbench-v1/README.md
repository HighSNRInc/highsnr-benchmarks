# LongBench v1 — HighSNR Context Optimizer

Benchmark evaluating the [HighSNR Context Optimizer](https://www.high-snr.com/#api) API against
[LongBench v1](https://github.com/THUDM/LongBench/tree/main/LongBench) QA tasks.

The optimizer compresses long documents to a token budget before passing them to a
downstream LLM. We measure whether answer quality is preserved after compression.

## Results

All runs: GPT-4o as the downstream LLM, 200 samples per dataset, seed=42.

### HotpotQA — QA F1 (multi-document QA)

| Config                  |   50% |   60% |   70% |   80% |   90% | 100% (full) |
|-------------------------|------:|------:|------:|------:|------:|------------:|
| generic (no hint)       | 63.78 | 66.56 | 65.88 | 66.15 | 69.52 |       69.71 |
| biased (with hint)      | 68.22 | 67.61 | 67.95 | 68.48 | 71.57 |       69.71 |

At **90% budget with query-aware compression**, the optimizer exceeds full-document F1.
Even at 50% budget, biased mode retains 97.9% of full-document performance.

### Qasper — QA F1 (single-document QA, scientific papers)

| Config                  |   50% |   60% |   70% |   80% |   90% | 100% (full) |
|-------------------------|------:|------:|------:|------:|------:|------------:|
| generic (no hint)       | 35.35 | 37.00 | 39.50 | 42.23 | 44.87 |       47.22 |
| biased (with hint)      | 41.27 | 42.66 | 44.04 | 44.59 | 46.25 |       47.22 |

At **90% budget with query-aware compression**, the optimizer retains 97.9% of full-document
F1 on dense scientific QA.

### Actual compression ratios

The optimizer respects the requested budget level closely.
Ratios are averaged across both `api_generic` and `api_biased` runs.

| Target | HotpotQA actual (mean) | Qasper actual (mean) |
|--------|------------------------|----------------------|
| 50%    | 49.8%                  | 50.2%                |
| 60%    | 59.7%                  | 60.1%                |
| 70%    | 69.7%                  | 70.2%                |
| 80%    | 79.6%                  | 80.1%                |
| 90%    | 89.4%                  | 90.0%                |

### Modes

| Mode | Description |
|---|---|
| `full` | Full document, no compression (baseline) |
| `api_generic` | HighSNR `/v1/optimize`, no `context_hint` |
| `api_biased` | HighSNR `/v1/optimize` with `context_hint` set to the question |

## Reproduce

### 1. Install

```bash
git clone https://github.com/HighSNRInc/highsnr-benchmarks
cd highsnr-benchmarks/longbench-v1
uv sync
```

### 2. Set environment variables

```bash
export CO_API_KEY="your-highsnr-api-key"   # https://console.high-snr.com
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o"

# Optional — defaults to https://api.high-snr.com/v1/optimize
# export CO_API_URL="..."
```

### 3. Dry run (3 samples, sanity check)

```bash
uv run python longbench_v1_run.py \
  --datasets qasper hotpotqa \
  --samples 3 \
  --levels 0.7 \
  --modes full api_generic api_biased \
  --providers openai
```

### 4. Full run (reproduce published results)

```bash
uv run python longbench_v1_run.py \
  --run-name-prefix co_v1_validate \
  --datasets qasper hotpotqa \
  --samples 200 \
  --modes api_generic api_biased \
  --levels 0.5 0.6 0.7 0.8 0.9 \
  --dump-api-output \
  --providers openai

# Full-doc baseline (level 1.0, mode full)
uv run python longbench_v1_run.py \
  --run-name-prefix co_v1_validate \
  --datasets qasper hotpotqa \
  --samples 200 \
  --modes full \
  --levels 1.0 \
  --providers openai
```

The runner is resumable — it skips samples already written to the output JSONL.
LongBench data is downloaded automatically on first run (~320 MB, cached under `data_cache/`).

### 5. Evaluate

```bash
uv run python eval_longbench.py --prefix co_v1_validate
```

Pre-computed results are already in `results/` — you can run the evaluator without
re-running inference.

## Methodology

- **Dataset**: LongBench v1, `qasper` and `hotpotqa` splits (200 samples each, seed=42).
- **Budget**: `int(orig_tokens × level)` tokens, computed with tiktoken `cl100k_base`.
- **Skipped**: samples whose raw context exceeds 200,000 characters (service input limit).
  No input truncation is applied — samples either fit or are skipped.
- **Downstream LLM**: GPT-4o (`temperature=0`, `max_tokens` per LongBench config).
- **Metric**: token-level F1 from the official LongBench v1 evaluation code.
- **`context_hint`** (biased mode): the question string, truncated to 2,000 characters.

## Files

```
longbench-v1/
├── longbench_v1_run.py          # inference runner
├── eval_longbench.py            # evaluator
├── pyproject.toml               # dependencies
├── longbench/
│   ├── metrics.py               # evaluation metrics (from THUDM/LongBench, MIT license)
│   └── config/
│       ├── dataset2prompt.json  # per-dataset prompt templates
│       └── dataset2maxlen.json  # per-dataset max generation lengths
└── results/                     # pre-computed predictions + API dumps
    ├── co_v1_validate__openai__gpt-4o__full__L1_0/
    ├── co_v1_validate__openai__gpt-4o__api_generic__L0_5/
    ├── co_v1_validate__openai__gpt-4o__api_generic__L0_6/
    ├── co_v1_validate__openai__gpt-4o__api_generic__L0_7/
    ├── co_v1_validate__openai__gpt-4o__api_generic__L0_8/
    ├── co_v1_validate__openai__gpt-4o__api_generic__L0_9/
    ├── co_v1_validate__openai__gpt-4o__api_biased__L0_5/
    ├── co_v1_validate__openai__gpt-4o__api_biased__L0_6/
    ├── co_v1_validate__openai__gpt-4o__api_biased__L0_7/
    ├── co_v1_validate__openai__gpt-4o__api_biased__L0_8/
    └── co_v1_validate__openai__gpt-4o__api_biased__L0_9/
```

Each results directory contains:
- `{dataset}.jsonl` — predictions (compatible with the official LongBench evaluator)
- `{dataset}.api_dump.jsonl` — per-sample API call details: input/output token counts,
  selected chunks, context hint used

### Output schema: `{dataset}.jsonl`

| Field | Type | Description |
|---|---|---|
| `_id` | string | Sample ID from LongBench |
| `dataset` | string | Sub-dataset name (e.g. `hotpotqa`) |
| `mode` | string | `full`, `api_generic`, or `api_biased` |
| `level` | float | Compression target as fraction of original tokens (e.g. `0.5` = 50%) |
| `pred` | string | LLM-generated answer |
| `answers` | list[str] | Ground-truth answer(s); F1 = max over all |
| `all_classes` | list[str] \| null | Classification labels (null for QA datasets) |
| `length` | int | Original context length in **characters** (from LongBench metadata) |

### Output schema: `{dataset}.api_dump.jsonl`

| Field | Type | Description |
|---|---|---|
| `_id` | string | Sample ID from LongBench |
| `dataset` | string | Sub-dataset name |
| `mode` | string | Compression mode used |
| `level` | float | Compression target fraction |
| `provider` | string | LLM provider (`openai` or `anthropic`) |
| `llm_model` | string | LLM model name (e.g. `gpt-4o`) |
| `api_called` | bool | Whether the optimizer API was called (`false` for `full`) |
| `api_version` | string | Optimizer API version (`v1`) |
| `api_input_tokens` | int | Original context token count (tiktoken `cl100k_base`) |
| `budget_tokens` | int | Target token budget: `floor(api_input_tokens × level)` |
| `used_context_tokens` | int | Actual tokens in optimized context (may exceed budget due to chunk boundaries) |
| `prompt_tokens` | int | Total tokens in the final prompt sent to the LLM (context + question + template) |
| `context_hint` | string \| null | Question passed as hint (`api_biased` only; null otherwise) |
| `selected_chunks` | list[str] | Text chunks selected by the v1 API (empty list for non-API modes) |

## Attribution

Evaluation metrics and prompt templates are from
[THUDM/LongBench](https://github.com/THUDM/LongBench/tree/main/LongBench), MIT license.
