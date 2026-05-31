# LongBench v1 — HighSNR Context Optimizer

Benchmark evaluating the [HighSNR Context Optimizer](https://www.high-snr.com/#api) API against
[LongBench v1](https://github.com/THUDM/LongBench/tree/main/LongBench) QA tasks.

The optimizer compresses long documents to a token budget before passing them to a
downstream LLM. We measure whether answer quality is preserved after compression.

## Results

Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) via AWS Bedrock · 200 samples per dataset · seed=42

### HotpotQA — QA F1 (multi-document QA)

| Config                  |   10% |   20% |   30% |   40% |   50% |   60% | 100% (full) |
|-------------------------|------:|------:|------:|------:|------:|------:|------------:|
| generic (no hint)       | 42.09 | 52.85 | 60.23 | 63.12 | 64.08 | 64.26 |       66.26 |
| biased (with hint)      | 58.21 | 64.88 | 65.00 | 67.49 | 67.53 | 69.10 |       66.26 |
| random                  | 34.72 | 44.68 | 51.12 | 55.53 | 58.16 | 58.22 |       66.26 |

At **50–60% budget with query-aware compression**, the optimizer matches or exceeds
full-document F1. Biased mode at 60% scores 69.10 — above the 66.26 full-document baseline.

### Qasper — QA F1 (single-document QA, scientific papers)

| Config                  |   10% |   20% |   30% |   40% |   50% |   60% | 100% (full) |
|-------------------------|------:|------:|------:|------:|------:|------:|------------:|
| generic (no hint)       | 23.70 | 31.85 | 35.27 | 40.96 | 42.10 | 44.22 |       50.69 |
| biased (with hint)      | 37.08 | 44.86 | 46.84 | 48.82 | 48.98 | 48.41 |       50.69 |
| random                  | 22.84 | 32.62 | 36.70 | 37.88 | 38.04 | 41.14 |       50.69 |

Biased mode at **50% budget scores 48.98 — 97% of full-document F1** using half the tokens.

### Modes

| Mode | Description |
|---|---|
| `full` | Full document, no compression (baseline) |
| `api_generic` | HighSNR `/v2/optimize`, no `context_hint` |
| `api_biased` | HighSNR `/v2/optimize` with `context_hint` set to the question |
| `random` | Random chunk selection at the same token budget (pre-computed baseline; not reproducible via `longbench_v1_run.py`) |

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
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"

# Optional — defaults to https://api.high-snr.com/v2/optimize
# export CO_API_URL="..."
```

### 3. Dry run (3 samples, sanity check)

```bash
uv run python longbench_v1_run.py \
  --datasets qasper hotpotqa \
  --samples 3 \
  --levels 0.5 \
  --modes full api_generic api_biased \
  --providers anthropic
```

### 4. Full run (reproduce published results)

The published results used Claude Sonnet 4.5 via AWS Bedrock. Running via
the direct Anthropic API with the same model will produce comparable scores.
The `random` baseline is pre-computed only and cannot be reproduced via this script.

```bash
uv run python longbench_v1_run.py \
  --run-name-prefix co_sonnet \
  --datasets qasper hotpotqa \
  --samples 200 \
  --modes api_generic api_biased \
  --levels 0.1 0.2 0.3 0.4 0.5 0.6 \
  --providers anthropic

# Full-doc baseline (level 1.0, mode full)
uv run python longbench_v1_run.py \
  --run-name-prefix co_sonnet \
  --datasets qasper hotpotqa \
  --samples 200 \
  --modes full \
  --levels 1.0 \
  --providers anthropic
```

The runner is resumable — it skips samples already written to the output JSONL.
LongBench data is downloaded automatically on first run (~320 MB, cached under `data_cache/`).

### 5. Evaluate

```bash
uv run python eval_longbench.py --prefix co_sonnet
```

Pre-computed results are already in `results/` — you can run the evaluator without
re-running inference.

## Methodology

- **Dataset**: LongBench v1, `qasper` and `hotpotqa` splits (200 samples each, seed=42).
- **Budget**: `int(orig_tokens × level)` tokens, computed with tiktoken `cl100k_base`.
- **Skipped**: samples whose raw context exceeds 200,000 characters (service input limit).
  No input truncation is applied — samples either fit or are skipped.
- **Downstream LLM**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`),
  `temperature=0`, `max_tokens` per LongBench config.
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
└── results/                     # pre-computed predictions
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__full__L1_0/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_1/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_2/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_3/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_4/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_5/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_generic__L0_6/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_1/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_2/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_3/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_4/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_5/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__api_biased__L0_6/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_1/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_2/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_3/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_4/
    ├── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_5/
    └── co_sonnet__bedrock__claude-sonnet-4-5-20250929__random__L0_6/
```

Each results directory contains one file per dataset:
- `{dataset}.jsonl` — predictions (compatible with the official LongBench evaluator)

### Output schema: `{dataset}.jsonl`

| Field | Type | Description |
|---|---|---|
| `_id` | string | Sample ID from LongBench |
| `dataset` | string | Sub-dataset name (e.g. `hotpotqa`) |
| `mode` | string | `full`, `api_generic`, `api_biased`, or `random` |
| `level` | float | Compression target as fraction of original tokens (e.g. `0.5` = 50%) |
| `pred` | string | LLM-generated answer |
| `answers` | list[str] | Ground-truth answer(s); F1 = max over all |
| `all_classes` | list[str] \| null | Classification labels (null for QA datasets) |
| `length` | int | Original context length in **characters** (from LongBench metadata) |

## Attribution

Evaluation metrics and prompt templates are from
[THUDM/LongBench](https://github.com/THUDM/LongBench/tree/main/LongBench), MIT license.
