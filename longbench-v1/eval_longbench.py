#!/usr/bin/env python3
"""
Evaluate LongBench prediction runs and print pivot tables with compression stats.

Usage:
    uv run python eval_longbench.py                          # all co_v* runs
    uv run python eval_longbench.py --prefix co_v1_validate  # specific prefix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent / "longbench"))
from metrics import qa_f1_score, rouge_score

DATASET_METRIC = {
    "qasper": ("QA F1", qa_f1_score),
    "hotpotqa": ("QA F1", qa_f1_score),
    "gov_report": ("ROUGE-L", rouge_score),
    "narrativeqa": ("QA F1", qa_f1_score),
    "multifieldqa_en": ("QA F1", qa_f1_score),
}


def score_file(path: Path, dataset_name: str) -> tuple[float, int] | tuple[None, int]:
    metric_fn = DATASET_METRIC.get(dataset_name, ("QA F1", qa_f1_score))[1]
    preds: list[str] = []
    answers: list[list[str]] = []
    all_classes_per_sample: list[Any] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "pred" not in d:
                continue
            preds.append(d["pred"])
            answers.append(d["answers"])
            all_classes_per_sample.append(d.get("all_classes"))
    if not preds:
        return None, 0
    total = 0.0
    for pred, gts, all_classes in zip(preds, answers, all_classes_per_sample):
        s = 0.0
        for gt in gts:
            s = max(s, metric_fn(pred, gt, all_classes=all_classes))
        total += s
    return round(100 * total / len(preds), 2), len(preds)


def compression_stats(path: Path) -> dict[str, float] | None:
    ratios: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            inp = d.get("api_input_tokens", 0)
            out = d.get("used_context_tokens", 0)
            if inp > 0 and out > 0:
                ratios.append(out / inp)
    if not ratios:
        return None
    return {
        "mean": mean(ratios),
        "median": median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "n": len(ratios),
    }


def parse_run_name(name: str) -> dict[str, str] | None:
    parts = name.split("__")
    if len(parts) < 5:
        return None
    level_str = parts[4]
    if not level_str.startswith("L"):
        return None
    try:
        level_frac = float(level_str.replace("L", "").replace("_", "."))
        level_pct = int(level_frac * 100)
    except (ValueError, OverflowError):
        return None
    return {
        "provider": parts[1],
        "model": parts[2],
        "mode": parts[3],
        "level_pct": str(level_pct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Run name prefix filter (e.g. co_v1_validate). Default: all co_ runs.",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    args = parser.parse_args()
    pred_root = Path(args.pred_dir)

    scores: dict[tuple[str, str, str], tuple[float, int]] = {}
    comp: dict[tuple[str, str], list[float]] = {}
    datasets_seen: set[str] = set()
    levels_seen: set[str] = set()
    models_seen: set[str] = set()

    for run_dir in sorted(pred_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        prefix = args.prefix or "co_"
        if not name.startswith(prefix):
            continue
        info = parse_run_name(name)
        if info is None:
            continue

        models_seen.add(info["model"])

        for ds_file in sorted(run_dir.glob("*.jsonl")):
            if "api_dump" in ds_file.name:
                continue
            ds_name = ds_file.stem
            score, n = score_file(ds_file, ds_name)
            if score is None or n == 0:
                continue
            key = (ds_name, info["mode"], info["level_pct"])
            scores[key] = (score, n)
            datasets_seen.add(ds_name)
            levels_seen.add(info["level_pct"])

        for dump_file in sorted(run_dir.glob("*.api_dump.jsonl")):
            if dump_file.stat().st_size == 0:
                continue
            ds_name = dump_file.name.replace(".api_dump.jsonl", "")
            stats = compression_stats(dump_file)
            if stats:
                ck = (ds_name, info["level_pct"])
                comp.setdefault(ck, [])
                with dump_file.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        inp = d.get("api_input_tokens", 0)
                        out = d.get("used_context_tokens", 0)
                        if inp > 0 and out > 0:
                            comp[ck].append(out / inp)

    if not scores:
        print("No matching runs found.")
        return

    levels_sorted = sorted(levels_seen, key=lambda x: int(x))
    model_label = ", ".join(sorted(models_seen)) if models_seen else "unknown"
    modes = [
        ("api_generic", "generic (no hint)"),
        ("api_biased",  "biased (with hint)"),
        ("random",      "random (baseline)"),
        ("full",        "full (no compression)"),
    ]

    for ds in sorted(datasets_seen):
        metric_name = DATASET_METRIC.get(ds, ("QA F1", None))[0]

        has_full = any((ds, m, "100") in scores for m, _ in modes)
        compressed_levels = [lv for lv in levels_sorted if lv != "100"]

        # Per-dataset sample count: max n across all modes/levels for this dataset.
        ds_sample_n = max(
            (v[1] for (d, _, _), v in scores.items() if d == ds),
            default=0,
        )

        config_w = max(len(ml) for _, ml in modes) + 2
        col_w = 8

        cols = [f"{lv}%" for lv in compressed_levels]
        if has_full:
            cols.append("100% (full)")
        header = f"| {'Config':<{config_w}} |" + "".join(
            f" {c:>{col_w}} |" for c in cols
        )
        sep = f"|{'-' * (config_w + 2)}|" + "".join(
            f"{'-' * (col_w + 1)}:|" for _ in cols
        )

        print(f"\n### {ds.upper()} — {metric_name}, {model_label}, n={ds_sample_n}\n")
        print(header)
        print(sep)

        for mode_key, mode_label in modes:
            cells = []
            for lvl in compressed_levels:
                key = (ds, mode_key, lvl)
                if key in scores:
                    cells.append(f"{scores[key][0]:>{col_w}}")
                else:
                    cells.append(f"{'—':>{col_w}}")
            if has_full:
                key = (ds, mode_key, "100")
                if key in scores:
                    cells.append(f"{scores[key][0]:>{col_w}}")
                else:
                    cells.append(f"{'—':>{col_w}}")
            row = f"| {mode_label:<{config_w}} |" + "".join(f" {c} |" for c in cells)
            print(row)

        comp_levels = [lv for lv in compressed_levels if (ds, lv) in comp]
        if comp_levels:
            tw = 8
            print(f"\nActual compression ratios ({ds}):\n")
            print(
                f"| {'Target':>{tw}} | {'Mean':>{tw}} | {'Median':>{tw}} "
                f"| {'Min':>{tw}} | {'Max':>{tw}} | {'N':>5} |"
            )
            print(
                f"|{'-' * (tw + 2)}|{'-' * (tw + 2)}|{'-' * (tw + 2)}"
                f"|{'-' * (tw + 2)}|{'-' * (tw + 2)}|{'-' * 7}|"
            )
            for lvl in comp_levels:
                ratios = comp[(ds, lvl)]
                if ratios:
                    print(
                        f"| {lvl + '%':>{tw}} | {mean(ratios):>{tw}.1%} "
                        f"| {median(ratios):>{tw}.1%} | {min(ratios):>{tw}.1%} "
                        f"| {max(ratios):>{tw}.1%} | {len(ratios):>5} |"
                    )

    print()


# Type alias used in score_file
from typing import Any  # noqa: E402

if __name__ == "__main__":
    main()
