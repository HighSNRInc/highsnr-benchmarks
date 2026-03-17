#!/usr/bin/env python3
"""
Generate LongBench v1 prediction JSONL files using the HighSNR Context Optimizer API.

This script is intentionally self-contained and does NOT implement metrics itself.
Instead, it writes JSONL in the shape expected by eval_longbench.py (and the official
LongBench v1 eval):
  - pred: str
  - answers: list[str]
  - all_classes: list[str] | None
  - length: int

Secrets:
  - Reads API keys from environment variables only.
  - Does not log or persist raw contexts beyond the LLM prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

import httpx
import tiktoken


Mode = Literal["full", "trunc", "api_generic", "api_biased"]

Provider = Literal["openai", "anthropic"]


_DEFAULT_INPUT_MAX_CHARS = 200_000
_MAX_RETRIES = 5
_RETRY_BASE_S = 2.0

# SHA-256 of the LongBench v1 data.zip from zai-org/LongBench on Hugging Face.
# Update this if the upstream file changes.
_LONGBENCH_ZIP_SHA256 = (
    "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
)

_T = TypeVar("_T")
_ENC: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoding (avoids a network download at import time)."""
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _token_len(text: str) -> int:
    return len(_enc().encode(text))


def _truncate_middle_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate by preserving head+tail (Lost-in-the-Middle style)."""
    max_tokens = max(1, max_tokens)
    enc = _enc()
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    return cast(str, enc.decode(toks[:head] + toks[-tail:]))


def _env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class LlmConfig:
    provider: Provider
    model: str
    max_tokens: int


def _retry_on_transient(fn: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Retry with exponential backoff on 429 / 5xx / connection errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                wait = _RETRY_BASE_S * (2**attempt)
                print(
                    f"  [retry] {status} — waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            wait = _RETRY_BASE_S * (2**attempt)
            print(
                f"  [retry] {type(exc).__name__} — waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(wait)
            continue
    return fn(*args, **kwargs)


def _call_openai(
    *, api_key: str, model: str, prompt: str, max_tokens: int
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    def _do() -> str:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

    return _retry_on_transient(_do)


def _call_anthropic(
    *, api_key: str, model: str, prompt: str, max_tokens: int
) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }

    def _do() -> str:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        return "".join(parts).strip()

    return _retry_on_transient(_do)


def _call_llm(cfg: LlmConfig, prompt: str) -> str:
    if cfg.provider == "openai":
        return _call_openai(
            api_key=_env("OPENAI_API_KEY"),
            model=cfg.model,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
        )
    return _call_anthropic(
        api_key=_env("ANTHROPIC_API_KEY"),
        model=cfg.model,
        prompt=prompt,
        max_tokens=cfg.max_tokens,
    )


def _compress_api(
    *,
    api_url: str,
    api_key: str,
    context: str,
    max_output_tokens: int,
    context_hint: str | None,
) -> tuple[str, list[str]]:
    """
    Call the Context Optimizer v1 API and return (optimized_text, selected_chunks).

    selected_chunks is the list of text chunks the optimizer selected, in order.
    The optimized_text is the chunks joined with double newlines.
    """
    payload: dict[str, Any] = {
        "document": context,
        "max_output_tokens": max(1, int(max_output_tokens)),
        "include_boundaries": True,
    }
    if context_hint is not None:
        payload["context_hint"] = context_hint

    key = api_key.strip()
    if key.lower().startswith("bearer "):
        key = key.split(" ", 1)[1].strip()

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "co-longbench-bench/0.1",
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(api_url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            try:
                j = exc.response.json()
                if isinstance(j, dict):
                    body = json.dumps(j, ensure_ascii=False)
            except Exception:
                pass
            raise httpx.HTTPStatusError(
                f"{exc} (response_body={body[:1000]!r})",
                request=exc.request,
                response=exc.response,
            ) from exc
        data = r.json()
    chunks: list[str] = data.get("selected_chunks", [])
    if not isinstance(chunks, list):
        chunks = []
    optimized_text = "\n\n".join(chunks)
    return optimized_text, chunks


def _load_prompt_configs(config_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    prompt_path = config_dir / "dataset2prompt.json"
    maxlen_path = config_dir / "dataset2maxlen.json"
    if not prompt_path.exists():
        raise SystemExit(f"Missing {prompt_path}")
    if not maxlen_path.exists():
        raise SystemExit(f"Missing {maxlen_path}")
    with prompt_path.open(encoding="utf-8") as f:
        prompts = json.load(f)
    with maxlen_path.open(encoding="utf-8") as f:
        maxlens = json.load(f)
    return prompts, maxlens


def _build_prompt(
    *, context: str, task_input: str, dataset_name: str, dataset2prompt: dict[str, str]
) -> str:
    template = dataset2prompt.get(dataset_name)
    if template is None:
        raise SystemExit(
            f"No prompt template for dataset {dataset_name!r}. "
            f"Available: {sorted(dataset2prompt)}"
        )
    return template.format(context=context, input=task_input)


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        _id = obj.get("_id")
        if isinstance(_id, str):
            done.add(_id)
    return done


def _download_file(*, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(timeout=300.0, follow_redirects=True) as client,
        client.stream("GET", url) as r,
    ):
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"Checksum mismatch for {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Delete the file and re-run to download a fresh copy."
        )


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip, rejecting any member whose resolved path escapes dest."""
    dest = dest.resolve()
    for member in zf.namelist():
        member_path = (dest / member).resolve()
        if not str(member_path).startswith(str(dest) + os.sep) and member_path != dest:
            raise SystemExit(
                f"Zip path traversal detected: {member!r} would escape {dest}"
            )
    zf.extractall(dest)


def _ensure_longbench_v1_extracted(cache_dir: Path) -> Path:
    """
    Download + extract LongBench v1 data.zip into cache_dir.

    We avoid `datasets.load_dataset(".../LongBench")` because newer `datasets` versions
    forbid dataset loading scripts (LongBench.py). The dataset repo provides `data.zip`
    with standard JSON/JSONL files; we load those directly.
    """
    zip_path = cache_dir / "data.zip"
    extracted_dir = cache_dir / "extracted"

    if not zip_path.exists():
        url = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
        print(f"Downloading LongBench v1 data to {zip_path} ...")
        _download_file(url=url, dest=zip_path)

    print("Verifying data.zip checksum ...")
    _verify_sha256(zip_path, _LONGBENCH_ZIP_SHA256)

    if not extracted_dir.exists():
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extractall(zf, extracted_dir)

    return extracted_dir


def _load_longbench_v1(dataset_name: str, cache_dir: Path) -> list[dict[str, Any]]:
    extracted_dir = _ensure_longbench_v1_extracted(cache_dir)

    candidates: list[Path] = []
    for ext in ("*.json", "*.jsonl"):
        candidates.extend(extracted_dir.rglob(ext))
    dataset_lower = dataset_name.lower()
    scored: list[tuple[int, Path]] = []
    for p in candidates:
        name = p.name.lower()
        if dataset_lower in name:
            score = 10
            if name == f"{dataset_lower}.json" or name == f"{dataset_lower}.jsonl":
                score = 100
            scored.append((score, p))
    scored.sort(reverse=True, key=lambda t: t[0])
    if not scored:
        raise SystemExit(
            f"Could not find dataset file for {dataset_name!r} under {extracted_dir}"
        )

    path = scored[0][1]
    text = path.read_text(encoding="utf-8")
    text_strip = text.lstrip()
    if text_strip.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(f"Unexpected JSON format in {path}")
        return [dict(x) for x in data]

    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["qasper", "hotpotqa"],
        help="LongBench v1 dataset names.",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[1.0, 0.8, 0.7, 0.6, 0.5],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "api_generic", "api_biased"],
        choices=["full", "trunc", "api_generic", "api_biased"],
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai"],
        choices=["openai", "anthropic"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory (JSONL grouped by run_name/dataset).",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="co",
        help="Prefix for each run_name folder.",
    )
    parser.add_argument("--max-hint-chars", type=int, default=2000)
    parser.add_argument(
        "--input-max-chars",
        type=int,
        default=int(os.getenv("CO_INPUT_MAX_CHARS", str(_DEFAULT_INPUT_MAX_CHARS))),
        help=(
            "Skip samples whose raw context exceeds this many characters (service would 413). "
            "Default matches the HighSNR service limit; override via CO_INPUT_MAX_CHARS."
        ),
    )
    parser.add_argument(
        "--dump-api-output",
        action="store_true",
        help="Write per-sample API debug JSONL alongside predictions.",
    )
    args = parser.parse_args()

    co_url = os.getenv("CO_API_URL", "https://api.high-snr.com/v1/optimize")
    co_key = _env("CO_API_KEY")

    data_cache_dir = Path(
        os.getenv(
            "LONGBENCH_V1_CACHE_DIR",
            str(Path(__file__).resolve().parent / "data_cache"),
        )
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    longbench_config_dir = Path(__file__).resolve().parent / "longbench" / "config"
    dataset2prompt, dataset2maxlen = _load_prompt_configs(longbench_config_dir)

    rng = random.Random(args.seed)

    llms: list[LlmConfig] = []
    if "openai" in args.providers:
        llms.append(
            LlmConfig(
                provider="openai",
                model=_env("OPENAI_MODEL"),
                max_tokens=128,
            )
        )
    if "anthropic" in args.providers:
        llms.append(
            LlmConfig(
                provider="anthropic",
                model=_env("ANTHROPIC_MODEL"),
                max_tokens=128,
            )
        )

    for dataset_name in args.datasets:
        if dataset_name not in dataset2prompt:
            raise SystemExit(
                f"No prompt template for dataset {dataset_name!r}. "
                f"Available: {sorted(dataset2prompt)}"
            )
        ds_max_gen = dataset2maxlen.get(dataset_name, 128)
        print(f"[{dataset_name}] prompt template loaded, max_gen_tokens={ds_max_gen}")
        ds = _load_longbench_v1(dataset_name, data_cache_dir)
        n = min(args.samples, len(ds))
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:n]
        subset = [ds[i] for i in indices]

        for llm in llms:
            for mode in args.modes:
                for level in args.levels:
                    run_name = (
                        f"{args.run_name_prefix}__{llm.provider}__{llm.model}"
                        f"__{mode}__L{str(level).replace('.', '_')}"
                    )
                    out_dir = out_root / run_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{dataset_name}.jsonl"
                    api_dump_path = out_dir / f"{dataset_name}.api_dump.jsonl"

                    done_ids = _load_done_ids(out_path)

                    llm_cfg = LlmConfig(
                        provider=llm.provider,
                        model=llm.model,
                        max_tokens=dataset2maxlen.get(dataset_name, 128),
                    )

                    t0 = time.monotonic()
                    processed = 0
                    skipped = 0
                    skipped_too_large_chars = 0
                    skipped_413 = 0
                    total_considered = 0

                    with out_path.open("a", encoding="utf-8") as f:
                        api_f = None
                        if args.dump_api_output:
                            api_f = api_dump_path.open("a", encoding="utf-8")
                        for item in subset:
                            _id = str(item["_id"])
                            if _id in done_ids:
                                skipped += 1
                                continue

                            task_input = str(item["input"])
                            context = str(item["context"])
                            answers = list(item["answers"])
                            all_classes = item.get("all_classes")
                            length = int(item.get("length", 0))

                            if len(context) > args.input_max_chars:
                                skipped_too_large_chars += 1
                                continue

                            original_tokens = _token_len(context)
                            budget = max(1, int(original_tokens * float(level)))
                            total_considered += 1
                            selected_chunks: list[str] = []
                            api_called = False
                            hint: str | None = None

                            if mode == "full":
                                used_context = context
                            elif mode == "trunc":
                                used_context = _truncate_middle_to_tokens(
                                    context, budget
                                )
                            else:
                                if mode == "api_biased":
                                    raw_hint = task_input.strip()
                                    raw_hint = raw_hint[: args.max_hint_chars]
                                    hint = raw_hint if raw_hint.strip() else None
                                try:
                                    api_called = True
                                    used_context, selected_chunks = (
                                        _retry_on_transient(
                                            _compress_api,
                                            api_url=co_url,
                                            api_key=co_key,
                                            context=context,
                                            max_output_tokens=budget,
                                            context_hint=hint,
                                        )
                                    )
                                except httpx.HTTPStatusError as exc:
                                    if exc.response.status_code == 413:
                                        skipped_413 += 1
                                        continue
                                    raise

                            prompt = _build_prompt(
                                context=used_context,
                                task_input=task_input,
                                dataset_name=dataset_name,
                                dataset2prompt=dataset2prompt,
                            )
                            pred = _call_llm(llm_cfg, prompt)
                            used_context_tokens = _token_len(used_context)
                            prompt_tokens = _token_len(prompt)

                            record = {
                                "_id": _id,
                                "dataset": dataset_name,
                                "mode": mode,
                                "level": level,
                                "pred": pred,
                                "answers": answers,
                                "all_classes": all_classes,
                                "length": length,
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()
                            if api_f is not None:
                                api_dump_record: dict[str, Any] = {
                                    "_id": _id,
                                    "dataset": dataset_name,
                                    "mode": mode,
                                    "level": level,
                                    "provider": llm.provider,
                                    "llm_model": llm.model,
                                    "api_called": api_called,
                                    "api_version": "v1",
                                    "api_input_tokens": original_tokens,
                                    "budget_tokens": budget,
                                    "used_context_tokens": used_context_tokens,
                                    "prompt_tokens": prompt_tokens,
                                    "context_hint": hint,
                                    "selected_chunks": selected_chunks,
                                }
                                api_f.write(
                                    json.dumps(api_dump_record, ensure_ascii=False)
                                    + "\n"
                                )
                                api_f.flush()
                            processed += 1
                        if api_f is not None:
                            api_f.close()

                    elapsed_s = round(time.monotonic() - t0, 2)
                    print(
                        f"[{dataset_name}] {run_name}: wrote={processed}, skipped_done={skipped}, "
                        f"skipped_too_large_chars={skipped_too_large_chars}, skipped_413={skipped_413}, "
                        f"total={total_considered} in {elapsed_s}s -> {out_path}"
                    )


if __name__ == "__main__":
    main()
