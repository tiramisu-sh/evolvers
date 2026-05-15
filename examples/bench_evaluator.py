"""Benchmark vLLM judge-call throughput at various LLM.max_concurrency levels.

Sweeps `LLM.max_concurrency` across configured values, measuring wall-clock
throughput for a fixed pool of judge-style prompts dispatched via async +
internal semaphore. Snapshots vLLM /metrics before/after each run.

This directly probes the concurrency knob that lives on the LLM — the same
one users tune in production. After the async refactor (#25), there is no
separate "max_workers" knob on Evolvable; only this.

Usage:
    uv run --with lurkers python -u examples/bench_evaluator.py
    uv run --with lurkers python -u examples/bench_evaluator.py --num-items 512 --batch-sizes 32 64 128 256 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.request
from statistics import mean

from pydantic import BaseModel

import evolvers as ev
import lurkers

# ─────────────────────────────────────────── config ──

VLLM_BASE = "http://localhost:8001/v1"
METRICS_URL = "http://localhost:8001/metrics"
MODEL = os.environ.get("EVOLVERS_BENCH_MODEL", "deepkek")
HN_RSS = "https://news.ycombinator.com/rss"

DEFAULT_BATCH_SIZES = [16, 32, 64, 128, 256]
DEFAULT_N = 256
DEFAULT_REPEATS = 2

INTERESTING_METRICS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
]


class JudgeResponse(BaseModel):
    """Mirrors evolvers' internal _JudgeResponse — what _evaluate_judge expects."""

    reasoning: str
    score: float


# ─────────────────────────────────────────── data ──


def fetch_dataset(n_needed: int) -> list[str]:
    """Pull base inputs from HN RSS, then pad to n_needed by cycling with a
    cache-busting suffix so vLLM's prefix-cache doesn't skew results."""
    print(f"fetching base dataset from HN RSS...", flush=True)
    docs = lurkers.feed(HN_RSS, limit=30)
    base = [d.content for d in docs if d.content and len(d.content.strip()) > 100]
    if not base:
        # fallback if HN is unreachable; still gives unique enough inputs for a bench
        base = [
            "The article discusses recent developments in artificial intelligence and "
            "their impact on software engineering productivity. " * 8
            + f" Article {i}."
            for i in range(30)
        ]
    print(f"  got {len(base)} unique base items (avg {mean(len(b) for b in base):.0f} chars)", flush=True)

    items: list[str] = []
    for i in range(n_needed):
        items.append(f"{base[i % len(base)]}\n\n[bench_id={i}]")
    return items


def make_judge_prompt(text: str) -> str:
    """Judge-style prompt mirroring evolvers' internal _evaluate_judge format."""
    return (
        "You are a strict evaluator. Reason briefly, then score in [-1, 1].\n\n"
        f"INPUT:\n{text}\n\n"
        f"OUTPUT:\n{text[:200]}...\n\n"
        "QUESTION: Does the OUTPUT directly summarize the main points as a TLDR?"
    )


# ─────────────────────────────────────────── prom ──


def parse_prometheus(text: str) -> dict[str, float]:
    """Minimal prometheus text-format parser. Returns flat dict, labels embedded in key."""
    out: dict[str, float] = {}
    line_re = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([0-9.eE+\-]+|NaN|\+Inf|\-Inf)"
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


def fetch_metrics() -> dict[str, float]:
    """GET /metrics. Sums across all label combinations for each metric base name."""
    with urllib.request.urlopen(METRICS_URL, timeout=5) as resp:
        parsed = parse_prometheus(resp.read().decode())
    # collapse labeled variants into a single sum per base metric name
    collapsed: dict[str, float] = {}
    for key, val in parsed.items():
        base = key.split("{", 1)[0]
        collapsed[base] = collapsed.get(base, 0.0) + val
    return collapsed


# ─────────────────────────────────────────── bench ──


async def run_one_batch(
    prompts: list[str],
    max_concurrency: int,
    label: str,
) -> dict:
    """Fire len(prompts) judge calls; LLM's internal semaphore (max_concurrency) gates them.

    Returns dict with elapsed, throughput, failures. Live progress + ETA on stderr.
    """
    # Fresh LLM per batch_size — semaphore is sized at construction.
    llm = ev.LLM(
        model=MODEL,
        base_url=VLLM_BASE,
        api_key="dummy",
        max_concurrency=max_concurrency,
    )

    n = len(prompts)
    t0 = time.perf_counter()
    completed = 0
    failures = 0
    last_print = t0 - 1  # force first paint immediately

    # asyncio.as_completed yields futures in completion order — perfect for progress.
    coros = [llm(p, schema=JudgeResponse) for p in prompts]
    for fut in asyncio.as_completed(coros):
        try:
            await fut
        except Exception:
            failures += 1
        completed += 1
        now = time.perf_counter()
        if now - last_print > 0.25 or completed == n:
            elapsed = now - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (n - completed) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:.1f}s" if eta != float("inf") else "—"
            msg = (
                f"  {label}  {completed:>3}/{n}  "
                f"{rate:>5.1f} req/s  ETA {eta_str:>6}  "
                f"fails={failures}"
            )
            print(msg.ljust(78), end="\r", flush=True)
            last_print = now

    print()  # newline after final progress
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "throughput_req_s": n / elapsed if elapsed > 0 else 0,
        "failures": failures,
    }


# ─────────────────────────────────────────── main ──


async def main() -> None:
    global METRICS_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-items",
        type=int,
        default=DEFAULT_N,
        dest="n",
        help="items per (batch_size, run); each item runs 1 rubric → 1 LLM call (default: 256)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="LLM.max_concurrency values to sweep (default: 16 32 64 128 256)",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="runs per batch size (default: 2)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=VLLM_BASE)
    parser.add_argument("--metrics-url", default=METRICS_URL)
    parser.add_argument("--out", default=None, help="JSON output path (default: bench_<ts>.json)")
    args = parser.parse_args()

    METRICS_URL = args.metrics_url

    print("=== evolvers judge-call throughput benchmark (async) ===")
    print(f"  endpoint:    {args.base_url}")
    print(f"  metrics:     {args.metrics_url}")
    print(f"  model:       {args.model}")
    print(f"  batch sizes: {args.batch_sizes}  (= LLM.max_concurrency)")
    print(f"  N per run:   {args.n}")
    print(f"  repeats:     {args.repeats}")
    print()

    # Probe metrics endpoint
    metrics_ok = True
    try:
        _ = fetch_metrics()
    except Exception as e:
        metrics_ok = False
        print(f"!! warning: couldn't reach metrics endpoint ({e}); continuing without server-side stats.")
        print()

    # Fetch dataset once, reuse across all batch sizes / repeats
    dataset = fetch_dataset(args.n)
    prompts = [make_judge_prompt(t) for t in dataset]
    print()

    # ── warmup: one call to load connection / structured-output codepath ──
    print("warmup (1 call)...", flush=True)
    warmup_llm = ev.LLM(model=args.model, base_url=args.base_url, api_key="dummy", max_concurrency=1)
    try:
        _ = await warmup_llm(prompts[0], schema=JudgeResponse)
        print("  ok")
    except Exception as e:
        print(f"  failed: {e}")
        print("  bailing — server-side schema/JSON support may be misconfigured.")
        return
    print()

    runs: list[dict] = []
    for bs in args.batch_sizes:
        for r in range(args.repeats):
            label = f"[bs={bs:>3}  run={r+1}/{args.repeats}]"
            before = fetch_metrics() if metrics_ok else {}
            result = await run_one_batch(prompts, max_concurrency=bs, label=label)
            after = fetch_metrics() if metrics_ok else {}

            delta = {
                k: (after.get(k, 0.0) - before.get(k, 0.0)) for k in INTERESTING_METRICS
            }
            result["batch_size"] = bs
            result["run"] = r + 1
            result["metrics_delta"] = delta
            runs.append(result)

            ptoks = int(delta.get("vllm:prompt_tokens_total", 0))
            gtoks = int(delta.get("vllm:generation_tokens_total", 0))
            print(
                f"  {label}  done in {result['elapsed_s']:>5.2f}s  "
                f"({result['throughput_req_s']:>5.1f} req/s  "
                f"prompt {ptoks:>8,} toks  gen {gtoks:>7,} toks  fails {result['failures']})"
            )
        print()

    # ── summary table (mean over repeats) ──
    print()
    print("=== Summary (mean over repeats) ===")
    print(f"  {'bs':>4}  {'elapsed_s':>9}  {'req/s':>7}  {'prompt_toks':>12}  {'gen_toks':>11}  {'fails':>5}")
    for bs in args.batch_sizes:
        subset = [r for r in runs if r["batch_size"] == bs]
        m_elapsed = mean(r["elapsed_s"] for r in subset)
        m_throughput = mean(r["throughput_req_s"] for r in subset)
        m_prompt = mean(r["metrics_delta"].get("vllm:prompt_tokens_total", 0) for r in subset)
        m_gen = mean(r["metrics_delta"].get("vllm:generation_tokens_total", 0) for r in subset)
        total_fails = sum(r["failures"] for r in subset)
        print(
            f"  {bs:>4}  {m_elapsed:>9.2f}  {m_throughput:>7.2f}  "
            f"{int(m_prompt):>12,}  {int(m_gen):>11,}  {total_fails:>5}"
        )

    # ── JSON dump ──
    out_path = args.out or f"bench_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "endpoint": args.base_url,
                    "model": args.model,
                    "batch_sizes": args.batch_sizes,
                    "n_per_run": args.n,
                    "repeats": args.repeats,
                },
                "runs": runs,
            },
            f,
            indent=2,
        )
    print()
    print(f"  raw results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
