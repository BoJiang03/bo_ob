#!/usr/bin/env python3
"""bench-compare.py — quantify LMCache's gain: run the same workload twice on the same
model — (1) vLLM + LMCache, (2) plain vLLM baseline — and print a side-by-side
comparison of the key metrics.

Python port of bench-compare.sh (pilot) — behavior-identical; the .sh stays until verified.
Drives bench.py / lmcache-ctl.py / the .py model scripts.

Fairness notes:
  - the baseline still has vLLM's own GPU prefix cache, so the comparison isolates
    what LMCache *adds* (CPU-tier capacity beyond the GPU KV pool). Pick a KV_VOLUME
    larger than the vLLM GPU KV pool for the difference to show.
  - the lmcache server is restarted before run (1) so no cache from earlier
    experiments leaks in, and stopped for run (2) so its GPU buffer is freed.
  - tokens-per-GB for the baseline is taken from run (1)'s summary, so both runs
    derive identical document sets (same seed, same working-set size).

Usage:
  MODEL=Qwen3.6-35B-A3B KV_VOLUME=40 ./bench-compare.py [extra bench args...]
Env config: everything bench.py takes, plus KEEP_UP=1 to skip the final teardown.
"""

import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

MODEL = os.environ.get("MODEL", "Qwen3-8B")
WORKLOAD = os.environ.get("WORKLOAD", "long-doc-qa")
KV_VOLUME = os.environ.get("KV_VOLUME", "10")
OUT_ROOT = os.environ.get("OUT_ROOT", str(c.COMMON_DIR / "bench-results"))
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CMP_DIR = f"{OUT_ROOT}/{ts}-{MODEL}-{WORKLOAD}-{KV_VOLUME}GB-compare"

MODEL_SCRIPT = str(c.COMMON_DIR / f"{MODEL}.py")
LMCACHE_CTL = str(c.COMMON_DIR / "lmcache-ctl.py")
BENCH = str(c.COMMON_DIR / "bench.py")
EXTRA = sys.argv[1:]


def bench_env(**overrides):
    env = os.environ.copy()
    env.update({"MODEL": MODEL, "WORKLOAD": WORKLOAD, "KV_VOLUME": KV_VOLUME})
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def compare_table(cmp_dir):
    w = json.load(open(os.path.join(cmp_dir, "with-lmcache", "bench_summary.json")))["results"]
    n = json.load(open(os.path.join(cmp_dir, "no-lmcache", "bench_summary.json")))["results"]
    rows = [
        ("mean TTFT (ms)",             "mean_ttft_ms",            "lower"),
        ("p50 TTFT (ms)",              "p50_ttft_ms",             "lower"),
        ("p90 TTFT (ms)",              "p90_ttft_ms",             "lower"),
        ("p99 TTFT (ms)",              "p99_ttft_ms",             "lower"),
        ("mean request latency (ms)",  "mean_request_latency_ms", "lower"),
        ("input throughput (tok/s)",   "input_throughput",        "higher"),
        ("mean decode speed (tok/s)",  "mean_decode_speed",       "higher"),
        ("elapsed time (s)",           "elapsed_time",            "lower"),
    ]
    print(f"\n{'metric':<28} {'with LMCache':>14} {'no LMCache':>14} {'gain':>9}")
    print("-" * 68)
    for label, key, better in rows:
        a, b = w[key], n[key]
        gain = (b / a) if better == "lower" else (a / b)   # x-times better with LMCache
        print(f"{label:<28} {a:>14.1f} {b:>14.1f} {gain:>8.2f}x")
    print(f"\nrequests ok: with={w['successful_requests']}/{w['total_requests']}"
          f"  no={n['successful_requests']}/{n['total_requests']}")


def main():
    print(f"=== bench-compare: {MODEL} / {WORKLOAD} / {KV_VOLUME}GB -> {CMP_DIR}")
    # clean slate: instance down, lmcache server down (run 1 restarts it with the right chunk)
    c.run_or_die([sys.executable, MODEL_SCRIPT, "stop"])
    c.run_or_die([sys.executable, LMCACHE_CTL, "stop"])

    print("=== [1/2] WITH LMCache")
    c.run_or_die([sys.executable, BENCH, *EXTRA],
                 env=bench_env(OUT_DIR=f"{CMP_DIR}/with-lmcache"))

    with open(f"{CMP_DIR}/with-lmcache/bench_summary.json") as f:
        tokens_per_gb = json.load(f)["config"]["tokens_per_gb_kvcache"]

    print(f"=== [2/2] WITHOUT LMCache (baseline, tokens-per-gb={tokens_per_gb})")
    c.run_or_die([sys.executable, MODEL_SCRIPT, "stop"])
    # baseline shouldn't keep the server's GPU buffer around
    c.run_or_die([sys.executable, LMCACHE_CTL, "stop"])
    c.run_or_die([sys.executable, BENCH, *EXTRA],
                 env=bench_env(OUT_DIR=f"{CMP_DIR}/no-lmcache",
                               NO_LMCACHE=1, TOKENS_PER_GB=tokens_per_gb))

    if os.environ.get("KEEP_UP", "0") != "1":
        c.run_or_die([sys.executable, MODEL_SCRIPT, "stop"])

    compare_table(CMP_DIR)
    print(f"=== results in {CMP_DIR}")


if __name__ == "__main__":
    main()
