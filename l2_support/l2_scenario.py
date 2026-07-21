#!/usr/bin/env python3
"""l2_scenario.py — decisive L2-benefit probe: populate -> reset GPU prefix cache -> measure.

The end-to-end `lmcache bench engine` run tied WITH-L2 vs L1-only because vLLM's OWN GPU
prefix cache masked the tiers.  This driver removes that confound the same way the 2-node
isolation experiment did: it resets vLLM's GPU prefix cache AFTER populating LMCache, so the
re-read must be served by LMCache's tiers — L1 (small) + L2 (disk) if present, or L1 only.

Phases (single vLLM, already running):
  1. populate : send every doc once (fills LMCache L1 -> spills to L2 when L1 overflows)
  2. settle   : wait so async L2 stores finish
  3. reset    : POST /reset_prefix_cache  (clears vLLM GPU prefix cache; LMCache tiers survive)
  4. measure  : re-send every doc, stream, record time-to-first-token (TTFT = prefill cost)

WITH L2 : evicted docs reload from disk  -> prefill skipped   -> low TTFT for all
L1-only : evicted docs are gone          -> full recompute    -> high TTFT for the overflow

Docs are deterministic (seeded) and unique, each ~doc_tokens words, so the working set =
num_docs * doc_tokens tokens can be sized to overflow L1 by a chosen margin.
"""
import argparse
import json
import random
import statistics
import sys
import time

import requests

WORDS = ("time year people way day man thing woman life child world school state family "
         "student group country problem hand part place case week company system program "
         "question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name "
         "president team minute idea body information back parent face others level office door "
         "health person art war history party result change morning reason research girl guy "
         "moment air teacher force education foot boy age policy process music market sense "
         "nation plan college interest death course someone experience behavior car front").split()


def make_doc(idx, doc_tokens, base_seed):
    rng = random.Random(base_seed + idx)
    body = " ".join(rng.choice(WORDS) for _ in range(doc_tokens))
    # the [DOC idx ...] tag guarantees a unique prefix per doc (distinct KV, no cross-doc reuse)
    return f"[DOC {idx} seed {base_seed + idx}] {body}\nSummary:"


def complete(base, model, prompt, stream):
    return requests.post(
        f"{base}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 1,
              "temperature": 0.0, "stream": stream},
        stream=stream, timeout=600,
    )


def measure_ttft(base, model, prompt):
    t0 = time.perf_counter()
    r = complete(base, model, prompt, stream=True)
    r.raise_for_status()
    ttft = None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        j = json.loads(data)
        ch = j.get("choices", [{}])[0]
        if ch.get("text", "") != "" or ch.get("finish_reason"):
            ttft = (time.perf_counter() - t0) * 1000.0
            break
    r.close()
    if ttft is None:
        ttft = (time.perf_counter() - t0) * 1000.0
    return ttft


def pctl(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--num-docs", type=int, default=64)
    ap.add_argument("--doc-tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--settle", type=float, default=6.0, help="seconds to let async L2 stores finish")
    ap.add_argument("--reset-url", default="http://localhost:8001/reset_prefix_cache")
    ap.add_argument("--no-reset", action="store_true", help="skip the GPU-prefix-cache reset")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    docs = [make_doc(i, args.doc_tokens, args.seed) for i in range(args.num_docs)]
    tag = f"[{args.label}] " if args.label else ""

    # phase 1: populate (also collect the real prompt-token count)
    print(f"{tag}populate: sending {args.num_docs} docs (~{args.doc_tokens} tok each)...", flush=True)
    total_prompt_tokens = 0
    t_pop = time.perf_counter()
    for i, d in enumerate(docs):
        r = complete(args.base_url, args.model, d, stream=False)
        r.raise_for_status()
        total_prompt_tokens += r.json().get("usage", {}).get("prompt_tokens", 0)
    pop_s = time.perf_counter() - t_pop
    print(f"{tag}populate done: {total_prompt_tokens} prompt tokens in {pop_s:.1f}s", flush=True)

    # phase 2: settle + reset GPU prefix cache
    if args.settle > 0:
        print(f"{tag}settle {args.settle}s (flush async L2 stores)...", flush=True)
        time.sleep(args.settle)
    if not args.no_reset:
        rr = requests.post(args.reset_url, timeout=30)
        print(f"{tag}reset_prefix_cache -> HTTP {rr.status_code}", flush=True)

    # phase 3: measure TTFT on the re-read (served by LMCache tiers now, GPU cache is cold)
    print(f"{tag}measure: TTFT over {args.num_docs} docs...", flush=True)
    ttfts = [measure_ttft(args.base_url, args.model, d) for d in docs]

    summary = {
        "label": args.label,
        "num_docs": args.num_docs,
        "doc_tokens": args.doc_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "reset": not args.no_reset,
        "populate_seconds": round(pop_s, 2),
        "ttft_ms": {
            "mean": round(statistics.mean(ttfts), 1),
            "median": round(statistics.median(ttfts), 1),
            "p50": round(pctl(ttfts, 0.50), 1),
            "p90": round(pctl(ttfts, 0.90), 1),
            "p99": round(pctl(ttfts, 0.99), 1),
            "min": round(min(ttfts), 1),
            "max": round(max(ttfts), 1),
        },
        "ttft_ms_per_doc": [round(x, 1) for x in ttfts],
    }
    print(f"{tag}TTFT ms: mean={summary['ttft_ms']['mean']} median={summary['ttft_ms']['median']} "
          f"p90={summary['ttft_ms']['p90']} max={summary['ttft_ms']['max']}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"{tag}-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
