#!/usr/bin/env python3
"""p2p_case.py — show the benefit of P2P KV sharing across two nodes.

Warms a long shared prefix P on node A only, then on node B measures TTFT for:
  - P  (shared): B's local L1 misses -> P2P finds it on A -> NIXL RDMA-reads it from A's
                 L1 -> B skips prefill.
  - Q  (cold)  : never seen anywhere -> B recomputes the full prefill.
If P2P works, TTFT(P on B) << TTFT(Q on B): B served the shared prefix from A's memory
over the network instead of recomputing it.

UNTESTED end-to-end: requires nixl on BOTH nodes + a live coordinator/P2P fleet (see
p2p-demo.sh).  The measurement logic mirrors l2_scenario.py.
"""
import argparse
import json
import random
import time

import requests

WORDS = ("time year people way day man thing woman life child world school state family "
         "student group country problem hand part place case week company system program "
         "question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house").split()


def make_prefix(seed, n_tokens):
    rng = random.Random(seed)
    return f"[P2P {seed}] " + " ".join(rng.choice(WORDS) for _ in range(n_tokens)) + "\nSummary:"


def send(base, model, prompt, stream):
    return requests.post(f"{base}/v1/completions",
                         json={"model": model, "prompt": prompt, "max_tokens": 1,
                               "temperature": 0.0, "stream": stream},
                         stream=stream, timeout=600)


def ttft_ms(base, model, prompt):
    t0 = time.perf_counter()
    r = send(base, model, prompt, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        if line and line.startswith(b"data: "):
            d = line[6:]
            if d == b"[DONE]":
                break
            ch = json.loads(d).get("choices", [{}])[0]
            if ch.get("text", "") != "" or ch.get("finish_reason"):
                r.close()
                return (time.perf_counter() - t0) * 1000.0
    r.close()
    return (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-a", required=True)
    ap.add_argument("--node-b", required=True)
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--prefix-tokens", type=int, default=6500)
    ap.add_argument("--seed-base", type=int, default=4242,
                    help="base RNG seed; bump it to get prefixes no node has cached yet")
    ap.add_argument("--repeats", type=int, default=4, help="number of DISTINCT prefix pairs (each read once)")
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # IMPORTANT: every prefix is measured EXACTLY ONCE on B. Re-reading the same prefix
    # would be served by B's own L1/GPU cache (not P2P), which silently destroys the
    # comparison -- measured: cold repeats went 1306ms -> 72ms -> 63ms. So use N DISTINCT
    # prefix pairs instead of repeating one.
    n = args.repeats
    shared = [make_prefix(args.seed_base + i, args.prefix_tokens) for i in range(n)]        # P: warmed on A
    cold = [make_prefix(args.seed_base + 50000 + i, args.prefix_tokens) for i in range(n)]  # Q: never warmed

    print(f"warm {n} shared prefixes on node A ({args.node_a}) ...", flush=True)
    toks = 0
    for sp in shared:
        r = send(args.node_a, args.model, sp, stream=False); r.raise_for_status()
        toks += r.json().get("usage", {}).get("prompt_tokens", 0)
    print(f"  A warmed {n} prefixes, {toks} prompt tokens total", flush=True)
    print(f"settle {args.settle}s ...", flush=True)
    time.sleep(args.settle)

    # reset B's GPU prefix cache so nothing local masks the measurement
    try:
        requests.post(f"{args.node_b}/reset_prefix_cache", timeout=30)
    except requests.RequestException:
        pass

    print(f"measure on node B ({args.node_b}): {n} P (P2P from A) vs {n} Q (cold), each read ONCE", flush=True)
    p = [ttft_ms(args.node_b, args.model, sp) for sp in shared]
    q = [ttft_ms(args.node_b, args.model, cp) for cp in cold]
    res = {
        "prefix_tokens": args.prefix_tokens,
        "shared_P2P_ttft_ms": {"mean": round(sum(p) / len(p), 1), "min": round(min(p), 1), "all": [round(x, 1) for x in p]},
        "cold_recompute_ttft_ms": {"mean": round(sum(q) / len(q), 1), "min": round(min(q), 1), "all": [round(x, 1) for x in q]},
    }
    speedup = res["cold_recompute_ttft_ms"]["mean"] / res["shared_P2P_ttft_ms"]["mean"]
    res["p2p_speedup"] = round(speedup, 2)
    print(f"  P (P2P)  mean TTFT = {res['shared_P2P_ttft_ms']['mean']} ms", flush=True)
    print(f"  Q (cold) mean TTFT = {res['cold_recompute_ttft_ms']['mean']} ms", flush=True)
    print(f"  => P2P speedup = {res['p2p_speedup']}x  (>1 means B was served from A's L1 over RDMA)", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
