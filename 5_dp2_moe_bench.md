# 2-node DP + EP MoE bench notes (Qwen3-30B-A3B)

*2026-07-22, second full run of the Part-1 item "Run a 2-node DP setup for a MoE model,
and test with LMCache". First run and its troubleshooting: `records/2026/07/21/1_*.md`
(gitignored, local). This time the whole stack was launched **by hand with bare
commands** — no `dp-2node.sh` — to see every moving part. Result dirs
`setup/bench-results/20260722-2301*/-2302*` (gitignored); durable numbers below.*

## Topology and setup

| | |
|---|---|
| Model | Qwen3-30B-A3B (MoE, 128 experts, top-8), bf16, 96 KB/token → 10,922 tok/GB |
| Ranks | rtx-024 GPU 7 (head, API :8100) + rtx-026 GPU 1 (`--headless`, rank 1) |
| Parallelism | `--data-parallel-size 2 --data-parallel-size-local 1 --enable-expert-parallel` |
| EP effect | 64/128 experts per rank (log-confirmed), weights 29.9 GB/rank instead of ~57 GB |
| LMCache | one MP server **per node** (CUDA IPC cannot cross machines), chunk 256, L1 20 GB each |
| Workload | `long-doc-qa`, `--kv-cache-volume 10` → 10 docs × 10k tokens × 2 queries = 20 requests |

Key architectural fact for everything below: **the KV cache is per-rank** (each rank has
its own GPU pool and its own node's L1), while the head's DP load balancer routes
requests **without looking at content**. A document warmed on one rank is a total miss
on the other — there is no cross-node path (that would be LMCache's P2P layer, not used
here).

## Run 1 — cold start

TTFT mean 646 / p50 **712** / p99 1529 ms (07-21 reference: p50 723 — reproduced).

```
   0-150ms:  6   measured query routed to the same rank warmup used (GPU hit)
 150-400ms:  0
 400ms+  : 14   routed to the other rank: full 10k-token prefill (~710 ms) + queueing
```

The bimodal shape *is* the content-blind router: after warmup each doc lives on exactly
one rank, and a measured query lands on that rank with p ≈ 0.5.

## Run 2 — `POST /reset_prefix_cache`, then the identical bench again

Reset broadcasts to both ranks and wipes both GPU prefix caches; at that moment the
only copies of the KV anywhere are the two nodes' L1s. Same seed → identical schedule,
so runs pair per-request.

TTFT mean 166 / p50 **107** / p99 718 ms. Decomposition (from per-request CSV + overlap
analysis):

```
20 requests = 14 GPU hits (re-warmed from L1 by run 2's own warmup phase)
            +  4 L1 retrieves (150–400 ms band)
            +  2 full prefills (doc7_q0 721 ms, doc8_q0 709 ms)
```

The isolation argument: run 2's warmup found empty GPUs, and the `Retrieved N tokens`
lines on both nodes' servers during that phase are the KV coming back from L1 — the only
possible source. p50 dropping 712 → 107 with zero recompute for 18/20 requests closes
the chain.

## Why almost no prefills in run 2 — and why not exactly zero

Not luck; two healing mechanisms plus a probability floor:

1. **A miss is a backfill.** Every run-1 cross-rank miss ended in a prefill *followed by
   a normal STORE* — the doc entered that rank's L1 too. 14 misses largely erased the
   initial one-sided-ness.
2. **Warmup absorbs the rest invisibly.** Run 2's warmup is unmeasured; if it routed a
   still-one-sided doc to the deprived rank, the full prefill happened outside the CSV
   (and stored a copy, healing it).
3. **The floor:** a doc stays one-sided only if warmup₁ + both run-1 queries + warmup₂
   all landed on the same rank (~1/8 per doc), and the measured query must then hit the
   deprived side. Expectation over 10 docs ≈ 1–2 requests; observed exactly 2. Overlap
   analysis rules out queueing for both (concurrent only with 60–70 ms requests), and
   their 709/721 ms match run 1's measured prefill cost; their sibling queries
   (doc7_q1 66 ms, doc8_q1 62 ms) hit the other side — routing luck, both faces.

**Takeaway:** with per-node caches behind a content-blind DP balancer, repeated traffic
converges to full bilateral cache coverage (misses self-heal), but a residual miss rate
set by routing probability is inherent — it cannot reach zero without a cross-node
sharing layer (LMCache P2P) or cache-aware routing.

## Ops note from the bare-command run

Launching `.venv/bin/vllm` by absolute path without the venv's `bin/` on `$PATH` kills
the worker during the profiling run: flashinfer JIT-compiles its sampling module and
spawns `ninja`, which lives in `.venv/bin/`. `FileNotFoundError: 'ninja'` → EngineCore
dies. This is exactly why `setup/_common.sh:19` exports
`PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"` — bare commands must replicate it. (Also
reconfirmed: rtx-026's port 8080 is permanently held by a system `xray` proxy →
`--http-port 18080` there, forever.)

## See also

- `setup/dp-2node.sh` header — topology rationale, role-based usage, all the traps
  (device-ids vs CUDA_VISIBLE_DEVICES, 4h rendezvous timeouts)
- [4_qwen3_bench.md](4_qwen3_bench.md) — single-node capacity findings (L1 vs GPU pool)
- [1_control_vs_data_plane.md](1_control_vs_data_plane.md) — why the L1 retrieve is one
  CUDA-IPC H2D copy, and why MP mode is machine-local
