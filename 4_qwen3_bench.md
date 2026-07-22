# Qwen3-8B bench notes

Running record of benchmarks against the `setup/` Qwen3-8B recipe (vLLM 0.25.1 + LMCache
0.5.1, MP mode, chunk-size 16, no L2). Raw outputs live in `setup/bench-results/`
(gitignored for this part); this doc keeps the durable numbers and what they mean.

## Finding 1 — L1 ≈ GPU KV pool gives (almost) no marginal capacity

*2026-07-22, rtx-024 GPU 7. Result dirs `20260722-2223*-Qwen3-8B-long-doc-qa-{15,30,60}GB`.*

### Setup

| Knob | Value |
|---|---|
| GPU KV pool | `GPU_MEM_UTIL=0.4` → **157,168 tokens ≈ 21.6 GB** (vLLM startup log; 7281 tok/GB at 144 KB/token) |
| LMCache L1 | `L1_SIZE_GB=20` (pinned DRAM, LRU), **no L2** |
| Workload | `long-doc-qa` defaults: 10k-token docs, 2 queries/doc, 3 in-flight, 128 output tokens, seed 42 |
| Sweep | `KV_VOLUME={15,30,60} ./bench.sh`, ascending, no cache clear in between |

Ascending order is deliberately safe: the doc generator is deterministic and a larger
volume's document set is a superset of a smaller one's, so leftovers from the previous
tier only speed up the (unmeasured) warmup and never pollute the measured phase.

### Results

| KV_VOLUME | docs | requests | TTFT p50 | p90 | p99 | decode tok/s (mean) |
|---|---|---|---|---|---|---|
| 15 GB | 10 | 20 | **61 ms** | 75 | 88 | 69.5 |
| 30 GB | 21 | 42 | **607 ms** | 1135 | 1543 | 56.9 |
| 60 GB | 43 | 86 | **913 ms** | 1442 | 1677 | 54.1 |

Per-request TTFT histograms (from `bench_results.csv`):

```
15GB      0-150ms: 20/20                          — single peak, all GPU prefix-cache hits
30GB      0-150ms:  8   150-400ms:  0   400ms+: 34 — bimodal, NO middle band
60GB      0-150ms:  4   150-400ms:  4   400ms+: 78 — almost all slow
```

### Reading

- **15 GB**: working set (~14.6 GB) fits the 21.6 GB GPU pool; every request hits vLLM's
  own prefix cache. LMCache never gets to act. 61 ms ≈ recompute last block + question.
- **30 GB — the finding.** The 150–400 ms band, where L1-retrieve hits would sit
  (~1.4 GB H2D per 10k-token doc at ~32 GB/s ≈ 45 ms copy + overhead), is **empty**.
  Requests are either GPU hits (<150 ms) or full prefills (~600–700 ms, plus queueing
  above that). L1 contributed essentially zero hits.

  Why: **L1 is inclusive** — LMCache stores every chunk that passes through, it is not a
  victim cache for GPU evictions. Both tiers run LRU over the same access stream, and at
  20 GB vs 21.6 GB they are nearly the same size, so they cache nearly the same set: by
  the time a doc has aged out of the GPU pool it has usually aged out of L1 too.
  Effective capacity of the two tiers is **max(GPU, L1) ≈ 21.6 GB, not the sum (~41 GB)**.
  A 30 GB working set overflows that, hence the misses.
- **60 GB**: far beyond both tiers, near-total miss; p50 ≈ one full 10k-token prefill.
- **Queueing, not caching, explains TTFT > ~700 ms**: with 3 in-flight requests, prefills
  serialize on the GPU, so a miss can also wait behind other misses (p99 1.5–1.7 s).
  Same contention shows up as decode speed sagging 69 → 54 tok/s.

### Takeaway

L1 only adds effective capacity where **L1 > GPU pool**. With the default
`L1_SIZE_GB=20` next to a ~21.6 GB pool, the with-LMCache arm cannot beat the baseline
on capacity at any working-set size — hit tiers are `≤ pool: both hit` and
`> pool: both miss`.

### Follow-up: same 30 GB tier with `L1_SIZE_GB=40` — L1 hits appear

*2026-07-22, result dir `20260722-223545-...-30GB`. Both processes restarted (server
restart wipes L1), so this run is fully cold-start; same workload, same seed → identical
request schedule, per-request pairing is exact.*

| L1 | TTFT mean | p50 | p90 | p99 | decode tok/s |
|---|---|---|---|---|---|
| 20 GB | 667 ms | 607 | 1135 | 1543 | 56.9 |
| **40 GB** | **302 ms** | **407** | **482** | **860** | **68.9** |

Fine-grained TTFT histogram of the 40 GB run:

```
   0-150ms: 17    (GPU prefix-cache hits)
 150-400ms:  4
 400-500ms: 18    (L1 retrieves — tight cluster)
 500-600ms:  2
 600-700ms:  0    (the full-prefill band: EMPTY now)
 700ms+  :   1    (residual queueing)
```

Paired per-request: every slow request of the L1=20 run (850–1630 ms) landed at
430–530 ms with L1=40 (e.g. `doc16_q0` 1632 → 444 ms). Reads:

- **The missing band filled in.** A 10k-token L1 retrieve costs ~440–500 ms TTFT; the
  600–700 ms recompute band emptied, and the queueing tail (16 requests > 800 ms → 1)
  collapsed with it. Confirms Finding 1's mechanism from the other side.
- **Per-request, L1 vs recompute is only ~1.4×** for this model (450 ms vs 650 ms):
  Qwen3-8B prefills at ~14k tok/s, so recompute is cheap to begin with. Most of the
  *aggregate* win (mean −55%, p99 −44%, decode 56.9 → 68.9) comes from decongesting the
  GPU — retrieves don't hog compute the way serialized prefills do.
- 440–500 ms is ~10× the raw H2D copy (~1.4 GB at ~32 GB/s ≈ 45 ms); the rest is
  lookup/prefetch-wait and per-chunk overhead — at chunk-size 16 a 10k-token doc is
  ~625 chunks (Part 2/B measured exactly this chunk-size effect).

## See also

- [5_dp2_moe_bench.md](5_dp2_moe_bench.md) — the 2-node DP+EP MoE counterpart of these notes
- [2_kv-cache-shapes.md](2_kv-cache-shapes.md) — where 144 KB/token / 7281 tok/GB comes from
- [1_control_vs_data_plane.md](1_control_vs_data_plane.md) — why an L1 hit costs one CUDA-IPC H2D copy
- `setup/bench.sh` header — KV_VOLUME semantics, NO_LMCACHE baseline arm
