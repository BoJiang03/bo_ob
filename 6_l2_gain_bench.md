# L2 gain bench notes (fs_native)

Part 2's "try one L2 adapter and demonstrate the performance gain of L2", done manually
with the `l2_support/` harness (vLLM 0.25.1 + LMCache 0.5.1, MP mode, chunk-size 16,
fs_native L2). Raw outputs live in `l2_support/bench-results/` (tracked for this part,
unlike `setup/`'s). The scripted equivalent is `l2_support/l2-gain.sh` (populate → reset →
re-read); this run instead extends [4_qwen3_bench.md](4_qwen3_bench.md)'s tiered-sweep
method with a third cache tier, then adds a paired no-L2 control.

## Pre-experiment source finding — L2 is a write-through superset, not a victim cache

Where L2 data comes from, per `/data1/bo/LMCache` sources
(`lmcache/v1/distributed/storage_controllers/`):

- `store_controller.py:121` — the trigger is `on_l1_keys_write_finished`: **every L1 write
  completion** notifies the StoreController (eventfd), which submits those keys to the L2
  adapters. Trigger is "written to L1", not "evicted from L1".
- `store_policy.py:144` — `DefaultStorePolicy`: "store all keys to all adapters, **never
  delete from L1**". (A `skip_l1` policy exists for the opposite extreme: L1 as pure
  write buffer.)
- `eviction_controller.py:192` — L1 eviction has exactly one destination: `DISCARD`.
  There is no demote-to-L2 path; eviction can afford to drop data because the write-through
  already put it in L2.

So **L2 ⊇ L1** asymptotically (gaps only from async store lag and L2's own eviction).
Confirmed live below: `lmcache_mp_l2_usage_bytes` reached **21.3 GB** while L1 is capped at
15 GB, and the on-disk footprint hit ~5 GB right after the 5 GB tier — long before L1 was
full.

## Setup

*2026-07-22, rtx-024 GPU 7. Result dirs `20260722-234206/234243/234405-Qwen3-8B-long-doc-qa-{5,12,20}GB`
and `20260722-235141-...-20GB` (no-L2 control).*

| Knob | Value |
|---|---|
| GPU KV pool | `GPU_MEM_UTIL=0.28` → **74,176 tokens ≈ 10.2 GB** (vLLM startup log; 7281 tok/GB) |
| LMCache L1 | `L1_SIZE_GB=15` (pinned DRAM, LRU) |
| LMCache L2 | `{"type":"fs_native","base_path":"/data1/bo/l2cache-demo/fs_native","num_workers":4}` — NVMe, uncapped (max working set 20 GB never fills it) |
| Workload | `long-doc-qa` defaults: 10k-token docs, 2 queries/doc, 3 in-flight, 128 output tokens, seed 42 |
| Sweep | `KV_VOLUME={5,12,20} ./bench.sh`, ascending, no cache clear (superset doc sets) |

The tiers deliberately straddle both boundaries: 5 < pool (10.2) < 12 < L1 (15) < 20, so
each tier isolates one cache level. (12 rather than 10 keeps the middle tier off the pool
boundary.)

## Tiered sweep results

| KV_VOLUME | docs | requests | TTFT mean | p50 | p90 | expected server |
|---|---|---|---|---|---|---|
| 5 GB | 3 | 6 | 62 ms | **52 ms** | 108 | GPU prefix cache |
| 12 GB | 8 | 16 | 120 ms | **62 ms** | 346 | GPU, spill → L1 |
| 20 GB | 14 | 28 | 437 ms | **476 ms** | 569 | GPU + L1 + **L2** |

```
 5GB     0-110ms: 6/6                                  — all GPU hits
12GB     0-100ms: 13   300-490ms: 3                    — ~1GB over pool: occasional L1
20GB     0-200ms: 7    400-570ms: 20   1990ms: 1       — see breakdown below
```

### 20 GB tier: who actually served each request

The server log's prefetch-completion lines (`storage_manager.py:709`) carry an exact
per-request L1/L2 split, e.g. `625/625 retained keys (0 L1, 625 L2) in 39.7 ms`.
Measured-phase composition, cross-checked against `:8081/metrics`:

| source | requests | evidence |
|---|---|---|
| GPU prefix cache | 7 | TTFT < 200 ms |
| L1 only | 4 | `(62x L1, 0 L2)` lines |
| **L2 (pure or mixed)** | **17** | 12× `(0 L1, 625 L2)` + 5 mixed; metrics: 17 load requests, 9,211 chunks ≈ 20.2 GB |
| recompute | **0** | nothing between 600–1900 ms |

Two things worth noticing:

- **L2 carried the load, not L1.** A static estimate says L1 (15 GB ≈ 11 of 14 docs)
  should miss only ~3/14 ≈ 21% of requests. Actually 17 of 21 non-GPU requests went to
  L2: every L2→L1 backfill (1.37 GB/doc) evicts other docs from L1, misses beget loads
  beget evictions — LRU churn makes L1's steady-state hit rate far worse than the
  capacity ratio suggests.
- **The L2 band hides inside the L1 band.** A full-doc L2 load (625 chunks = 1.37 GB)
  took 40–56 ms server-side ≈ 25–34 GB/s — that is **page cache (DRAM), not NVMe**; the
  data had been write-through'd minutes earlier. Client TTFT is dominated by the same
  L1→GPU retrieve path as an L1 hit (~450 ms at chunk 16 = 625 chunks/doc), so L2 only
  adds ~+40 ms (+9%): the 400–499 ms cluster is mostly L1-sourced, 500–599 ms mostly
  L2-sourced. No separate band appears.

## The A/B: same 20 GB tier, L2 removed

Same pool (10.2 GB), same L1 (15 GB), same seed → identical request schedule, exact
per-request pairing. Both runs fully cold-started (server restart wipes L1).

| | no-L2 | **fs_native L2** | Δ |
|---|---|---|---|
| TTFT mean | 713 ms | **437 ms** | **−39%** |
| p50 | 588 ms | 476 ms | −19% |
| p90 | 1722 ms | **569 ms** | **−67%** |
| max | 3110 ms | 1990 ms* | |
| decode tok/s (mean) | 49.5 | **67.7** | +37% |
| wall time (28 reqs) | 31.1 s | 22.4 s | −28% |

Composition shift (server log): no-L2 = 4 GPU + 6 full-L1 + 3 partial-L1 + **~15
recomputes** (550–608 ms band); with-L2 = 7 GPU + 4 L1 + 17 L2 + **0 recomputes**.

The gain has two layers:

1. **Per-miss, L2 vs recompute is only ~1.2×** (+90–150 ms per paired request in the main
   band: recompute 550–590 ms vs L2 read 440–510 ms). Same lesson as the L1=40 follow-up
   in doc 4: Qwen3-8B prefills at ~14k tok/s, recompute is cheap per-request.
2. **The queueing collapse is the real win.** Without L2, ~15 serialized 10k-token
   prefills stack on the GPU (3 in-flight); the tail shows it:

   ```
   doc10_q1   3110 → 440 ms   (+2671)
   doc7_q1    1855 → 491 ms   (+1363)
   doc12_q1   1722 →  48 ms   (+1674)
   doc11_q0   1116 → 453 ms    (+663)
   ```

   With L2 the GPU does no prefill at all, so nothing queues (regular max 570 ms), decode
   stops being preempted (+37%), and the whole run finishes 28% sooner. **As with L1, the
   aggregate value of a cache tier is decongesting the GPU, not raw per-request speed.**

*Honest footnotes:* the one negative pair (`doc4_q1` 1135 → 1990 ms) is the with-L2 run's
own first-burst outlier (3 simultaneous t=0 submissions), not L2 being slow. And the L2
side ran on a warm page cache; a genuinely cold NVMe read (~7 GB/s → ~200 ms/doc) would
shrink layer-1's margin to roughly break-even — but layer 2 survives cold reads, since it
comes from prefill work disappearing, not read speed. (Can't `drop_caches` on a shared
box to measure it.)

## Takeaway

fs_native L2 turned every capacity miss into a load: the recompute band vanished, mean
TTFT −39%, p90 −67%, decode +37%. On a fast-prefill 8B the per-request margin is thin
(and page-cache-flattered); the durable mechanism is that L2 keeps prefill work off the
GPU. Design-wise, L2 is a **write-through superset of L1** (store-time copy, eviction
discards), so it needs no L1-overflow "demotion" machinery — and under churn it, not L1,
serves most of the over-capacity working set.

## See also

- [7_l2_adapter_micro.md](7_l2_adapter_micro.md) — `lmcache bench l2` microbench of fs / fs_native / resp; decomposes these e2e numbers
- [8_p2p_2node_bench.md](8_p2p_2node_bench.md) — the cross-node P2P arm of the same break-even framework
- [4_qwen3_bench.md](4_qwen3_bench.md) — the L1-only tiered-sweep method this extends; L1=40 control with the same two-layer reading
- [1_control_vs_data_plane.md](1_control_vs_data_plane.md) — lookup-starts-prefetch, retrieve-does-H2D: why the L2 load overlaps the client-visible path
- [2_kv-cache-shapes.md](2_kv-cache-shapes.md) — 144 KB/token / 7281 tok/GB derivation
- `l2_support/l2-gain.sh` header — the scripted arms (no-l2 / fs / fs_native) version of this comparison
