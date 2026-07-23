# nsys analysis of P2P loading throughput (Part 3, task ③)

Part 3's "Use nsys to analyze the P2P loading throughput in LMCache". Done 2026-07-23
on a **real 2-node fleet** — rtx-024 GPU 0 (node A, owner) + rtx-026 GPU 1 (node B,
requester), RoCE via mlx5_0 — driven end-to-end by `profiling/p2p-nsys.sh`
(check / install / up / cap / report / down / restore, node B over ssh). The running
lmcache in both venvs was **v0.5.1 + cherry-picked NVTX annotation commit** `d5bdded7`
([fork PR #1](https://github.com/BoJiang03/lmcache_test/pull/1), task ②) — without it
every NVTX table comes back empty. Four captures (`run1`–`run4`); raw data in
`profiling/run/p2p-run*-{A,B}.nsys-rep` and `profiling/bench-results/p2p-nsys/run*/`
(gitignored, local).

**Headline: P2P loading runs at 5.5 → 9.2 GB/s on the wire (ramping to a plateau at
~3/4 of RoCE line rate), but the goodput a request actually sees is 4.0–6.7 GB/s
per load — because the transport layer moves ~2× the payload bytes on average, a
cache-state-independent amplification we localized (but did not root-cause) to the
NIXL/UCX layer. Control-plane cost is ~2 ms per load (~1%); the post-transfer
L1→GPU copy reproduces task ④'s 32 GB/s H2D ceiling exactly. Eliminating the
amplification would nearly halve P2P load time — the top idea for task ⑤.**

## Scenario

Same topology and case driver as the Part 2 cross-node demo (`8_p2p_2node_bench.md`),
instrumented: warm N distinct prefixes on A only, then read each ONCE on B — B's L1
misses, the coordinator points at A, and the transfer channel RDMA-reads A's L1.

| Knob | Value |
|---|---|
| Fleet | coordinator :9300 (024) + per-node "MP server :5558 + vLLM :8002", p2p ports 9402(A)/9401(B) |
| Model / KV | Qwen3-8B bf16 — 36 layers × 2(K,V) × 8 kv-heads × 128 dim × 2 B = **147,456 B/token**, chunk 16 (2.36 MB) |
| L1 | 8 GB pinned per node, LRU, watermark 0.8 |
| Transfer | NIXL over UCX, RoCE (mlx5_0), 100 Gb/s ≈ 12.5 GB/s line rate |
| Loads | run1/run2: 4 × 6,512 tok = **0.89 GiB payload each**; run3/run4: 8 × 5,012 tok = 0.69 GiB each |
| Capture | both MP servers under `nsys launch -t cuda,nvtx`, window opened only around the case (`cap`) |

The profiled process is the **MP server** on each node — in MP mode vLLM moves no KV
bytes, so an nsys on vLLM would show nothing but the connector's submit overhead.

## Method: one throughput, three signals

The defining problem of this task: **the RDMA transfer is structurally invisible to
nsys' CUDA tracing.** The bytes move NIC-to-NIC between pinned host buffers — no CUDA
API is called, not even the CPU touches them — so the GPU timeline is blank exactly
where the interesting 150–250 ms lives. (Same shape as py-spy's blindness to
GIL-released native code in doc 9: outside a tool's hook boundary there is only
negative space.) Three signals triangulate it:

1. **NVTX envelope** (the task-② annotations): on B, one load is bracketed by
   `P2PL2Adapter.submit_load_task` … last `P2PL2Adapter.query_load_result`. The
   envelope over-counts time (submit→start queueing, plus ~5 ms poll quantization,
   26–47 polls/load), so payload ÷ envelope is a **lower bound** on wire speed.
2. **NIC byte counters, sampled**: nsys' own answer (`--nic-metrics`) needs the
   NVIDIA OFED driver — absent here, and a host-wide install is off-limits — but the
   same mlx5 counters are user-readable in sysfs. `cap` samples
   `/sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data` (unit: 4-byte words)
   on B at 50 Hz; `report` clusters >0.5 GB/s samples into bursts = **direct wire
   bandwidth**, one burst per load.
3. **CUDA memops** (`cuda_gpu_mem_time_sum`): the only stage CUDA *can* see — the
   post-arrival L1→GPU copy on B, and D2H on whichever node stores.

Signals 1 and 2 agree burst-for-burst (every burst starts within ~10 ms of its
load's submit and ends at its unlock), and both live on the same wall clock via
`TARGET_INFO_SESSION_START_TIME.utcEpochNs` in the exported sqlite. Being
duration-based, none of this needs cross-node clock alignment — which is why a real
2-node run costs nothing in measurement quality over same-node.

## Results

Canonical run (`run2`, 4 × 0.89 GiB payload; NIC-direct numbers):

| load | wire bytes | wire window | wire rate | goodput (payload ÷ window) |
|---|---|---|---|---|
| 0 | 0.94 GiB | 176 ms | 5.7 GB/s | 5.0 GB/s |
| 1 | 0.94 GiB | 132 ms | 7.7 GB/s | **6.7 GB/s** |
| 2 | 1.33 GiB | 177 ms | 8.1 GB/s | 5.0 GB/s |
| 3 | 1.81 GiB | 220 ms | 8.9 GB/s | 4.0 GB/s |

Anatomy of one load, all three planes (times from the NVTX tables):

```
B: submit_lookup_and_lock_task (2.0 ms)  ─┐  control plane, MQ RPC ×3
A:   p2p_lookup_and_lock       (0.56 ms)  │  A-side handler cost per load:
B: query_lookup_and_lock_result(1.8 ms)  ─┤  lock 0.56 + query 0.48 + unlock 0.95
A:   p2p_query_lookup_results  (0.48 ms)  │  ≈ 2 ms total  (~1% of the load)
B: submit_load_task            (1.5 ms)  ─┘
      ── RDMA read A's L1 → B's L1: 125–233 ms @ 5.7–9.2 GB/s  (>95% of the load)
B: query_load_result           (×26–47 polls @ ~5 ms)
B: submit_unlock → A: p2p_unlock_objects (0.95 ms)
B: retrieve (L1→GPU H2D): 1,628 chunks = 3.84 GB in 121.6 ms = 31.6 GB/s
```

Cross-validations, all landed:
- wire rate ∈ experiment A's TTFT-inferred **5–10 GB/s** RoCE range (`8_p2p_2node_bench.md`);
- B's H2D 31.6 GB/s ≈ task ④'s measured **32 GB/s** ceiling; D2H 56 GB/s reproduced too;
- byte ledger exact: 1,628 H2D chunks = 4 × 407 = A's `total_object_count`, and
  clean bursts = payload × 1.06 (RoCE header overhead);
- the empty `nvtx_startend_sum` tables are themselves a result: the async-annotated
  `P2PBackend` (`v1/storage_backend/`) never executes on this stack — the MP topology
  runs `P2PL2Adapter` + transfer_channel (`v1/distributed/`), confirming the
  two-parallel-stacks reading of the source.

End-to-end: TTFT 278 ms (P2P) vs 340 ms (recompute) = **1.22×** at 6.5k-token
prefixes (run2 re-measured 253/333 = 1.32×). Both transfer time and prefill scale
linearly with prefix length, so the ratio moves slowly; at ~5k tokens and below the
two paths converge (run3 suggested P2P can lose outright there, but its cold arm was
contaminated — see below — so only the direction is citable, not the number).

## Open finding: ~2× transport-layer amplification

The wire consistently carries **more bytes than the payload**, growing load over
load. run4 — both L1s wiped, A at 0.69 usage (never crossing the 0.8 watermark),
8 × 0.69 GiB loads — is the controlled version:

| load | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| wire ÷ payload | 1.06 | 1.04 | 1.20 | 1.85 | 2.16 | **3.23** | 2.48 | 2.45 |

Whole-run average **1.93×**. The exclusion chain that localizes it:

| Hypothesis | Killed by |
|---|---|
| link-layer retransmission | HW counters clean on both HCAs (`rnr_nak`, `out_of_sequence`, `out_of_buffer` all 0) |
| A pushing (write-through) | zero wire traffic during A's `submit_store_task`s — P2P store is metadata-only |
| watermark eviction / TTL / cache state | run4: clean caches, no watermark crossing, amplification unchanged |
| LMCache requesting extra chunks | B's ledger exact: `Prefetch request completed: 313/313 retained keys (0 L1, 313 L2)` per load |
| a second concurrent stream | 50 Hz NIC profile is a single-stream plateau per burst |

What remains: the duplication happens **below LMCache's chunk accounting and above
the RoCE link layer — inside NIXL/UCX transfer execution**. Signature: the first 1–2
transfers of a fresh connection are clean (1.06× = header overhead), then
amplification grows in step with the rate ramp (5.1 → 9.1 GB/s) — consistent with
re-issued RDMA reads during UCX's connection/pipeline adaptation, not with any
LMCache-level bug. Not root-caused; next probes need no GPU: `UCX_LOG_LEVEL=debug`
re-run, or reading the transfer_channel submit path.

**Consequence:** the wire is capable of ~9 GB/s but requests see 4–6.7 GB/s. Removing
the amplification ≈ halving P2P load time — task ⑤'s headline idea, two orders of
magnitude above the micro-optimizations (uncached `inspect.signature`, per-store INFO
logging) doc 9 catalogued.

One unresolved anomaly for completeness: in run3 (dirty caches), B's log showed later
P-loads partially or fully pre-satisfied from L1 — and even the "cold" Q arm hit
`(313 L1, 0 L2)`, which nothing we know of explains (prefix-chained hashing forbids
cross-prefix sharing) and which invalidated run3's TTFT comparison. Reproducible only
with dirty caches; parked with the amplification investigation.

## Method notes

- `annotate` (sync) ranges land in `nvtx_pushpop_sum`; `start_range/end_range`
  (async, the PR's decorator fix) land in `nvtx_startend_sum` — query the wrong table
  and the data "doesn't exist".
- In the exported sqlite, `NVTX_EVENTS.textId → StringIds.value` has **no** `lmcache:`
  prefix (the domain is prepended at display time); match on the bare name.
- `nsys start` intermittently fails with "Configuring is not allowed in this state"
  right after another session-state change; one 5 s retry clears it (`cap` does this
  automatically). Trace set is fixed at `nsys launch`; `start` only opens the window.
- Port 9400 is a root docker-proxy on **both** nodes (loopback-bound; NIXL on the NIC
  IP coexisted in experiment A, but the defaults moved to 9402/9401 anyway).
- The envelope method quantizes at the `query_load_result` poll interval (~5 ms);
  with the NIC sampler in place it is only the fallback.

## Conclusion

The homework question — analyze P2P loading throughput — resolves to: **wire
5.5–9.2 GB/s (ramping, single stream, ~3/4 of line rate), goodput 4.0–6.7 GB/s per
load, control plane ~1%, destination-side H2D not the bottleneck (32 GB/s)**. The
loading path is network-bound as expected, but not by the link: by a ~2× byte
amplification inside the transfer layer that the three-signal method exposed and an
eight-load clean-cache control isolated. The instrument built for this task (NVTX
annotations from PR #1 + sysfs NIC sampling + sqlite alignment) is reusable as-is for
any future transfer-channel investigation.
