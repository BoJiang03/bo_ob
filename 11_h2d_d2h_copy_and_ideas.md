# nsys analysis of H2D/D2H copy throughput + copy-performance ideas (Part 3, tasks ④+⑤)

Part 3's "Use nsys to analyze the H2D/D2H copy throughput in LMCache" and "Any potential
ideas to improve the copy performance?". Two captures of the same scenario on two
different nodes: `srv_h2d_1` (2026-07-21, rtx-026 GPU 0, via `nsys.sh`) and `run1`
(2026-07-23, rtx-024 GPU 0, via the one-script harness `profiling/copy-nsys.sh` —
check / up / cap / report / down, the single-node sibling of task ③'s `p2p-nsys.sh`).
Stock lmcache 0.5.1 in both cases: it already carries NVTX annotations on the
server-side copy path, so no fork build is needed here. Raw data in
`profiling/run/{srv_h2d_1,copy-run1}.nsys-rep` (gitignored, local).

**Headline: the two directions invert depending on which throughput you ask for.
Per-op, store (D2H) is the fast one — 56 GB/s, PCIe-saturating — and load (H2D) the
slow one at 32 GB/s. Per wall-clock phase, it flips: a store phase achieves only
7.7–8.8 GB/s because the copy engine sits idle 84–86% of the time between per-chunk
submissions, while a load phase runs at 27–29 GB/s at 88% duty cycle. Store is
submit-bound, not copy-bound — its `batch_size=1` D2H path (vs retrieve's batched
H2D) is the single biggest copy-performance lever inside the node, second overall
only to task ③'s ~2× P2P transport amplification.**

## Scenario

`workload.sh` drives store/reset/reload cycles against the profiling fleet
(ports 5559/8002/8082): N distinct ~7k-token prompts are prefilled cold (KV stored
to LMCache = D2H), vLLM's own prefix cache is wiped via `/reset_prefix_cache`, then
the same prompts are re-sent (LMCache hit, KV loaded back = H2D).

| Knob | srv_h2d_1 (07-21) | run1 (07-23) |
|---|---|---|
| Node / GPU | rtx-026 GPU 0 | rtx-024 GPU 0 |
| Traffic | 16 prompts × 2 rounds | 16 prompts × 3 rounds |
| KV volume | 27.9 GB per direction | 48.2 GB per direction |
| Model / chunk | Qwen3-8B bf16, chunk 16 = 2.36 MB (147,456 B/token) | same |

The profiled process is the **MP server** (`NSYS_TARGET=server`): in MP mode vLLM
moves no KV bytes, and the first 07-21 attempt proved it — an nsys around vLLM saw
3.7 MB of H2D where the server saw 27.7 GB.

## Method: op-level vs wall-level throughput

`nsys stats --report cuda_gpu_mem_time_sum` (what `nsys.sh report` prints) answers
only one question: Σbytes ÷ Σper-op-device-time. That is the **op-level** number —
how fast the DMA engine moves a 2.36 MB chunk once the copy has been issued. It says
nothing about the gaps *between* copies, and the gaps turn out to be the whole story.

`copy-nsys.sh report` therefore computes a second number from the exported sqlite
(`CUPTI_ACTIVITY_KIND_MEMCPY`): memcpys less than 50 ms apart are clustered into
**bursts** (one burst = one store or load phase of one prompt batch), and each burst
gets a wall-clock throughput (bytes ÷ burst duration) and a duty cycle
(Σop-time ÷ burst duration). Wall throughput is what a request actually experiences;
duty cycle says whether the phase is copy-bound (→ raise bandwidth) or submit-bound
(→ fix the software between copies). The same report prints the stream layout and
the NVTX push/pop shares, so one command answers task ④ end to end.

## Results (both runs; run1 numbers, 07-21 in parentheses)

| direction | op-level | burst wall | duty cycle |
|---|---|---|---|
| D2H store | **56.0** (56.0) GB/s, 41 µs/chunk | 8.8 (7.7) GB/s | **16% (14%)** |
| H2D load | 32.7 (31.9) GB/s, 69 µs/chunk | 28.7 (27.1) GB/s | 88% (88%) |

Two different machines, 1.7× different KV volume, near-identical numbers in every
cell — this is structural behaviour of the copy path, not a machine artifact.

What the three report sections establish:

1. **Store is submit-bound.** A store burst moves ~1.04 GB in ~455 chunk-copies over
   118 ms of wall time, of which the copy engine is active for only 19 ms. The
   ~220 µs of dead time per chunk is the software between submissions: the per-store
   allocator call (`TensorMemoryAllocator.batched_allocate`, 5.5 ms avg, **22.4% of
   all NVTX range time** — 18.9% on 07-21), the per-chunk Python loop, and the
   per-store logging/`inspect.signature` chain doc 9 measured. The code pins this
   down exactly: store calls `transfer_kv_per_object_group(..., batch_size=1, ...)`
   with the comment "batch_size must stay 1 for store"
   (`lmcache_driven_transfer.py:1089-1096`), while retrieve passes
   `batch_size=cache_context.max_batch_size` (`:1272`) — and retrieve's 88% duty
   cycle shows what batching buys.
2. **Why store must (currently) be 1 — and why that's fixable.** Store's
   `memory_objs` may contain `None` holes (chunks the storage manager skipped or
   failed to allocate), which D2H must skip individually while copying the rest.
   The batching helper's None-handling is per-*batch*, not per-object: a single
   `None` inside a batch makes the whole batch `continue` (`:481-490`), which for
   `batch_size>1` would silently drop valid chunks. So the constraint is an
   implementation limit of the batch splitter, not a semantic property of store.
3. **Both directions share one stream.** All ~44k memcpys of run1 (22,628 D2H +
   21,260 H2D) run on stream 13. A background store flush and a latency-critical
   load can never overlap; they interleave on one queue.
4. **Load's 88% is not always available.** Most load bursts sit at 28–29 GB/s, but
   some degrade to 12–17 GB/s at 37–53% duty cycle — exactly the bursts that overlap
   prefill kernels of the uncached suffix. The H2D copy contends with compute for
   scheduling; 07-21 hypothesized this, run1's burst table shows it directly.
5. Byte ledger self-consistent: 48.2 GiB ÷ 48 prompts ≈ 1.04 GB/prompt ≈ 7,080
   tokens × 147,456 B — matching the ~1200-word random prompts.

## Task ⑤: ideas to improve copy performance, ranked by measured headroom

| # | idea | evidence | expected effect |
|---|---|---|---|
| 1 | Root-cause and eliminate the **~2× P2P transport amplification** | doc 10: wire carries 1.9× payload on avg, localized to NIXL/UCX | ~halves P2P load time |
| 2 | **Batch store's D2H submissions** like retrieve's H2D | 16% duty cycle; retrieve at 88% with the same chunks proves the ceiling | store phase 8.8 → ~40+ GB/s (up to ~5×) |
| 3 | **Split + prioritize streams**: dedicated high-priority load stream, separate store stream | all copies on one stream; load bursts drop to 12–17 GB/s under prefill contention | store/load overlap; protects TTFT-critical loads |
| 4 | **Pool / pre-allocate memory objects** | `batched_allocate` = 19–22% of range time, 3.8–5.5 ms per store op | removes the largest single non-copy cost |
| 5 | Micro items (doc 9): `functools.cache` on `check_interprocess_event_support` (per-op `inspect.signature`, `utils.py:49`), demote per-store INFO logging (~5%), hoist `torch.cuda.stream` ctx (~6%) | flamegraph + source | a few % each, one-liners |

Notes on the top items:

- **#1** is two orders of magnitude above the micro items and sits in the transfer
  layer below LMCache's accounting; next probes need no GPU (`UCX_LOG_LEVEL=debug`
  re-run, or reading the transfer_channel submit path). Until then P2P goodput is
  4–6.7 GB/s on a wire that demonstrably does 9.2.
- **#2** has a concrete shape: make the batch splitter None-aware — split
  `memory_objs` into contiguous non-None runs and batch within each run (or compact
  `None`s out and adjust the block-id mapping). Everything downstream already takes
  a `batch_size`. The 88%-duty-cycle retrieve path is the existence proof; even
  reaching load's *wall* rate would cut store-phase time ~3×. Store is async, so
  requests don't wait on it directly — but it occupies the shared stream, fills L1
  at 1/6 of the achievable rate, and stretches the window in which a
  reset-then-reload finds chunks not yet flushed.
- **#3** is the classic CUDA recipe (`cudaStreamCreateWithPriority`): once store is
  batched (#2), overlap becomes worth having; today the single queue makes #2 and #3
  compound.
- **#4** shows up in both captures at ~1/5 of all annotated range time. An
  arena/pool sized to the steady-state store rate turns a 5.5 ms allocate into a
  free-list pop.

## Conclusion

Task ④ resolves to: **per-op D2H 56 GB/s / H2D 32 GB/s, but per-phase the effective
numbers are 8.8 GB/s (store, 16% duty cycle) and 28.7 GB/s (load, 88%)** —
reproduced across two nodes. The copy hardware is not the bottleneck in either
direction; the submission software is, and only on the store side. Task ⑤'s answer
follows directly: fix the P2P transport amplification (network path), batch store's
per-chunk D2H submission (node path), then split/prioritize streams and pool the
allocator — each idea carrying its own measured number from tasks ①–④.
