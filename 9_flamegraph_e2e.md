# Flamegraph of the end-to-end serving path (Part 3, task ①)

Part 3's "Use flamegraph to profile the end to end — any obvious hotspot?". Done
2026-07-23 on rtx-026 GPU 7, manually, with the `profiling/` harness (`flame.sh` =
py-spy gil+wall under `sudo -n`; py-spy is the only CPU profiler this box permits —
perf and nsys CPU sampling are both dead on `perf_event_paranoid=4`, root or not).
SVGs in `profiling/flames/*_{pop30,mix30}.svg` (gitignored, local). The 07-21 first
pass only profiled the vLLM process; this run adds the **lmcache server** — the
process that actually moves KV in MP mode — plus a store-phase/mixed-phase split.

**Headline: there is no Python hotspot. The server holds the GIL 5.2% of wall time
while sustaining the full copy load, every one of its 12 threads waits event-driven
(no busy-poll), and vLLM's 13 threads are 99.3% wait — the only "working" thread
spends 92% of its wall time inside `cuda.synchronize`. The e2e cost lives in native
copy code and GPU prefill, exactly where the nsys measurements (doc: task ④) pick
up. The flamegraph's finding is the negative space — plus a list of the per-chunk
control-plane costs that would surface first if load or chunk count grew.**

## Scenario

Two-fates design: every request either hits vLLM's GPU prefix cache or loads from
LMCache L1 — no L2, no recompute (by intent; see the watermark surprise below).

| Knob | Value |
|---|---|
| GPU pool | `GPU_MEM_UTIL=0.38` → 143,456 tokens ≈ **19.7 GB** (vLLM log) |
| LMCache L1 | 40 GB pinned, chunk 16, no L2 adapters |
| Working set | `KV_VOLUME=30` → 21 docs × ~10k tokens |
| Traffic | `long-doc-qa`, `qpd=16` → 336 measured requests, in-flight 3 (`profiling/bench.sh`) |
| Captures | `flame.sh server pop30` (45 s, aligned to warmup = populate/D2H), `flame.sh server mix30`, `flame.sh vllm mix30` (30 s each, measured phase) |

Since pool (19.7) < volume (30) < L1 (40), the bench's own warmup is the populate
pass and its measured phase is the random re-query — no `reset_prefix_cache`
choreography needed (contrast `workload.sh`, which exists for phase-pure nsys
windows). TTFT confirms the two fates: **64% of requests in 0–99 ms (GPU hit — ≈
the 19.7/30 capacity ratio), 33% in 400–599 ms (L1 load)**, plus a small recompute
tail explained below. Ops lesson: start the capture *before* launching the bench —
warmup is ~1 min and chasing it fails (first attempt lost).

## Server process: 5.2% GIL, 95% sleep

*(graphs: `server_gil_pop30.svg` / `server_gil_mix30.svg` for the GIL numbers,
`server_wall_{pop30,mix30}.svg` for the thread columns)*

- **GIL held during only 5.2% of sample ticks, identically in both phases** (233/4455
  in populate, 154/2970 in mixed). The process was moving tens of GB of KV at the
  time; the copies are native and release the GIL, so py-spy structurally cannot see
  them — that blindness *is* the result: the Python layer adds ~0.05 cores of cost.
- **The wall graph is 12 equal columns, every leaf a wait primitive** — eviction /
  store / prefetch controller loops, EventBus, native-completion watcher, ZMQ poll,
  queue gets. No column bottoms out in loop code: the eventfd-driven design really
  sleeps (the busy-poll EventBus complaint that motivated PR #4088's author does not
  reproduce on 0.5.1). Full thread-by-thread census below.
- **Inside the 5.2 % slice**, the cost is per-chunk control-plane bookkeeping:

  | cost | populate | mixed | note |
  |---|---|---|---|
  | object-key machinery (`ipc_key_to_object_keys` + dataclass `__hash__/__eq__/__init__`) | ~15% | ~12% | generated dunders churn on every chunk key |
  | blake3 token hashing (`hash_tokens` path) | ~10% | ~11% | scales inversely with chunk size |
  | msgspec decode of MQ messages | 7% | 12% | scales with request rate |
  | L1 bookkeeping (`reserve_write` 6.4%, `finish_read`, LRU `on_keys_touched`) | ~10% | ~4% | `reserve_write` is the Python rim of the allocator cost nsys saw as 18.9% of store-range time |
  | EventBus `_run` | 8% | ~1% | healthy |

  None of these matter at 5.2% GIL; they are the watch-list for smaller chunks or
  10× load. Phase alignment sanity-check: the pop capture is store-family frames
  (`store`, `reserve_write`), the mix capture retrieve-family (`retrieve`,
  `reserve_read`).

### The wall graph (pop30): a 12-thread census

`server_wall_pop30.svg` covers 45 s of the store phase — 30 GB of KV crossing D2H
into L1 while it recorded. Total 53,436 samples ÷ 4,453 ticks = **exactly 12.0
threads, all full-attendance** (a long-lived server has no late-born pool workers,
unlike vLLM's executor). The whole SVG is 34 frames, 9 deep — the flattest of the
six graphs. Single process, single GIL: the root splits only into the main thread
(`<module>`, lmcache CLI → uvicorn) and 11 `_bootstrap` children.

| # | thread | leaf | samples | waits on (wheel source checked) |
|---|---|---|---|---|
| 1 | main | `asyncio/runners.py:118` | 4,453 | uvicorn epoll for the HTTP/metrics port (:8082 — the silent-collision port) |
| 2 | MQ `_main_loop` | `zmq poll.py:106` | 4,409 | `poller.poll(1000)` on ZMQ socket + outbound eventfd (`mq.py:600`) |
| 3 | store controller | `_store_loop:452` (own line) | 4,453 | `poller.poll(timeout)` over adapter eventfds |
| 4 | prefetch controller | `_prefetch_loop:650` (own line) | 4,452 | same shape; benched all run (no L2) |
| 5 | eviction task loop | `_eviction_loop:288` (own line) | 4,453 | `time.sleep(1)` |
| 6 | watermark patrol | `eviction_loop:150` (own line) | 4,452 | `time.sleep(1)` then usage check — the 0.8-watermark loop |
| 7 | EventBus `_run:308` | `threading.py:359` | 4,429 | timed `Event.wait` |
| 8 | native_completion `_run:107` | `threading.py:359` | 4,435 | `wake.wait(timeout=drain_interval)` — drains native-copy completions |
| 9–10 | periodic_thread ×2 | `threading.py:359` | 8,906 | double-width merged frame; `stop_event.wait(interval)` |
| 11 | stdlib executor worker | `:89` q.get 4,367 + `:92`→run **86 working** | 4,453 | one of the only two working slivers |
| 12 | affinity_pool worker | `:77`→queue→`wait:355` 4,304 + `:83` `fn(*args)` **142 working** | 4,446 | lmcache's pinned copy-worker pool — the other |

Readings that survive beyond this run:

- **The 228-tick work sliver cross-checks the GIL graph.** The only visible work in
  45 s is executor 86 + affinity_pool 142 ≈ 228 ticks (~2.3 s); the same phase's
  gil graph totals 233. Two independent counts agree: essentially all of the
  server's awake Python happens inside those two pool-thread windows — wall says
  *who* woke, gil says *what awake did* (key hashing, msgspec, `reserve_write`).
- **The copy itself is a 142-tick thin line.** Those samples pin at
  `affinity_pool.py:83` `result = fn(*args, **kwargs)` — the last Python frame
  before the native D2H call, which releases the GIL. Classic wall-mode blind spot:
  the graph proves the copy is native and cheap in Python terms; its actual
  throughput is nsys's job (task ④).
- **The KV request path maps onto 5 of the 12 threads**: MQ (#2) decodes the
  connector's ZMQ request → store controller (#3) schedules on eventfd → an
  affinity_pool worker (#12) runs the native copy → native_completion (#8) drains
  the finish event and acks → eviction pair (#5/#6) keeps L1 writable. Everything
  else is infrastructure (HTTP, EventBus broadcast, periodic housekeeping). Three
  wait styles, all with timeouts or eventfds — no busy-poll anywhere.
- **The watermark patrol (#6) did this capture's 5,825-chunk eviction yet shows
  zero visible work ticks** — an LRU discard swing is pointer-pops and accounting,
  finished between 99 Hz samples. The flamegraph certifies the evictor is cheap; it
  says nothing about its *consequence* (the recompute tail), which only the TTFT
  histogram and metrics exposed. Structure from the graph, causality from
  elsewhere.
- Small bookkeeping: the MQ thread's 44 missing ticks (4,409 vs 4,453) are its
  message-handling stacks, each individually below inferno's render cutoff — ≈1%
  handling time, consistent with the gil graph's msgspec share.

### The GIL graph (pop30): 233 ticks, split by thread

`server_gil_pop30.svg` is the wall graph's inverse: 306 frames, 19 deep — every
sample a real call stack. Splitting the 233 by which thread held the GIL:

| thread | samples | share | doing |
|---|---|---|---|
| affinity_pool worker | 87 | 37% | transfer's Python rim: `store` family 73 + `retrieve` family 13 |
| stdlib executor worker | 84 | 36% | **the entire Lookup module runs here**: `lookup` 32, `free_lookup_locks` 26, `end_session` 20 |
| MQ `_main_loop` | 27 | 12% | message handling; 16 of it `unwrap_request_payloads → msgspec_decode` |
| EventBus | 19 | 8% | `_drain_all` → metrics-subscriber callbacks |
| native_completion | 14 | 6% | `_drain_once` → completion callbacks |
| watermark patrol / main | 1 + 1 | — | the patrol's single awake tick in 4,455 (`get_memory_usage`) |

Takeaways:

- **73% of GIL time belongs to the two pool threads** — the same moments as the
  wall census's 228 working ticks, seen from the other side.
- **Cross-referencing wall and gil isolates the copy itself.** Executor: 86
  wall-working ≈ 84 gil — Lookup is pure Python, awake means holding the lock.
  Affinity worker: 142 wall-working vs 87 gil — the **55-tick gap is awake time
  with the GIL released, i.e. the native D2H copy body** (~0.55 s of copy wall
  time in 45 s). The flamegraph can prove the copy exists and is native; its
  throughput is structurally invisible here and belongs to nsys (task ④).
- **Phase contamination, quantified**: this 45 s capture overran the warmup
  window, so measured-phase frames leaked in — `retrieve` 13 vs `store` 73 on the
  transfer thread plus ~40–50 ticks of read-side bookkeeping (`reserve_read`,
  `finish_read`, `finish_read_prefetched`) ≈ 20% of the graph. The Lookup slice is
  *not* contamination (populate does lookup-miss before every store; it runs in
  both phases). The populate picture stands; the clean mix30 pair exists for
  contrast.
- **Honest information-content verdict**: for a system whose design goal is
  keeping the hot path out of Python, a low-load gil graph is *supposed* to be
  boring — no plateau, no lock contention, no surprise consumer is the passing
  grade. Its durable outputs are the negative-space proof, the ranked watch-list
  in the cost table above, and the nothing-is-wrong certificate; the physics
  (queueing, batching, throughput) needs the nsys timeline.

### The wall graph (mix30): same skeleton, smaller slivers

`server_wall_mix30.svg` (30 s, measured phase) is frame-for-frame isomorphic to
the pop30 census: 34 frames, 9 deep, the same 12 full-attendance threads
(35,616 ÷ 2,968 = 12.0), every leaf the same wait primitive. The only thing that
moves between phases is the size of the two working slivers:

| working sliver | pop30 (store) | mix30 (mixed) |
|---|---|---|
| executor / Lookup | 86/4,453 = 1.9% | 47/2,968 = 1.6% |
| affinity_pool / transfer | 142/4,453 = 3.2% | 86/2,968 = 2.9% |
| total visible awake Python | 5.1% | 4.5% (gil_mix30: 154/2,970 = 5.2% ✓) |

The mixed phase is slightly *quieter*, which checks out: only ~1/3 of measured
requests touch L1 at all (the rest are GPU prefix-cache hits that cost the server
one lookup), and the read path is thinner per byte than the write path — retrieve
issues one batched H2D (`batch_size=max`), store does per-chunk D2H
(`batch_size=1`). The prefetch controller idles through both phases (no L2). The
graph's entire increment over pop30 is one sentence: **a phase change does not
change the server's resting shape, it only scales the two working slivers.**

### The GIL graph (mix30): still write-flavored, plus one real crumb

`server_gil_mix30.svg`: 154 samples, 278 frames, 21 deep. Thread split: affinity
pool 67 (43%, `store` family 51 + `retrieve` family 16), executor/Lookup 47 (31%),
MQ 23 (15%, of which `msgspec_decode` 18), EventBus 10, native_completion 7; the
main thread and both eviction loops were never caught awake.

- **The "read" phase is still write-flavored in Python terms — store 51 vs
  retrieve 16.** Three compounding reasons: (1) the watermark's consequence shows
  up here — chunks evicted during populate get recomputed in the measured phase
  and *re-stored*; (2) every random query's suffix is new chunks, written through;
  (3) store submits per-chunk (`batch_size=1`) while retrieve issues one batched
  H2D, so store's per-byte Python rim is an order of magnitude thicker. Bytes are
  read-dominated, GIL ticks write-dominated; `gil_pop30` and `gil_mix30` are
  near-twins (73:13 vs 51:16) — "mix" is literal.
- **The wall×gil cross-check holds a second time**: executor 47 wall-working vs
  47 gil (exactly 1:1 — Lookup is pure Python, confirmed twice); affinity 86 wall
  vs 67 gil → 19-tick gap = the native copy body. As wall-time share the copy is
  pop 55/4,453 = 1.2% vs mix 19/2,968 = 0.64% — the measured phase pushes roughly
  half the bytes/s through the server, consistent with the 1/3 L1-hit rate plus
  batched H2D.
- **One actionable crumb — per-operation `inspect.signature`.**
  `check_interprocess_event_support()` (`lmcache/utils.py:49`) reflects over
  `torch.cuda.Event` and `Event.__new__` via `inspect.signature` on *every* call,
  uncached — and it is called per transfer operation from
  `lmcache_driven_transfer.py:971` (store) and `:1196` (retrieve). ≈3% of the GIL
  slice here; the answer never changes within a process, so one `functools.cache`
  removes it. Same genre, smaller: per-store INFO logging chain ~5%,
  `torch.cuda.stream` context-manager Python overhead ~6%. All noise at 5.2% GIL,
  but the signature check is the first concrete entry for task ⑤ ("ideas to
  improve copy performance") — zero-risk, micro-PR sized.

## vLLM process tree: 13 threads, 12 waiting, 1 pacing the GPU

*(graphs: `vllm_gil_mix30.svg` for this paragraph, `vllm_wall_mix30.svg` for the
census below)*

### The GIL graph: 393 awake moments, one plateau

First number to read: **393 samples against 2,970 ticks**. `vllm serve` is a process
*tree* (API server + spawned `VLLM::EngineCore` + a resource_tracker), each Python
process with its own GIL, so the graph is two subtrees with separate ceilings: API
server 202/2,970 = **6.8%**, EngineCore 191/2,970 = **6.4%**. The sparseness is not
under-sampling — at almost every tick, *no* thread in either process was executing
bytecode: 12 of 13 threads sit in GIL-releasing waits and the EngineCore main thread
spends 92% of its time in `cuda.synchronize` (also GIL-releasing). GIL utilization
read this way is a **single-core headroom gauge**: both processes are ~15× below the
one-core bytecode ceiling that no amount of threading can raise.

What the 393 awake moments contain:

- **API server — the stack's only true plateau: `_protected_step`
  (`detokenizer.py:225`), self 77 samples = 38% of that process's GIL.** Incremental
  detokenization is pure Python, runs once per output token per request, and is
  serialized by the API process's GIL. Here that is ~2.6% of a core — noise — but it
  scales with `concurrency × output length`, which makes it a *low-load sliver,
  high-load throughput ceiling*. The distance to that ceiling is the 15× GIL
  headroom, i.e. roughly in-flight ~45 at this output length (linear extrapolation
  from in-flight 3) — far for this bench, but inside the normal production operating
  range for an 8B model on a 96 GB card, and GIL contention degrades tail latency
  well before 100% utilization. Long-output workloads multiply it further; whether
  the API GIL or the GPU saturates first depends on the model/card ratio. This is
  why vLLM already runs the API in its own process (own GIL) and why detokenizer
  optimization keeps receiving upstream attention. The rest of the API
  slice is serving overhead: SSE streaming generators, response serialization,
  msgspec decode of EngineCore replies, the uvloop loop itself.
- **EngineCore — no plateau, only launch glue**: triton JIT kernel-launch machinery
  (`jit.py` cache-key/lambda/driver `__call__`, ~10% of its samples), sampling prep
  (`get_top_k_top_p`), small `async_tensor_h2d` staging, model-forward glue frames.
  The shape itself is the finding: model execution is CUDA; Python only fires
  kernels. **The LMCache connector's total footprint is 3 samples
  (`GetStoreMetadata`) ≈ 0.1% of a core** — searching `lmcache` highlights
  practically nothing.
- The gil and wall graphs capture the same threads in opposite states: the
  99%-asleep wall columns wake for sub-millisecond bursts that 99 Hz wall sampling
  almost never catches, but the gil graph collects exactly those bursts — e.g.
  `process_outputs_socket → serial_utils.decode` is the output thread's awake
  moment. Read wall for where lives are spent, gil for what awake looks like.

### The wall graph: a 13-thread census

The wall graph (`vllm_wall_mix30.svg`) is worth reading as a census. Wall mode gives
every live thread exactly `duration × rate` ≈ 2,969 samples ("full attendance"), so
subtree width ÷ 2,969 = thread count. Leaf-frame inventory (self ≥ 1,500 samples
captures all 13 threads):

| process | thread(s) | leaf (where its life goes) |
|---|---|---|
| API server | main | `asyncio/runners.py:118` — uvloop epoll, idle between HTTP requests |
| API server | watcher | `selectors.py:415` — select() on the EngineCore pipe |
| API server | executor worker | `futures/thread.py:89` — idle `queue.get`; **2,932 samples: born ~0.4 s into capture** (pool workers spawn lazily — partial attendance dates a thread's birth) |
| EngineCore | **main** | **`torch/cuda/streams.py:254` `synchronize` — 2,729/2,969 = 92% of wall time waiting for the GPU**; the remaining ~240 samples are the entire Python execution of the engine loop |
| EngineCore | input / output | `zmq poll` / condition wait — request/response plumbing |
| EngineCore | **lmcache MQ loop** | `zmq poll` (`mq.py:236`) — connector's client thread, asleep |
| EngineCore | **lmcache periodic** | timed `wait` (`periodic_thread.py:346`) — heartbeat |
| EngineCore | liveness, usage reporter, 2× tqdm TMonitor | assorted waits/sleeps |
| resource_tracker | main | pipe read, pure standby |

Bookkeeping details that generalize: leaves are the *last Python frame* before a C
call (epoll/zmq/cuda-sync all look alike — wall mode cannot distinguish sleeping
from native-busy; `cuda.synchronize` is "waiting on GPU" by function semantics, not
by graph shape). `threading.py:355` vs `:359` separates wait-forever threads from
timed-wakeup threads. Summing the leaves: **99.3% of the graph's area is waiting**;
~254 samples are all the Python vLLM actually ran in 30 s of mixed load.

## Surprise finding: L1's usable capacity is 0.8× nominal

The TTFT tail (~7 requests at 600–1400 ms) exposed a leak in the "no recompute"
design: the server evicted 5,825 chunks and L1 usage settled at 71.7%. Cause
(`lmcache/v1/distributed/config.py:217,220`): the eviction loop wakes every 1 s and,
once usage ≥ **`trigger_watermark` (default 0.8)**, LRU-discards ~**`eviction_ratio`
(0.2)** of keys in one swing — headroom an async evictor needs so the hot-path
`reserve_write` never meets a full pool. So a 40 GB L1 starts shedding at 32 GB; the
30 GiB doc set plus query-suffix chunks brushed that line, and LRU-cold docs got
recomputed on next touch. **Rule: size working set < 0.8 × L1** (48 GB here would
have been clean); knobs `--eviction-trigger-watermark` / `--eviction-ratio`.

## Method notes (for reuse)

- py-spy SVGs are inferno icicles: root at min-y, `fg:x`/`fg:w` on each rect are
  exact sample counts — parseable without the browser (scratchpad `flame_top.py`
  yields self/total tables).
- `gil` and `wall` differ only in which snapshots count (GIL-holder only vs every
  thread every tick). Read wall first for structure, gil for CPU; a busy-poll claim
  needs both graphs abnormal. Width = time share under the graph's denominator —
  never call counts, never time order (siblings sort alphabetically; that's what
  makes before/after diffs stable).
- Thread arithmetic works only in wall mode, and "width = k × full attendance" is an
  inference, not proof (`--threads` splits identity at the cost of fragmenting
  hotspots; `py-spy dump` is the quick cross-check).

## See also

- [4_qwen3_bench.md](4_qwen3_bench.md) / [6_l2_gain_bench.md](6_l2_gain_bench.md) — the tiered-sweep method this scenario reuses (pool < volume < tier)
- [1_control_vs_data_plane.md](1_control_vs_data_plane.md) — why lookup/retrieve split keeps the Python layer thin
- `profiling/flame.sh` header — the three traps (sudo -n, PATH, chown) the wrapper closes
- The 07-21 nsys capture (task ④, local records): D2H 56 GB/s vs H2D 32 GB/s, allocator = 18.9% of store-range time — the native-side numbers this run cross-validates from the Python side
- [LMCache PR #4088](https://github.com/LMCache/LMCache/pull/4088) — `lmcache tool flamegraph`; its gil/wall modes are what `flame.sh` wraps (perf/bcc modes are blocked on this box)

## Conclusion — "Can you see any obvious hotspot?"

**No — an evidenced No, and the negative result is the answer's correct shape.**

Evidence, three measurements that interlock:

1. **Server**: GIL held 5.2% of ticks, identical across store and mixed phases —
   while the process was moving 30 GB of KV. All 12 threads wait event-driven, no
   busy-poll; 73% of awake time sits in the two pool threads, and the largest
   single Python consumer (the Lookup module) is ~0.03 of a core.
2. **vLLM tree**: 6.8% + 6.4% GIL per process; the EngineCore main thread spends
   92% of wall time inside `cuda.synchronize`; the biggest Python item anywhere is
   the detokenizer at 2.6% of a core. The LMCache connector hooks: 0.1%.
3. **The copy body itself was sighted, not seen**: the wall−gil gap on the
   affinity worker (55 ticks populate / 19 mixed) proves the copy is native and
   GIL-released — py-spy's structural blindness to it *is* the measurement.

Why "No", on two layers. *Structurally*, both systems deliberately push the hot
path out of Python — CUDA/DMA copies, blake3 in C, msgspec in C, GIL released
throughout; a large plateau would have been a bug, and this is a clean bill of
health. *Operationally*, a bottleneck is a property of (code × operating point),
and at in-flight 3 nothing is CPU-bound anywhere — all Python combined is under
a fifth of one core and the GPU idles between small batches. Where the e2e time
actually goes, the TTFT split already names: 0–99 ms on GPU hit vs 400–599 ms on
L1 load. **That ~400 ms delta is the real end-to-end "hotspot", and it lives in
the native copy path and its scheduling — invisible to py-spy by construction,
measurable by nsys (task ④ takes over).**

What the exercise yielded besides the No: the ranked Python watch-list with its
scaling laws (blake3 ∝ 1/chunk, msgspec ∝ QPS, detokenizer ∝ concurrency ×
output length, ceiling ≈ in-flight 45); one genuine bug-grade discovery (usable
L1 = 0.8 × nominal, which explained the recompute tail); and one fixable crumb
(per-operation `inspect.signature`, filed for task ⑤). One sentence: **the
flamegraph proved the cost is not in Python, legitimately redirecting all further
optimization attention to the native/GPU side — the negative space is the answer,
and elimination has done its step.**
