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
  reproduce on 0.5.1).
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

## Answering the homework question

"Any obvious hotspot?" — **No, and the absence is the finding.** Two layers to that:

*Structurally*, LMCache's design goal is to push the hot path out of Python (CUDA/DMA
copies, blake3 in C, msgspec in C, GIL released throughout); a large Python hotspot
would have been a bug.

*Operationally*, a bottleneck is a property of an operating point, not of code — and
at this operating point (in-flight 3) **nothing is CPU-bound anywhere**: all Python
combined is ~15% of one core, the largest single consumer 2.6%, and even the GPU
mostly idles between small batches. A low-load flamegraph *cannot* name a bottleneck;
what it yields is structure (which code is native vs Python) and slopes (what grows
with what — the detokenizer and control-plane annotations above). Naming an actual
bottleneck would take a saturating run (crank in-flight until the GPU or a Python
component pegs) and a fresh capture — a natural follow-up, out of scope here. What the
flamegraph delivers is (a) proof the cost is not in Python, redirecting all further
effort to the native/GPU side that nsys measures, (b) the per-chunk control-plane
watch-list above, (c) a health certificate for the event-driven design (no busy-poll),
and (d) the watermark discovery, courtesy of the TTFT tail it made us explain.
Flamegraphs are the mouth of the funnel: they answer "where should we look?", not
"how fast is it?" — for timing and ordering, the nsys timeline (tasks ③/④) takes
over.

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
