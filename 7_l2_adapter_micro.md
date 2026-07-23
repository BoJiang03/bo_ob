# L2 adapter microbench notes (fs vs fs_native vs resp)

Part 2's "pick 3 different L2 adapters and compare their performances using
`lmcache bench l2`", run via `l2_support/bench-l2.sh` (zero GPU — safe anytime). Raw
outputs in `l2_support/bench-results/l2-adapters/` (tracked). Headline: **the adapter
ranking depends on object size** — at our real serving chunk size (2.25 MB) the order
flips vs the tool's 256 KB default.

## What the tool measures

`lmcache bench l2` instantiates one adapter in-process (no vLLM, no GPU, no MP server)
and drives the three async `L2AdapterInterface` calls phase by phase, waiting on the
matching `query_*`:

| Phase | API | Real-system counterpart |
|---|---|---|
| Store | `submit_store_task` | write-through, L1 → L2 |
| Lookup | `submit_lookup_and_lock_task` | the L2 probe inside `lookup()` |
| Load | `submit_load_task` | prefetch backfill, L2 → L1 |

The three adapters differ only in implementation, not role: **fs** = pure-Python POSIX
files (GIL, per-key open/write/close); **fs_native** = C++ worker threads writing the
same disk; **resp** = TCP loopback to a Redis 7 container (in-memory store + network
protocol).

## Runs

*2026-07-23, rtx-024, zero GPU. Result dirs `l2-adapters/20260723-000054` (256 KB) and
`20260723-000111` (2304 KB); `20260721-170504` is the earlier 256 KB run it reproduces
(±20%, normal for a page-cache microbench).*

Common workload: `num_keys=32 in_flight=4 rounds=5 (warmup 2) lookup_max_hit_rate=1.0`,
files on `/data1` (xfs, NVMe). The two runs differ only in `DATA_SIZE_KB`:

- **256** — the script default ("~one KV chunk" for a small-KV model);
- **2304** — our actual serving object: chunk 16 × 144 KB/token (Qwen3-8B) = 2.25 MB,
  the same objects doc 6's end-to-end run moved.

## Results

### Store (L1 → L2)

| adapter | 256 KB MB/s | 2304 KB MB/s | p50 256 KB | p50 2304 KB |
|---|---|---|---|---|
| fs | 687 | **5,761 (×8.4)** | 47.4 ms | 49.3 ms (**flat**) |
| fs_native | 8,370 | **30,944** | 3.9 ms | 9.3 ms |
| resp | 3,280 | **3,721 (flat)** | 9.5 ms | **76.7 ms (×8)** |

### Load (L2 → L1)

| adapter | 256 KB MB/s | 2304 KB MB/s | p50 256 KB | p50 2304 KB |
|---|---|---|---|---|
| fs | 1,240 | 9,441 | 25.9 ms | 30.4 ms (flat) |
| fs_native | 34,320 | **43,958** | 0.93 ms | 6.6 ms |
| resp | 5,534 | **4,065 (dropped)** | 5.9 ms | **66.4 ms (×11)** |

### Lookup (query + lock) — size-independent, both runs agree

| adapter | p50 | ops/s |
|---|---|---|
| fs_native | **0.3 ms** | ~450k |
| resp | 1.0 ms | ~120k |
| fs | 6.8 ms | ~19k |

Ranking at 256 KB: fs_native ≫ resp ≫ fs. **At 2304 KB: fs_native ≫ fs ≫ resp.**

## Three cost shapes

- **fs: fixed per-op cost, byte-count-insensitive.** p50 doesn't move when objects grow
  9× — the cost is the pure-Python path (GIL, per-key syscall round trip), not moving
  bytes. Bigger objects amortize it: throughput scales almost linearly. At real chunk
  size, "pure-Python files" stops being a bottleneck at all (5.8 GB/s store dwarfs one
  instance's write-through rate).
- **resp: byte-proportional cost.** p50 grows 8–11× with the 9× object — two loopback-TCP
  copies plus single-threaded Redis handling 2.25 MB values — so throughput is capped
  around ~4 GB/s regardless of size. The bigger the chunk, the worse "in-memory store
  over a network" fares against "local file + page cache"; at 2.25 MB it loses to
  pure-Python fs.
- **fs_native: low fixed cost *and* fast byte path** (C++ workers, 44 GB/s load). Wins
  every phase at every size.

Cross-check with doc 6: fs_native's 44 GB/s microbench load is the same phenomenon as
the end-to-end run's 625-chunk doc loads completing in 40–56 ms (25–34 GB/s server-side,
`storage_manager.py:709` log lines) — microbench and e2e agree.

## Takeaways

1. **Adapter choice couples with chunk size** (object bytes = chunk_size ×
   cache_size_per_token). A ranking measured at one object size doesn't transfer:
   resp beats fs at 256 KB objects and loses to it at 2.25 MB ones.
2. Lookup cost is pure per-key overhead (metadata), where fs's 6.8 ms vs fs_native's
   0.3 ms matters because `lookup()` sits on the TTFT path even for L1 hits.
3. Standing caveat: every "disk" number here is **page cache speed** (30.9 GB/s writes
   exceed NVMe hardware); the microbench compares adapter implementation overhead, not
   storage media. Cold-media performance would reshuffle fs/fs_native vs resp again.

## See also

- [6_l2_gain_bench.md](6_l2_gain_bench.md) — end-to-end gain of fs_native; the e2e numbers these microbenches decompose
- [2_kv-cache-shapes.md](2_kv-cache-shapes.md) — where 144 KB/token (hence 2.25 MB/chunk) comes from
- `l2_support/bench-l2.sh` header — adapter matrix, Redis autostart, all knobs
