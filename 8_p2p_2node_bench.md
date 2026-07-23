# P2P KV sharing across two physical nodes (benefit case)

Part 2's last item: "Run P2P KV cache sharing across two physical nodes; create a case to
show the benefit". Done 2026-07-23 on **rtx-024 (node A) + rtx-026 (node B)**, GPU 7 on
each, manually (raw `lmcache server` / `vllm serve` commands, no wrapper scripts). Raw
outputs in `l2_support/bench-results/p2p/harness-*` (tracked). This closes
`pending_experiments.md` experiment A — the single-node run's 1.30× had a standing caveat
that loopback NIXL "can't extrapolate to 2 nodes"; it turned out to under-predict.

**Headline: cross-node P2P served a never-seen-locally 10k-token prefix at p50 366 ms vs
528 ms full recompute — 1.44× (repeat arm 1.63×), with per-load RDMA throughput of
5–10 GB/s, ~2–3× above this node's recompute break-even line.**

## What P2P is (one paragraph)

Each node runs its own vLLM + MP server pair; vLLM only ever talks to its **local**
server (`LMCacheMPConnector` → `tcp://localhost:5558`) and needs zero config changes.
The servers federate among themselves: a **coordinator** (pure HTTP roster, never touches
KV) tells each server which peers exist; on a local L1 miss the server RPCs the peer
("lock these chunks, give me their addresses") and NIXL **RDMA-reads the bytes straight
out of the peer's L1** into local L1, from where the normal L1→GPU retrieve proceeds.
P2P is plumbed as just another L2 adapter — read-only (store to a peer is a no-op),
single-hop, L1-resident data only. Registration carries a quad
`(instance_id, ip, p2p_advertised_url, mq_port)`: only the IP and the transfer-channel
port need flags; the ZMQ port is self-reported.

## Setup

| Knob | Value |
|---|---|
| Nodes | A = rtx-024 (172.16.176.27), B = rtx-026 (172.16.176.28), GPU 7 each |
| Fabric | RoCE via mlx5_0 (`enp41s0f0np0`), ACTIVE on both; NIXL 1.3.1 in both venvs |
| Coordinator | `lmcache coordinator` on A:9300 |
| Servers | `--l1-size-gb 16 --chunk-size 16 --l1-align-bytes 65536 --p2p-advertise-url <self>:9401 --p2p-transfer-engine nixl` (9401 because a foreign process squats 127.0.0.1:9400 on **both** nodes) |
| vLLM | Qwen3-8B from `/dev/shm`, `--gpu-memory-utilization 0.3`, `kv_load_failure_policy=recompute` |
| Workload | `lmcache bench engine --workload long-doc-qa --kv-cache-volume 10` → 7 docs × ~10k tokens (9.6 GB KV, fits A's 16 GB L1), `qpd=1`, `in-flight=1` |

The measuring tool needed a feature that didn't exist: engine bench **hard-codes a warmup
pass** (each doc sent once before measurement), which populates the local cache and makes
first-touch scenarios — the only place P2P's benefit lives — unmeasurable. Added
`--no-warmup` ([lmcache_test PR #12](https://github.com/BoJiang03/lmcache_test/pull/12)):
`EngineBenchConfig.warmup` → `BaseWorkload.run(run_warmup=...)`, one base-class change
covering all five workloads. Run here via `PYTHONPATH=/data1/bo/lmcache-warmup-toggle`
(a git worktree of the branch; bench is a pure HTTP client, so the serving stack stays on
the installed 0.5.1).

## The case: three arms, same 7 documents

1. **A-local** — normal bench against A (warmup included). The warmup populates A's
   GPU/L1; the measured phase is A reading its own caches. Baseline "cache is local".
2. **B-P2P** — bench against B with `--no-warmup`, same docs: B has never seen them,
   A has. Every request must cross the wire. (Ran twice; the second run — mislabeled
   `harness-B-cold` — accidentally became a clean repeat, see finding 2.)
3. **B-recompute** — same command after removing A from the fleet (stop A's server →
   coordinator expires it in ~30 s → B's `p2p_peer_count` drops to 0) and restarting B's
   server cold. Same docs, nowhere to fetch from: pure prefill.

| Arm | serves from | TTFT mean | p50 | p90 | steady band |
|---|---|---|---|---|---|
| A-local | own GPU prefix cache | 42 ms | 41 ms | 45 ms | 41–45 ms |
| **B-P2P** | **A's L1 over RDMA** | 417 ms | **366 ms** | 512 ms | 360–383 ms (first: 707) |
| B-P2P repeat | same | 368 ms | 323 ms | 474 ms | 302–378 ms |
| B-recompute | full prefill | 570 ms | **528 ms** | 652 ms | 522–543 ms (first: 816) |

Both B arms have a symmetric first-request outlier (707 / 816 ms — connection & kernel
warm-up, not arm-specific). Decode ~80 tok/s in every arm (sequential, no queueing story
at in-flight 1).

**Evidence it was really P2P** (three independent layers): B's `/metrics` counted
`l2_load_completed{l2_name="p2p"} = 14` requests totalling **8,750 chunks = 14 × 625 =
exactly 14 full docs** across the two P2P arms; the per-load throughput histogram puts
13/14 loads in the **5–10 GB/s** bucket; and the recompute arm's fresh server log has
**zero** prefetch-completion lines and zero p2p counters — nothing served that arm but
the GPU.

## Why 1.4–1.6×: the break-even line, revisited

Same yardstick as the L2 work: a KV path must beat the GPU's *KV production rate*
(prefill tok/s × 144 KB/token) to beat recompute.

- B recomputes 10k tokens in ~525 ms → ~19k tok/s → **break-even ≈ 2.75 GB/s**.
- The cross-node pipe delivered **5–10 GB/s** → clearly above the line → P2P wins.
- Decomposition checks out: 1.38 GB ÷ ~7 GB/s ≈ 200 ms transfer + ~150 ms fixed cost
  (L1→GPU retrieve + first-token forward, paid by *both* arms) ≈ 350 ms ≈ the observed
  steady band. The fixed cost is why a 2–3× bandwidth margin compresses to 1.5×
  end-to-end — same dilution as every tier before it.

**The surprise: real RDMA beat loopback.** Single-node (both peers on one box, NIXL over
loopback/shm) measured ~3 GB/s end-to-end → 1.30×. The real RoCE path does 5–10 GB/s.
The old caveat "single-node numbers can't extrapolate" was right, but in the opposite
direction than feared.

## Two mechanism findings (worth more than the headline)

1. **P2P-loaded chunks are not retained in B's L1.** After 14 full-doc loads, B's L1
   held 56 objects ≈ 126 MB — only the tail/question chunks B computed itself; the
   prefix chunks fetched from A were temporary objects, dropped after the retrieve.
   Consequence: the benefit is **per-first-touch** — a second cold read of the same
   prefix crosses the wire again (which is precisely what made the repeat arm valid).
   Contrast with disk L2, whose loads we saw backfill (and churn) L1 in doc 6.
2. **engine-bench docs don't depend on `--seed`.** Content is literally
   `"Document {id}: hi hi hi…"`; seed only shuffles the schedule. So "different seed =
   cold control" is invalid — my first "cold" arm silently re-measured P2P (7 more
   loads in the counters gave it away; the client TTFTs alone looked plausible). A true
   cold arm requires removing the peer + a fresh local server. Result dir
   `harness-B-cold` is that mislabeled repeat arm; `harness-B-recompute` is the real
   control. Follow-up feature idea: a doc-id offset flag.

## Operational gotchas (all bitten this session)

- **Pin the server to a GPU too.** `lmcache server` without `CUDA_VISIBLE_DEVICES`
  opens a ~554 MiB context on physical GPU 0 — which was another user's busy card.
  `p2p-demo.sh` has the same latent bug (it pins vLLM only).
- **Model-name mismatch:** the MP server registers the model *path*
  (`/dev/shm/models/Qwen3-8B`); bench auto-detect picks the first served name
  (`Qwen3-8B`) → `resolve_tokens_per_gb` dies. Pass `--model <path>` explicitly.
- `/data1` (and the PYTHONPATH worktree) is per-node — run the bench client from the
  node that has it; a wrong PYTHONPATH is silently ignored and you get the installed
  package (visible as "no --no-warmup flag").
- `pkill -f "vllm serve"` matches the ssh remote shell carrying the pattern → kills
  itself mid-teardown, exit 255. Kill by exact pid.
- The 30 s coordinator expiry is the clean way to take a peer out of the fleet
  (`p2p_peer_count` → 0) without touching the surviving node.

## Takeaway

Two-node P2P works and pays: same-prefix TTFT 528 → 366 ms (1.44×, repeat 1.63×) with
zero recompute, because real RoCE RDMA (5–10 GB/s) clears the 2.75 GB/s break-even line
that loopback barely crossed. As with L1/L2 before it, the per-request gain is diluted by
fixed retrieve+forward costs; the structural value is that a prefix computed anywhere in
the fleet never needs prefill again — but note it's re-fetched on every first touch,
since peer loads don't persist in the local L1.

## See also

- [6_l2_gain_bench.md](6_l2_gain_bench.md) — the disk-L2 arm of the same story; break-even framework and L1-backfill contrast
- [7_l2_adapter_micro.md](7_l2_adapter_micro.md) — adapter-level microbench; P2P can't be driven by `lmcache bench l2` (read-only adapter vs the tool's store-then-load protocol)
- [1_control_vs_data_plane.md](1_control_vs_data_plane.md) — lookup-starts-prefetch / retrieve-does-H2D, the fixed cost both arms pay
- [2_kv-cache-shapes.md](2_kv-cache-shapes.md) — 144 KB/token, the other half of the break-even arithmetic
- `l2_support/p2p-demo.sh` header — scripted bring-up of the same topology (ports, flags, the bind-0.0.0.0 trap)
- [lmcache_test PR #12](https://github.com/BoJiang03/lmcache_test/pull/12) — the `--no-warmup` feature this measurement depends on
