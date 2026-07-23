# Pending experiments — onboarding (bo_ob)

Living list of **suspended experiments** across all parts. Report-window only.

> **铁律:以下全部只在报告窗口跑,绝不在未获明确同意下自动抢卡。** 不打扰他人的 GPU 作业。

Last updated: 2026-07-22. Detailed handoff history in `records/` (gitignored, local-only).

## All suspended items (5)

| # | Experiment | Cards | Time | Runnable now? | Note |
|---|---|---|---|---|---|
| **B** | ~~chunk-size 16→256~~ / prefetch-in-flight 8→64 re-test | any **1** | ~30min | ✅ | **chunk-size DONE 2026-07-22** (see below); only in-flight sub-item left |
| **③** | P2P loading throughput (nsys, Part 3) | same-node **2** | ~40min | ❌ | extra prereq below |
| **D** | DSv4-Flash with / without LMCache | rtx-1 ×2 (TP=2) | — | ❌ | highest academic value (MLA compresses KV) |
| **A** | ~~real 2-node P2P (Part 2 ③ closeout)~~ | 024×1 + 026×1 | — | ✅ | **DONE 2026-07-23**: 1.44–1.63×, RoCE 5–10 GB/s beat loopback; see `8_p2p_2node_bench.md` |
| **C** | Kimi-K2.6 recipe | rtx-1 full node ×8 (TP=8) | — | ❌ | long-term |

**Priority: ③ > D > C** (A and B's main gap done). Optional tiny B-followup: in-flight sweep via
`bench-l2.sh` if we want that dimension too.
(③ annotations ready, only needs 2 cards; D high value but needs a full 2-card node; C needs a
whole node.) A's cross-node result (5–10 GB/s real RDMA) sharpens ③'s question: nsys should see
those load timings same-node.

## ③ — P2P loading throughput (Part 3)

**Goal:** use nsys to measure P2P transfer load throughput between two LMCache instances (range duration → bytes/time).

**Two prereqs (each half-ready):**
1. **2 cards on the same node** (two vLLM instances transferring to each other).
2. **The running lmcache must carry the P2P annotations** — they live on branch `feat/p2p-nvtx-annotations` ([PR #1](https://github.com/BoJiang03/lmcache_test/pull/1), clone at 024 `/data1/bo/LMCache`), **not yet installed into any running venv**. Before ③: `uv pip install` that branch into the target vLLM venv (or PYTHONPATH inject), otherwise `nvtx_startend_sum` comes back empty.

**Reusable:** `profiling/nsys.sh` (launch/start/stop, `NSYS_TARGET`), `profiling/workload.sh`; P2P bring-up via `l2_support/p2p-demo.sh`.

**vs A:** both are 2-card P2P. A tests real cross-node bandwidth (024+026 simultaneous); ③ only needs same-node 2 cards for load throughput — easier to land if any node frees 2 cards.

## B — chunk-size DONE (2026-07-22), in-flight sub-item remains

**Ran** the chunk-size half on borrowed rtx-026 GPU1 (lmcache 0.5.1), `l2-gain.sh`, L2=fs_native on
tmpfs, 8×20k-tok, post-reset re-read TTFT. Full record:
`records/2026/07/22/3_expB_l2_chunksize_sweep.md`.

Result: no-l2 recompute 1251ms; fs_native @chunk16 1262ms (0.99×); fs_native @chunk256 1197.7ms
(1.045×). **chunk 16→256 = −64ms (−5.1%).** The Part 2 "inferred ~0.78s pure polling wait" is
**corrected to a measured ~64ms** end-to-end effect (polling overlaps work; direction right,
magnitude ~10× smaller). Near-break-even regime here (tmpfs L2, 20k ctx) keeps the absolute L2 gain
small; real-disk/longer-ctx would amplify it.

**Remaining (optional):** the in-flight 8→64 dimension — lives in `bench-l2.sh` (`--in-flight`,
an L2-adapter microbench), NOT `l2-gain.sh`. Cheap, 1 card, if we want it.

## A / C / D — details

- **A:** `check-rdma.sh` / `p2p-demo.sh` / `p2p_case.py` all validated (single-node dual-card link works, hit_chunks 0→5128). Only unverified: single-node NIXL was likely loopback, so 1.30× can't extrapolate to 2-node.
- **C:** 555G weights at `/data1/bo/models`, needs full rtx-1 node.
- **D:** 149G fp8 ready; MLA compresses KV → near the bandwidth break-even line L2/P2P gains should amplify, giving Part 2 a second data point.

## Environment memos (cross-session)

- **`/home` and `/data1` are per-node, not shared** (distinct fsids). Cross-node run = rsync scripts + install env on each.
- **026 = 172.16.176.28**, passwordless ssh + sudo -n; lmcache **0.5.1** (024 is 0.5.2rc1); Qwen3-8B at `/dev/shm/models/`; py-spy 0.4.2 in 026 `~/.local/bin`.
- **nsys dual-install trap:** on 026 `/usr/local/bin/nsys`=2025.3.2 vs `/usr/local/cuda/bin/nsys`=2025.5.2, mutually invisible sessions; always use the scripts' PATH (=2025.5.2). `nsys start` rejects `-t` (trace fixed at launch).
- **MP mode: the real KV copy runs in the lmcache *server* process**, so nsys needs `NSYS_TARGET=server`.

## Loose ends

- **setup/ py pilot (2026-07-22):** all 10 setup/ scripts now have a parallel `.py` port
  (`_common.py` engine + 9 scripts; shares `run/`/`logs/`/`lmcache-server.conf` with the sh
  family). No-GPU paths (usage/status/stop/error paths/process-group kill) verified; the
  **GPU paths (`start`, bench, dp) are UNVERIFIED** — next GPU window, bring up Qwen3-8B once
  via `.py` and once via `.sh` and compare. Until then **the `.sh` files stay canonical**.
  If the pilot passes, decide whether to port `l2_support/` + `profiling/` the same way.
- **Unpushed:** onbording (bo_ob) local is ahead of origin/main by local commits (Part 3 harness `da80511`, code_structure `2657fe9`, …). Push only on request.
- **PR #1** open on `BoJiang03/lmcache_test` (base=dev), **not sent upstream**, awaiting review.
- **Background watcher** `b8cmo5ynf` still watching 024 free cards (notify-only, never grabs). Single-side only → limited value for ③/A's 2-card need; disable/repoint TBD.
