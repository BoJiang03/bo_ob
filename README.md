# bo_ob — LMCache onboarding homework

Harnesses, benchmarks, and write-ups for the four onboarding parts. Layout: one
directory per part (`setup/`, `l2_support/`, `profiling/`, `code_structure/`), plus
numbered root docs (`1_…` – `8_…`) for the durable findings and
`pending_experiments.md` for what is still suspended. Start at
[code_structure/overview.md](code_structure/overview.md) for how LMCache itself is
organized; each script prints its own usage from its header comment.

## Companion LMCache repo

Code contributions that came out of this homework live in my LMCache fork:
**[BoJiang03/lmcache_test](https://github.com/BoJiang03/lmcache_test)**

- [PR #12](https://github.com/BoJiang03/lmcache_test/pull/12) —
  `--no-warmup` for `lmcache bench engine` (branch `feat/engine-bench-warmup-toggle`).
  Built for, and used by, the 2-node P2P case in
  [8_p2p_2node_bench.md](8_p2p_2node_bench.md): the bench's hard-coded warmup made
  first-touch scenarios unmeasurable.
- [PR #1](https://github.com/BoJiang03/lmcache_test/pull/1) —
  NVTX annotations on the P2P transfer path (branch `feat/p2p-nvtx-annotations`),
  prep for the Part 3 nsys throughput measurement in `profiling/`.

The `code_structure/` docs cite `file:line` into that clone (`/data1/bo/LMCache` on
rtx-024), matching upstream `dev` paths.
