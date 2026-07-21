#!/usr/bin/env bash
# bench-compare.sh — quantify LMCache's gain: run the same workload twice on the same
# model — (1) vLLM + LMCache, (2) plain vLLM baseline — and print a side-by-side
# comparison of the key metrics.
#
# Fairness notes:
#   - the baseline still has vLLM's own GPU prefix cache, so the comparison isolates
#     what LMCache *adds* (CPU-tier capacity beyond the GPU KV pool). Pick a KV_VOLUME
#     larger than the vLLM GPU KV pool for the difference to show.
#   - the lmcache server is restarted before run (1) so no cache from earlier
#     experiments leaks in, and stopped for run (2) so its GPU buffer is freed.
#   - tokens-per-GB for the baseline is taken from run (1)'s summary, so both runs
#     derive identical document sets (same seed, same working-set size).
#
# Usage:
#   MODEL=Qwen3.6-35B-A3B KV_VOLUME=40 ./bench-compare.sh [extra bench args...]
# Env config: everything bench.sh takes, plus KEEP_UP=1 to skip the final teardown.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL=${MODEL:-Qwen3-8B}
WORKLOAD=${WORKLOAD:-long-doc-qa}
KV_VOLUME=${KV_VOLUME:-10}
OUT_ROOT=${OUT_ROOT:-$COMMON_DIR/bench-results}
CMP_DIR="$OUT_ROOT/$(date +%Y%m%d-%H%M%S)-$MODEL-$WORKLOAD-${KV_VOLUME}GB-compare"

echo "=== bench-compare: $MODEL / $WORKLOAD / ${KV_VOLUME}GB -> $CMP_DIR"
# clean slate: instance down, lmcache server down (run 1 restarts it with the right chunk)
"$COMMON_DIR/$MODEL.sh" stop
"$COMMON_DIR/lmcache-ctl.sh" stop

echo "=== [1/2] WITH LMCache"
MODEL=$MODEL WORKLOAD=$WORKLOAD KV_VOLUME=$KV_VOLUME \
    OUT_DIR="$CMP_DIR/with-lmcache" "$COMMON_DIR/bench.sh" "$@"

TOKENS_PER_GB=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config']['tokens_per_gb_kvcache'])" \
    "$CMP_DIR/with-lmcache/bench_summary.json")

echo "=== [2/2] WITHOUT LMCache (baseline, tokens-per-gb=$TOKENS_PER_GB)"
"$COMMON_DIR/$MODEL.sh" stop
"$COMMON_DIR/lmcache-ctl.sh" stop   # baseline shouldn't keep the server's GPU buffer around
MODEL=$MODEL WORKLOAD=$WORKLOAD KV_VOLUME=$KV_VOLUME \
    NO_LMCACHE=1 TOKENS_PER_GB=$TOKENS_PER_GB \
    OUT_DIR="$CMP_DIR/no-lmcache" "$COMMON_DIR/bench.sh" "$@"

if [[ ${KEEP_UP:-0} != 1 ]]; then
    "$COMMON_DIR/$MODEL.sh" stop
fi

python3 - "$CMP_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
w = json.load(open(os.path.join(d, 'with-lmcache', 'bench_summary.json')))['results']
n = json.load(open(os.path.join(d, 'no-lmcache', 'bench_summary.json')))['results']
rows = [
    ('mean TTFT (ms)',             'mean_ttft_ms',            'lower'),
    ('p50 TTFT (ms)',              'p50_ttft_ms',             'lower'),
    ('p90 TTFT (ms)',              'p90_ttft_ms',             'lower'),
    ('p99 TTFT (ms)',              'p99_ttft_ms',             'lower'),
    ('mean request latency (ms)',  'mean_request_latency_ms', 'lower'),
    ('input throughput (tok/s)',   'input_throughput',        'higher'),
    ('mean decode speed (tok/s)',  'mean_decode_speed',       'higher'),
    ('elapsed time (s)',           'elapsed_time',            'lower'),
]
print(f"\n{'metric':<28} {'with LMCache':>14} {'no LMCache':>14} {'gain':>9}")
print('-' * 68)
for label, key, better in rows:
    a, b = w[key], n[key]
    gain = (b / a) if better == 'lower' else (a / b)   # x-times better with LMCache
    print(f"{label:<28} {a:>14.1f} {b:>14.1f} {gain:>8.2f}x")
print(f"\nrequests ok: with={w['successful_requests']}/{w['total_requests']}"
      f"  no={n['successful_requests']}/{n['total_requests']}")
EOF
echo "=== results in $CMP_DIR"
