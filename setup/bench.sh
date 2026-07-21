#!/usr/bin/env bash
# bench.sh — benchmark a vLLM+LMCache instance with `lmcache bench engine`,
# per https://docs.lmcache.ai/getting_started/benchmarking.html and
#     https://docs.lmcache.ai/cli/bench.html
#
# Ensures the model instance is up (via its per-model script, which also brings up the
# LMCache server), then runs the chosen workload against it.
#
# The key knob is KV_VOLUME (--kv-cache-volume, GB): the benchmark derives its document
# set from this working-set size. If the volume fits within the cache tiers (vLLM GPU KV
# pool + lmcache server L1), everything hits after warmup; if it exceeds capacity, LRU
# eviction kicks in and cache misses appear.
#
# Usage:
#   ./bench.sh [extra `lmcache bench engine` args...]
# e.g.
#   ./bench.sh                                # long-doc-qa over a 10GB working set
#   KV_VOLUME=60 ./bench.sh                   # overflow GPU pool + 20GB L1 -> misses
#   WORKLOAD=multi-round-chat ./bench.sh --mrc-qps 2 --mrc-duration 120
#   MODEL=Qwen3.5-9B ./bench.sh               # bench another managed model
#   NO_LMCACHE=1 TOKENS_PER_GB=7281 ./bench.sh   # baseline: vLLM without LMCache
#
# NO_LMCACHE=1 starts the instance without the LMCache connector (plain vLLM baseline,
# still with vLLM's own GPU prefix cache). There is no lmcache server to auto-detect
# tokens-per-GB from, so TOKENS_PER_GB is required — take config.tokens_per_gb_kvcache
# from any previous with-LMCache bench_summary.json of the same model.
#
# Env config: MODEL (default Qwen3-8B; needs a matching ./<MODEL>.sh), WORKLOAD,
#   KV_VOLUME (GB), VLLM_PORT, LMCACHE_HTTP_PORT, OUT_ROOT, OUT_DIR,
#   NO_LMCACHE, TOKENS_PER_GB
# Results (CSV + JSON summary) land in bench-results/<timestamp>-<model>-<workload>-<vol>/

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL=${MODEL:-Qwen3-8B}
# the MP server registers models by path, so --model must be the exact local path the
# model script serves (see _common.sh); look in the known weight roots
if [[ -z ${MODEL_PATH:-} ]]; then
    for d in /dev/shm/models /data1/bo/models; do
        if [[ -e "$d/$MODEL" ]]; then MODEL_PATH="$d/$MODEL"; break; fi
    done
fi
: "${MODEL_PATH:?no local weights found for $MODEL — set MODEL_PATH}"
WORKLOAD=${WORKLOAD:-long-doc-qa}
KV_VOLUME=${KV_VOLUME:-10}
VLLM_PORT=${VLLM_PORT:-8000}
LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT:-8080}   # MP server's HTTP endpoint (tokens-per-GB autodetect)
OUT_ROOT=${OUT_ROOT:-$COMMON_DIR/bench-results}

NO_LMCACHE=${NO_LMCACHE:-0}
TOKENS_PER_GB=${TOKENS_PER_GB:-}

if [[ ! -x "$COMMON_DIR/$MODEL.sh" ]]; then
    echo "ERROR: no model script $COMMON_DIR/$MODEL.sh" >&2
    exit 1
fi

cache_args=(--lmcache-url "http://localhost:$LMCACHE_HTTP_PORT")
if [[ $NO_LMCACHE == 1 ]]; then
    if [[ -z $TOKENS_PER_GB ]]; then
        echo "ERROR: NO_LMCACHE=1 needs TOKENS_PER_GB (no lmcache server to auto-detect from;" >&2
        echo "       take config.tokens_per_gb_kvcache from a with-LMCache bench_summary.json)" >&2
        exit 1
    fi
    cache_args=(--tokens-per-gb-kvcache "$TOKENS_PER_GB")
fi

# idempotent: no-op if the instance is already running — but it does NOT check which
# mode the running instance was started in; bench-compare.sh stops it between modes
NO_LMCACHE=$NO_LMCACHE "$COMMON_DIR/$MODEL.sh" start

suffix=""
[[ $NO_LMCACHE == 1 ]] && suffix="-nolmcache"
OUT_DIR=${OUT_DIR:-$OUT_ROOT/$(date +%Y%m%d-%H%M%S)-$MODEL-$WORKLOAD-${KV_VOLUME}GB$suffix}
mkdir -p "$OUT_DIR"

echo "bench: workload=$WORKLOAD model=$MODEL kv-cache-volume=${KV_VOLUME}GB lmcache=$([[ $NO_LMCACHE == 1 ]] && echo no || echo yes) -> $OUT_DIR"
lmcache bench engine \
    --engine-url "http://localhost:$VLLM_PORT" \
    "${cache_args[@]}" \
    --model "$MODEL_PATH" \
    --workload "$WORKLOAD" \
    --kv-cache-volume "$KV_VOLUME" \
    --no-interactive \
    --json \
    --output-dir "$OUT_DIR" \
    "$@"
echo "bench: results in $OUT_DIR"
