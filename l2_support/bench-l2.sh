#!/usr/bin/env bash
# bench-l2.sh — compare several L2 adapters with `lmcache bench l2` (homework Part 2,
# task "Pick 3 different L2 adapters and compare their performances").
#
# `lmcache bench l2` is a STANDALONE micro-benchmark of one L2 adapter: no vLLM, no
# GPU, no lmcache server — it drives the adapter's Store / Lookup / Load APIs directly.
# So this whole script is zero-GPU and safe to run anytime; it never touches anyone's
# card.  See records/2026/07/21/4_l2_benchmarking.md for the tool write-up.
#
# It runs the SAME workload (num-keys / in-flight / data-size / rounds) against each
# adapter in turn, saving one JSON per adapter, then calls summarize-l2.py to print a
# side-by-side table.  Default trio:
#   fs         — pure-Python POSIX files on local disk
#   fs_native  — C++ worker-thread POSIX files on local disk (same disk, different impl)
#   resp       — Redis/Valkey over the network (in-memory store) — auto-started via docker
#
# Usage:
#   ./bench-l2.sh                       # fs, fs_native, resp with defaults
#   ADAPTERS="fs fs_native" ./bench-l2.sh
#   ADAPTERS="fs fs_fp8" ./bench-l2.sh  # fs vs fs+fp8-serde (disk footprint halved)
#   DATA_SIZE_KB=512 ROUNDS=10 ./bench-l2.sh
#
# Env config:
#   ADAPTERS            space list from {fs, fs_native, resp, fs_fp8, mock} (default "fs fs_native resp")
#   NUM_KEYS IN_FLIGHT DATA_SIZE_KB ROUNDS WARMUP_ROUNDS LOOKUP_MAX_HIT_RATE   (bench l2 knobs)
#   BENCH_DATA_DIR      local dir for fs/fs_native KV files (default /data1/bo/l2bench)
#   REDIS_PORT REDIS_CONTAINER REDIS_IMAGE   (docker Redis for the resp adapter)
#   OUT_DIR             results dir (default bench-results/l2-adapters/<timestamp>)
#   VENV                default /home/bo/lmcache/.venv
#
# Results (one <adapter>.json + <adapter>.log each, plus summary.md) land in OUT_DIR.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-/home/bo/lmcache/.venv}
export PATH="$VENV/bin:$PATH"
export LMCACHE_DISABLE_BANNER=1

# identical workload across every adapter
NUM_KEYS=${NUM_KEYS:-32}
IN_FLIGHT=${IN_FLIGHT:-4}
DATA_SIZE_KB=${DATA_SIZE_KB:-256}          # 256KB ~ one KV chunk
ROUNDS=${ROUNDS:-5}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-2}
LOOKUP_MAX_HIT_RATE=${LOOKUP_MAX_HIT_RATE:-1.0}   # 1.0 = measure the hit path (keys were just stored)

ADAPTERS=${ADAPTERS:-"fs fs_native resp"}
BENCH_DATA_DIR=${BENCH_DATA_DIR:-/data1/bo/l2bench}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_CONTAINER=${REDIS_CONTAINER:-lmcache-bench-redis}
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}

OUT_ROOT=${OUT_ROOT:-$HERE/bench-results/l2-adapters}
OUT_DIR=${OUT_DIR:-$OUT_ROOT/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT_DIR"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

WE_STARTED_REDIS=0
redis_up() {
    if port_open "$REDIS_PORT"; then
        echo "redis: port $REDIS_PORT already open — reusing existing instance"
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "redis: docker not found and nothing on :$REDIS_PORT — cannot run resp" >&2
        return 1
    fi
    docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
    echo "redis: starting docker $REDIS_CONTAINER ($REDIS_IMAGE) on :$REDIS_PORT ..."
    if ! docker run -d --rm --name "$REDIS_CONTAINER" -p "$REDIS_PORT:6379" "$REDIS_IMAGE" >/dev/null 2>&1; then
        echo "redis: docker run failed (image pull blocked? port taken?) — skipping resp" >&2
        return 1
    fi
    WE_STARTED_REDIS=1
    local i=0
    while (( i < 20 )); do port_open "$REDIS_PORT" && { echo "redis: up"; sleep 1; return 0; }; sleep 1; i=$((i+1)); done
    echo "redis: container did not open :$REDIS_PORT — skipping resp" >&2
    return 1
}
redis_down() { [[ $WE_STARTED_REDIS == 1 ]] && { echo "redis: stopping $REDIS_CONTAINER"; docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true; }; }
trap redis_down EXIT

run_adapter() { # name json
    local name=$1 json=$2                       # NOTE: keep name/json on their own `local`
    local out="$OUT_DIR/$name.json"             # line — a later RHS on the same `local` would
    local log="$OUT_DIR/$name.log"              # expand $name BEFORE it is assigned (set -u trap)
    echo ">>> [$name] $json"
    if lmcache bench l2 --l2-adapter "$json" \
            --num-keys "$NUM_KEYS" --in-flight "$IN_FLIGHT" --data-size-kb "$DATA_SIZE_KB" \
            --rounds "$ROUNDS" --warmup-rounds "$WARMUP_ROUNDS" \
            --lookup-max-hit-rate "$LOOKUP_MAX_HIT_RATE" \
            --format json --output "$out" > "$log" 2>&1; then
        echo "    -> $out"
    else
        echo "    !! [$name] bench failed — last log lines:" >&2
        tail -n 6 "$log" >&2
        return 1
    fi
}

fresh_dir() { rm -rf "$1"; mkdir -p "$1"; }   # clean disk backing so Store isn't skewed by leftovers

# record what we ran against
{
    echo "date        : $(date -Is)"
    echo "adapters    : $ADAPTERS"
    echo "num_keys    : $NUM_KEYS"
    echo "in_flight   : $IN_FLIGHT"
    echo "data_size_kb: $DATA_SIZE_KB"
    echo "rounds      : $ROUNDS (warmup $WARMUP_ROUNDS)"
    echo "lookup_hit  : $LOOKUP_MAX_HIT_RATE"
    echo "data_dir    : $BENCH_DATA_DIR"
    echo "-- disk backing $BENCH_DATA_DIR --"
    df -hT "$(dirname "$BENCH_DATA_DIR")" 2>/dev/null || true
    echo "lmcache     : $(python -c 'import lmcache; print(lmcache.__version__)' 2>/dev/null || echo '?')"
} | tee "$OUT_DIR/meta.txt"
echo

for a in $ADAPTERS; do
    case $a in
        fs)
            fresh_dir "$BENCH_DATA_DIR/fs"
            run_adapter fs "{\"type\":\"fs\",\"base_path\":\"$BENCH_DATA_DIR/fs\"}" || true ;;
        fs_native)
            fresh_dir "$BENCH_DATA_DIR/fs_native"
            run_adapter fs_native "{\"type\":\"fs_native\",\"base_path\":\"$BENCH_DATA_DIR/fs_native\",\"num_workers\":4}" || true ;;
        fs_fp8)  # fs with the fp8 serde layer — same disk, KV quantized bf16->fp8 (half the bytes)
            fresh_dir "$BENCH_DATA_DIR/fs_fp8"
            run_adapter fs_fp8 "{\"type\":\"fs\",\"base_path\":\"$BENCH_DATA_DIR/fs_fp8\",\"serde\":{\"type\":\"fp8\"}}" || true ;;
        resp)
            if redis_up; then
                run_adapter resp "{\"type\":\"resp\",\"host\":\"localhost\",\"port\":$REDIS_PORT}" || true
            else
                echo "    (skipping resp — no Redis)" ;
            fi ;;
        mock)  # zero-I/O control: measures the framework's own submit/notify overhead floor
            run_adapter mock "{\"type\":\"mock\"}" || true ;;
        *) echo "unknown adapter '$a' — skipping" >&2 ;;
    esac
done

echo
echo "===== summary ====="
python "$HERE/summarize-l2.py" "$OUT_DIR" | tee "$OUT_DIR/summary.md"
echo
echo "bench-l2: all results in $OUT_DIR"
