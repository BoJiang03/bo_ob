#!/usr/bin/env bash
# lmcache-ctl.sh — start / stop / manage the shared LMCache MP server on this node.
#
# vLLM instances are managed separately by the per-model scripts (e.g. ./Qwen3-8B.sh),
# which connect to this server and auto-start it when needed — passing the chunk-size
# their model requires. The running server's config is recorded in run/lmcache-server.conf
# so model scripts can refuse to attach to a server with a mismatched chunk-size.
# Stop the server only when no instance is using it — its cache (L1 CPU memory) dies with it.
#
# Usage:
#   ./lmcache-ctl.sh {start|stop|restart|status|logs [-f]}
#
# GPU visibility: the MP server resolves the device UUIDs that vLLM instances send
# when registering their KV caches — it MUST be able to see every GPU any attached
# vLLM instance runs on, or registration dies with "Device UUID ... not found".
# Its own idle GPU footprint is small (~0.5GB), so the default is to leave it
# unrestricted (sees all GPUs). Set LMCACHE_GPU only if you really want to pin it.
#
# Config via env, e.g.: CHUNK_SIZE=528 L1_SIZE_GB=40 ./lmcache-ctl.sh start
#   LMCACHE_GPU (default: all GPUs visible), LMCACHE_PORT, L1_SIZE_GB, CHUNK_SIZE, VENV
#
# l2_support/ COPY of setup/lmcache-ctl.sh, with two differences from setup:
#   - isolated ports (LMCACHE_PORT 5557 via _common.sh, HTTP 8081) so it never
#     collides with a setup/ server.
#   - L2_ADAPTER env: an L2 backend JSON (e.g. '{"type":"fs","base_path":"/x"}') that,
#     when set, is passed as --l2-adapter to enable the persistent L2 tier. Empty =
#     L1-only, identical to setup/ behavior. This is the whole point of l2_support.
#
# e.g.:  L1_SIZE_GB=4 L2_ADAPTER='{"type":"fs","base_path":"/data1/bo/l2cache"}' ./lmcache-ctl.sh start

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${LMCACHE_GPU:-}            # empty = unrestricted (server sees all GPUs)
if [[ -n $GPU ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU
else
    unset CUDA_VISIBLE_DEVICES
    GPU=all
fi
L1_SIZE_GB=${L1_SIZE_GB:-20}    # CPU pinned-memory cache size
CHUNK_SIZE=${CHUNK_SIZE:-16}    # must match what the attached models need (dense: free choice;
                                # mamba/linear-attention models: must equal vLLM's unified block size)
HTTP_PORT=${LMCACHE_HTTP_PORT:-8081}  # mgmt/metrics HTTP port (l2_support default; setup uses
                                      # 8080). If taken by a foreign process the whole server dies
                                      # seconds after "up" (bind Errno 98 kills it AFTER the ZMQ
                                      # port already passed the check)
L2_ADAPTER=${L2_ADAPTER:-}            # optional L2 backend JSON; empty = L1-only (setup behavior)
PIDFILE="$RUN_DIR/lmcache-server.pid"
LOGFILE="$LOG_DIR/lmcache-server.log"

start() {
    if alive "$PIDFILE"; then
        echo "lmcache server: already running (pid $(cat "$PIDFILE"), chunk-size $(server_chunk_size))"
        return 0
    fi
    if port_open "$LMCACHE_PORT"; then
        echo "ERROR: port $LMCACHE_PORT already in use by another process — not starting" >&2
        return 1
    fi
    local l2_args=()
    [[ -n $L2_ADAPTER ]] && l2_args=(--l2-adapter "$L2_ADAPTER")
    echo "lmcache server: starting on port $LMCACHE_PORT (GPU $GPU, L1 ${L1_SIZE_GB}GB, chunk $CHUNK_SIZE${L2_ADAPTER:+, L2=$L2_ADAPTER})..."
    setsid nohup lmcache server \
        --host localhost --port "$LMCACHE_PORT" --http-port "$HTTP_PORT" \
        --l1-size-gb "$L1_SIZE_GB" --eviction-policy LRU --chunk-size "$CHUNK_SIZE" \
        "${l2_args[@]}" \
        > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    if wait_port "$LMCACHE_PORT" 120; then
        sleep 3   # the HTTP mgmt server binds after the ZMQ port; a bind failure kills the process
        if ! alive "$PIDFILE"; then
            echo "ERROR: lmcache server died right after opening port $LMCACHE_PORT (HTTP port $HTTP_PORT taken?); last log lines:" >&2
            tail -n 10 "$LOGFILE" >&2
            rm -f "$PIDFILE"
            return 1
        fi
        {
            echo "CHUNK_SIZE=$CHUNK_SIZE"
            echo "L1_SIZE_GB=$L1_SIZE_GB"
            echo "GPU=$GPU"
            echo "PORT=$LMCACHE_PORT"
            echo "L2_ADAPTER=$L2_ADAPTER"
        } > "$SERVER_CONF"
        echo "lmcache server: up (pid $(cat "$PIDFILE"), log $LOGFILE)"
    else
        echo "ERROR: lmcache server did not open port $LMCACHE_PORT in 120s; last log lines:" >&2
        tail -n 20 "$LOGFILE" >&2
        return 1
    fi
}

status() {
    if alive "$PIDFILE"; then
        local pid state
        pid=$(cat "$PIDFILE")
        port_open "$LMCACHE_PORT" && state="port $LMCACHE_PORT open" || state="port $LMCACHE_PORT NOT responding"
        echo "lmcache server: running (pid $pid, $state, chunk-size $(server_chunk_size), L1 $(sed -n 's/^L1_SIZE_GB=//p' "$SERVER_CONF" 2>/dev/null)GB)"
    elif port_open "$LMCACHE_PORT"; then
        echo "lmcache server: no pidfile, but port $LMCACHE_PORT is in use (foreign process?)"
    else
        echo "lmcache server: stopped"
    fi
    show_gpu
}

case ${1:-} in
    start)   start ;;
    stop)    stop_one lmcache-server "$PIDFILE"; rm -f "$SERVER_CONF" ;;
    restart) stop_one lmcache-server "$PIDFILE"; rm -f "$SERVER_CONF"; start ;;
    status)  status ;;
    logs)    if [[ ${2:-} == "-f" ]]; then tail -f "$LOGFILE"; else tail -n 50 "$LOGFILE"; fi ;;
    *)       print_usage; exit 1 ;;
esac
