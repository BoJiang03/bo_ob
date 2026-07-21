#!/usr/bin/env bash
# lmcache-ctl.sh — start / stop / manage the shared LMCache MP server on this node.
#
# COPY of setup/lmcache-ctl.sh for Part 3 (profiling). Only the HTTP mgmt port
# default differs (8082 vs setup's 8080); the ZMQ port comes from _common.sh (5559).
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
# Config via env, e.g.: CHUNK_SIZE=528 L1_SIZE_GB=40 ./lmcache-ctl.sh start
#   LMCACHE_GPU (default: all GPUs visible), LMCACHE_PORT, L1_SIZE_GB, CHUNK_SIZE, VENV

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
CHUNK_SIZE=${CHUNK_SIZE:-16}    # must match what the attached models need
HTTP_PORT=${LMCACHE_HTTP_PORT:-8082}  # mgmt/metrics HTTP port (profiling block)
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
    # Optional nsys launcher (nsys.sh sets NSYS_PREFIX to "nsys launch ..." to profile
    # the SERVER — where the real KV H2D/D2H copy happens in MP mode). Empty = normal.
    local launcher=()
    [[ -n ${NSYS_PREFIX:-} ]] && read -r -a launcher <<< "$NSYS_PREFIX"
    echo "lmcache server: starting on port $LMCACHE_PORT (GPU $GPU, L1 ${L1_SIZE_GB}GB, chunk $CHUNK_SIZE)${launcher:+ [under: ${launcher[*]}]}..."
    setsid nohup "${launcher[@]}" lmcache server \
        --host localhost --port "$LMCACHE_PORT" --http-port "$HTTP_PORT" \
        --l1-size-gb "$L1_SIZE_GB" --eviction-policy LRU --chunk-size "$CHUNK_SIZE" \
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
