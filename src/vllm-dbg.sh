#!/usr/bin/env bash
# vllm-dbg.sh — vLLM instance for single-stepping LMCache in PyCharm (dev env).
#
# Starts vLLM (Qwen3-8B) from the /data1/bo/dev venv (editable vllm + lmcache), bound to
# an LMCache MP server on port 5560 — WITHOUT starting that server (unlike setup/'s
# scripts, there is deliberately no ensure_server). You start the server yourself, under
# the PyCharm debugger, and step through LMCache line by line:
#
#   ./vllm-dbg.sh servercmd    # prints the exact server command + PyCharm run-config fields
#
# Start order is free, but NOT open-ended: both connector adapters block at init on a
# GET_CHUNK_SIZE request (vllm_multi_process_adapter.py:626/:1149) and raise
# ConnectionError("server unreachable") after lmcache.mp.mq_timeout — default 300s.
# ZMQ queues the request, so a server started within the window answers it and vLLM
# proceeds normally. This script raises the window to MQ_TIMEOUT (default 3600s) so a
# leisurely PyCharm setup can't kill vLLM. Server-first is still the fastest bring-up;
# vllm-first lets you breakpoint the GET_CHUNK_SIZE / REGISTER_KV_CACHE handlers.
#
# Usage:
#   ./vllm-dbg.sh {start|stop|restart|status|logs [-f]|servercmd}
#
# Debug loop (with ./req.sh):
#   ./req.sh send          # 1st time: miss -> compute -> STORE into lmcache (D2H)
#   ./req.sh clear_cache   # wipe vLLM's own prefix cache
#   ./req.sh send          # same fixed prompt -> LOOKUP hit -> RETRIEVE (H2D)
#
# Ports (src/ block; clear of setup/ 5556/8000/8080, l2_support/ 5557/8001/8081,
# p2p-demo 5558, profiling/ 5559/8002/8082): ZMQ 5560, vLLM 8003, lmcache HTTP 8083.
#
# Env: GPU (default 7 — check nvidia-smi first, shared box!), MODEL_PATH, VENV,
#      VLLM_PORT, LMCACHE_PORT, GPU_MEM_UTIL, CHUNK_SIZE, L1_SIZE_GB, VLLM_START_TIMEOUT,
#      MQ_TIMEOUT (3600 — how long vLLM waits for the lmcache server before dying)

set -euo pipefail

VENV=${VENV:-/data1/bo/dev/venv}
LMCACHE_PORT=${LMCACHE_PORT:-5560}
LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT:-8083}
VLLM_PORT=${VLLM_PORT:-8003}
CHUNK_SIZE=${CHUNK_SIZE:-16}
L1_SIZE_GB=${L1_SIZE_GB:-20}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
MQ_TIMEOUT=${MQ_TIMEOUT:-3600}                   # lmcache.mp.mq_timeout: how long vLLM's
                                                 # connector init waits for the server
                                                 # before dying with ConnectionError
HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-300}    # lmcache.mp.heartbeat_interval: ping period
                                                 # AND ping timeout — a suspend-all breakpoint
                                                 # pause longer than this flips vLLM into
                                                 # degraded mode (skips lmcache until recovery)
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-1200}   # generous: start may block until you
                                                 # launch the lmcache server in PyCharm
MODEL_NAME=Qwen3-8B
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL_NAME}

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$COMMON_DIR/logs"; RUN_DIR="$COMMON_DIR/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"
PIDFILE="$RUN_DIR/vllm-dbg.pid"
LOGFILE="$LOG_DIR/vllm-dbg.log"
CONF="$RUN_DIR/vllm-dbg.conf"   # records the GPU the running instance was started on,
                                # so status/stop don't need GPU= repeated on every call
if [[ -z ${GPU:-} && -f $CONF ]]; then
    GPU=$(sed -n 's/^GPU=//p' "$CONF")
fi
GPU=${GPU:-7}

# MP mode requires identical hashing across ALL processes — the PyCharm-launched
# server must set PYTHONHASHSEED=0 too (servercmd reminds you).
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=$GPU

alive() { local p; p=$(cat "$PIDFILE" 2>/dev/null) || return 1; [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
print_usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; }

start() {
    if alive; then echo "vllm-dbg: already running (pid $(cat "$PIDFILE"))"; return 0; fi
    [[ -e $MODEL_PATH ]] || { echo "ERROR: model path $MODEL_PATH does not exist" >&2; return 1; }
    if port_open "$VLLM_PORT"; then
        echo "ERROR: port $VLLM_PORT already in use — not starting" >&2; return 1
    fi
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || true)
    if [[ -n ${used:-} ]] && (( used > 1000 )); then
        echo "WARN: GPU $GPU already has ${used} MiB in use — someone else's job? (shared box)"
    fi
    port_open "$LMCACHE_PORT" \
        && echo "vllm-dbg: lmcache server already reachable on port $LMCACHE_PORT" \
        || echo "vllm-dbg: no lmcache server on port $LMCACHE_PORT yet — vLLM will wait at KV registration; start it in PyCharm (./vllm-dbg.sh servercmd)"
    echo "vllm-dbg: starting $MODEL_NAME on port $VLLM_PORT (GPU $GPU, mem-util $GPU_MEM_UTIL, venv $VENV)..."
    # VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache (req.sh clear_cache)
    VLLM_SERVER_DEV_MODE=1 setsid nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" "$MODEL_PATH" --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":$MQ_TIMEOUT,\"lmcache.mp.heartbeat_interval\":$HEARTBEAT_INTERVAL}}" \
        > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "GPU=$GPU" > "$CONF"
    echo "vllm-dbg: waiting for http://localhost:$VLLM_PORT/v1/models (up to ${VLLM_START_TIMEOUT}s)..."
    local i=0
    while (( i < VLLM_START_TIMEOUT )); do
        curl -sf -o /dev/null "http://localhost:$VLLM_PORT/v1/models" && break
        alive || { echo "ERROR: vLLM died during startup; last log lines:" >&2
                   tail -n 30 "$LOGFILE" >&2; rm -f "$PIDFILE"; return 1; }
        sleep 2; i=$((i + 2))
    done
    if (( i >= VLLM_START_TIMEOUT )); then
        echo "ERROR: vLLM not healthy after ${VLLM_START_TIMEOUT}s (still waiting for the lmcache server?); check $LOGFILE" >&2
        return 1
    fi
    echo "vllm-dbg: up (pid $(cat "$PIDFILE"), log $LOGFILE)"
}

stop() {
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -z ${pid:-} ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "vllm-dbg: not running"; rm -f "$PIDFILE" "$CONF"; return 0
    fi
    echo "vllm-dbg: stopping (pid $pid)..."
    # setsid made $pid a group leader — negative pid kills the whole tree (EngineCore children)
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local i=0
    while (( i < 30 )) && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -0 "$pid" 2>/dev/null && { echo "vllm-dbg: force killing"; kill -KILL -- "-$pid" 2>/dev/null || true; }
    rm -f "$PIDFILE" "$CONF"
    echo "vllm-dbg: stopped (the PyCharm lmcache server is yours to stop)"
}

status() {
    if alive; then
        curl -sf -o /dev/null "http://localhost:$VLLM_PORT/v1/models" \
            && echo "vllm-dbg: running (pid $(cat "$PIDFILE")), API healthy at http://localhost:$VLLM_PORT/v1" \
            || echo "vllm-dbg: running (pid $(cat "$PIDFILE")), API not responding (starting, or waiting for the lmcache server)"
    else
        echo "vllm-dbg: stopped"
    fi
    port_open "$LMCACHE_PORT" \
        && echo "lmcache server: reachable on port $LMCACHE_PORT" \
        || echo "lmcache server: NOT running — ./vllm-dbg.sh servercmd for the PyCharm config"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU" 2>/dev/null | sed 's/^/GPU /'
}

servercmd() {
    cat <<EOF
# The lmcache server this vLLM binds to. Run it in a shell to sanity-check, or in
# PyCharm to single-step. PYTHONHASHSEED=0 is MANDATORY (MP mode hashes must match
# across processes — vllm-dbg.sh exports the same).

PYTHONHASHSEED=0 $VENV/bin/lmcache server \\
    --host localhost --port $LMCACHE_PORT --http-port $LMCACHE_HTTP_PORT \\
    --l1-size-gb $L1_SIZE_GB --eviction-policy LRU --chunk-size $CHUNK_SIZE

# PyCharm Run/Debug configuration:
#   interpreter        : $VENV/bin/python
#   script path        : $VENV/bin/lmcache
#   script parameters  : server --host localhost --port $LMCACHE_PORT --http-port $LMCACHE_HTTP_PORT --l1-size-gb $L1_SIZE_GB --eviction-policy LRU --chunk-size $CHUNK_SIZE
#   environment vars   : PYTHONHASHSEED=0
#   working directory  : /home/bo/onbording/src/lmcache   (NOT onbording/src itself — the
#                        vllm/lmcache dirs there shadow the installed packages via sys.path)
#   leave CUDA_VISIBLE_DEVICES unset — the server must see the GPU vLLM runs on (GPU $GPU)
#
# Stopping/restarting the server wipes its cache (L1 dies with the process).
EOF
}

case ${1:-} in
    start)     start ;;
    stop)      stop ;;
    restart)   stop; start ;;
    status)    status ;;
    logs)      if [[ ${2:-} == "-f" ]]; then tail -f "$LOGFILE"; else tail -n 50 "$LOGFILE"; fi ;;
    servercmd) servercmd ;;
    *)         print_usage; exit 1 ;;
esac
