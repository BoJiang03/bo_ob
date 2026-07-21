# _common.sh — shared config, helpers, and the generic vLLM-instance engine.
# Sourced by lmcache-ctl.sh and the per-model scripts; not executable.
#
# A per-model script only needs to set its config variables (MODEL_NAME, MODEL_PATH,
# VLLM_PORT, GPU_MEM_UTIL, NEED_CHUNK, EXTRA_VLLM_ARGS...) and call `model_dispatch "$@"`.
#
# NOTE: this is the l2_support/ COPY of setup/_common.sh (Bo asked to copy rather than
# share, so the two parts never interfere). It defaults to ISOLATED ports/dirs — a
# separate LMCACHE_PORT (5557) plus its own logs/ and run/ under l2_support/ — so an
# l2_support server/vLLM can run alongside a setup/ one without clashing.

VENV=${VENV:-/home/bo/lmcache/.venv}
GPU=${GPU:-7}                       # global default GPU (shared box: 0-6 used by others);
                                    # each script may override per role, then pins CUDA_VISIBLE_DEVICES itself
LMCACHE_PORT=${LMCACHE_PORT:-5557}  # l2_support default (setup/ uses 5556) — isolation

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$COMMON_DIR/logs"
RUN_DIR="$COMMON_DIR/run"
SERVER_CONF="$RUN_DIR/lmcache-server.conf"   # written by lmcache-ctl.sh on start
mkdir -p "$LOG_DIR" "$RUN_DIR"

# MP mode requires identical hashing across all processes; ninja on PATH for flashinfer JIT
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONHASHSEED=0

alive() { # pidfile
    local pid
    pid=$(cat "$1" 2>/dev/null) || return 1
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

port_open() { # port
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

wait_port() { # port timeout_s
    local i=0
    while (( i < $2 )); do
        port_open "$1" && return 0
        sleep 1; i=$((i + 1))
    done
    return 1
}

wait_http() { # url timeout_s pidfile
    local i=0
    while (( i < $2 )); do
        curl -sf -o /dev/null "$1" && return 0
        alive "$3" || return 2   # process died while we waited
        sleep 2; i=$((i + 2))
    done
    return 1
}

gpu_mem_warn() { # threshold_mib message   ($GPU may be a comma list or "all"; checks the busiest one)
    local used sel=()
    [[ $GPU != all ]] && sel=(-i "$GPU")
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "${sel[@]}" 2>/dev/null \
           | sort -n | tail -1) || return 0
    [[ -n $used ]] || return 0
    if (( used > $1 )); then
        echo "WARN: GPU $GPU already has ${used} MiB in use — $2"
    fi
}

show_gpu() {
    local sel=()
    [[ $GPU != all ]] && sel=(-i "$GPU")
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader "${sel[@]}" 2>/dev/null \
        | sed 's/^/GPU /'
}

server_chunk_size() { # prints the running server's chunk size, empty if unknown
    sed -n 's/^CHUNK_SIZE=//p' "$SERVER_CONF" 2>/dev/null
}

stop_one() { # name pidfile
    local name=$1 pf=$2 pid
    pid=$(cat "$pf" 2>/dev/null || true)
    if [[ -z ${pid:-} ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$name: not running"
        rm -f "$pf"
        return 0
    fi
    echo "$name: stopping (pid $pid)..."
    # setsid made $pid a group leader — negative pid kills its whole tree
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local i=0
    while (( i < 30 )) && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$name: still alive after 30s, force killing"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
    echo "$name: stopped"
}

print_usage() { # prints the leading comment block of the executed script
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print } NR > 1 && !/^#/ { exit }' "$0"
}

# ---------- generic vLLM model-instance engine ----------

ensure_server() { # requires NEED_CHUNK; auto-starts or validates the lmcache server
    if port_open "$LMCACHE_PORT"; then
        local chunk
        chunk=$(server_chunk_size)
        if [[ -n $chunk && $chunk -ne $NEED_CHUNK ]]; then
            echo "ERROR: running lmcache server uses chunk-size $chunk but $MODEL_NAME needs $NEED_CHUNK." >&2
            echo "       Restart it with: CHUNK_SIZE=$NEED_CHUNK $COMMON_DIR/lmcache-ctl.sh restart  (wipes its cache)" >&2
            echo "       or point this instance at a second server via LMCACHE_PORT." >&2
            return 1
        elif [[ -z $chunk ]]; then
            echo "WARN: server on port $LMCACHE_PORT not started by lmcache-ctl.sh — cannot verify chunk-size (need $NEED_CHUNK)"
        fi
    else
        echo "$MODEL_NAME: lmcache server not up, starting it (chunk-size $NEED_CHUNK)..."
        CHUNK_SIZE=$NEED_CHUNK "$COMMON_DIR/lmcache-ctl.sh" start
    fi
}

model_start() {
    if alive "$PIDFILE"; then
        echo "$MODEL_NAME: already running (pid $(cat "$PIDFILE"))"
        return 0
    fi
    if [[ $MODEL_PATH == /* && ! -e $MODEL_PATH ]]; then
        echo "ERROR: model path $MODEL_PATH does not exist" >&2
        return 1
    fi
    local kv_args=()
    if [[ ${NO_LMCACHE:-0} == 1 ]]; then
        echo "$MODEL_NAME: NO_LMCACHE=1 — baseline mode, running WITHOUT LMCache"
    else
        ensure_server || return 1
        kv_args=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":$LMCACHE_PORT}}")
    fi
    if port_open "$VLLM_PORT"; then
        echo "ERROR: port $VLLM_PORT already in use by another process — not starting" >&2
        return 1
    fi
    gpu_mem_warn 55000 "vLLM (mem-util $GPU_MEM_UTIL) may OOM on top of it"
    echo "$MODEL_NAME: starting vLLM on port $VLLM_PORT (GPU $GPU, mem-util $GPU_MEM_UTIL)${kv_args:+, bound to lmcache on port $LMCACHE_PORT}..."
    # VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache (needed for cache benchmarks)
    # both names accepted by the API: the short name for humans, the path for
    # `lmcache bench engine` (the MP server registers models under their path)
    VLLM_SERVER_DEV_MODE=1 setsid nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" "$MODEL_PATH" --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        "${EXTRA_VLLM_ARGS[@]}" \
        "${kv_args[@]}" \
        > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "$MODEL_NAME: waiting for http://localhost:$VLLM_PORT/v1/models (up to ${VLLM_START_TIMEOUT}s)..."
    local rc=0
    wait_http "http://localhost:$VLLM_PORT/v1/models" "$VLLM_START_TIMEOUT" "$PIDFILE" || rc=$?
    if (( rc == 0 )); then
        echo "$MODEL_NAME: up (pid $(cat "$PIDFILE"), log $LOGFILE)"
    elif (( rc == 2 )); then
        echo "ERROR: vLLM process died during startup; last log lines:" >&2
        tail -n 30 "$LOGFILE" >&2
        rm -f "$PIDFILE"
        return 1
    else
        echo "ERROR: vLLM not healthy after ${VLLM_START_TIMEOUT}s; check $LOGFILE" >&2
        return 1
    fi
}

model_status() {
    if alive "$PIDFILE"; then
        local pid
        pid=$(cat "$PIDFILE")
        if curl -sf -o /dev/null "http://localhost:$VLLM_PORT/v1/models"; then
            echo "$MODEL_NAME: running (pid $pid), API healthy at http://localhost:$VLLM_PORT/v1"
        else
            echo "$MODEL_NAME: running (pid $pid), but API on port $VLLM_PORT not responding (still starting, or wedged)"
        fi
    elif port_open "$VLLM_PORT"; then
        echo "$MODEL_NAME: stopped (but port $VLLM_PORT is in use by another process)"
    else
        echo "$MODEL_NAME: stopped"
    fi
    if port_open "$LMCACHE_PORT"; then
        local chunk
        chunk=$(server_chunk_size)
        echo "lmcache server: reachable on port $LMCACHE_PORT${chunk:+ (chunk-size $chunk; this model needs $NEED_CHUNK)}"
    else
        echo "lmcache server: NOT running ($COMMON_DIR/lmcache-ctl.sh start)"
    fi
    show_gpu
}

model_dispatch() {
    PIDFILE="$RUN_DIR/$MODEL_NAME.pid"
    LOGFILE="$LOG_DIR/$MODEL_NAME.log"
    case ${1:-} in
        start)   model_start ;;
        stop)    stop_one "$MODEL_NAME" "$PIDFILE" ;;
        restart) stop_one "$MODEL_NAME" "$PIDFILE"; model_start ;;
        status)  model_status ;;
        logs)    if [[ ${2:-} == "-f" ]]; then tail -f "$LOGFILE"; else tail -n 50 "$LOGFILE"; fi ;;
        *)       print_usage; exit 1 ;;
    esac
}
