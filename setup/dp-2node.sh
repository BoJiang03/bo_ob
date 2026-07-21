#!/usr/bin/env bash
# dp-2node.sh — 2-node data-parallel deployment of a MoE model: one DP rank per node,
# each rank bound to a LOCAL LMCache server.
#
# Topology (why one lmcache server per node): LMCache MP mode shares KV tensors with
# vLLM via CUDA IPC, which only works within one machine — a vLLM on another node
# cannot attach to this node's server. Cross-node KV sharing is LMCache's distributed
# tier (P2P / coordinator), separate from MP mode.
#
#   rtx-1 (head, owns the API):   ./dp-2node.sh start head
#   rtx-2 (secondary, headless):  ./dp-2node.sh start node2
#
# The head runs vLLM's internal DP load balancer: requests to :$VLLM_PORT are routed
# across both ranks. --enable-expert-parallel shards the MoE experts over the DP
# ranks (wide-EP) — the reason DP deployments exist for MoE models. Both nodes must
# use the same model, chunk-size, and PYTHONHASHSEED=0 (from _common.sh).
#
# Single-box simulation also works: run both roles on one machine with
# DP_ADDRESS=127.0.0.1 and a different VLLM_GPU per role.
#
# Usage:
#   ./dp-2node.sh {start|stop|restart|status} <head|node2>
#   ./dp-2node.sh logs <head|node2> [-f]
#
# Env config:
#   DP_ADDRESS   head node's reachable IP (default 172.16.176.27 = rtx-1 LAN)
#   DP_RPC_PORT  DP rendezvous port on the head (default 13345)
#   DP_MODEL     model name (default Qwen3-30B-A3B); weights auto-located in
#                /dev/shm/models or /data1/bo/models, or set MODEL_PATH
#   VLLM_GPU     GPU index for this node's rank (default 7)
#   NCCL_IFACE   NIC for NCCL/GLOO cross-node traffic (default enp41s0f0np0);
#                without it NCCL may pick docker0 and hang
#   VLLM_PORT, GPU_MEM_UTIL, LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV,
#   VLLM_START_TIMEOUT

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-7}
# DP engines re-derive their device from the local DP rank and stomp any
# CUDA_VISIBLE_DEVICES pin — rank 0 then always grabs physical GPU 0, which on a
# shared box is usually someone else's. --device-ids (below) is vLLM's supported
# way to pin DP ranks to physical GPUs; CUDA_VISIBLE_DEVICES must stay unset or
# the physical id in --device-ids can no longer be resolved.
unset CUDA_VISIBLE_DEVICES

MODEL_NAME=${DP_MODEL:-Qwen3-30B-A3B}
if [[ -z ${MODEL_PATH:-} ]]; then
    for d in /dev/shm/models /data1/bo/models; do
        if [[ -e "$d/$MODEL_NAME" ]]; then MODEL_PATH="$d/$MODEL_NAME"; break; fi
    done
fi
: "${MODEL_PATH:?no local weights found for $MODEL_NAME — set MODEL_PATH}"

DP_ADDRESS=${DP_ADDRESS:-172.16.176.27}
DP_RPC_PORT=${DP_RPC_PORT:-13345}
VLLM_PORT=${VLLM_PORT:-8100}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-900}
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-256}   # Qwen3 MoE recipe uses the lmcache default

# cross-node NCCL/GLOO must use the LAN NIC, not docker0/lo
export NCCL_SOCKET_IFNAME=${NCCL_IFACE:-enp41s0f0np0}
export GLOO_SOCKET_IFNAME=${NCCL_IFACE:-enp41s0f0np0}

# how long the head waits at the rendezvous for the other rank (vLLM default is
# 900s, after which the head half-dies while still listening — a trap). 4h lets
# a pre-launched head stand by until a GPU window opens on the other node.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-14400}
# ...and the torch TCPStore rendezvous has its OWN 1800s default ("Timed out
# ... 1/2 clients joined") -- raised via the two --*distributed-timeout-seconds
# args in DP_COMMON_ARGS below.
DP_WAIT_SECONDS=${DP_WAIT_SECONDS:-14400}

KV_CFG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
DP_COMMON_ARGS=(
    --served-model-name "$MODEL_NAME" "$MODEL_PATH"
    --device-ids "$GPU"
    --distributed-timeout-seconds "$DP_WAIT_SECONDS"
    --cpu-distributed-timeout-seconds "$DP_WAIT_SECONDS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable-expert-parallel
    --data-parallel-size 2
    --data-parallel-size-local 1
    --data-parallel-address "$DP_ADDRESS"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --kv-transfer-config "$KV_CFG"
)

dp_start() { # role
    local role=$1
    local pf=$RUN_DIR/dp-$role.pid
    local lf=$LOG_DIR/dp-$role.log
    if alive "$pf"; then
        echo "dp-$role: already running (pid $(cat "$pf"))"
        return 0
    fi
    ensure_server || return 1
    gpu_mem_warn 40000 "DP rank (mem-util $GPU_MEM_UTIL) may OOM on top of it"

    if [[ $role == head ]]; then
        if port_open "$VLLM_PORT"; then
            echo "ERROR: port $VLLM_PORT already in use — not starting" >&2
            return 1
        fi
        echo "dp-head: starting rank 0 (GPU $GPU, api :$VLLM_PORT, rendezvous $DP_ADDRESS:$DP_RPC_PORT)..."
        VLLM_SERVER_DEV_MODE=1 setsid nohup \
            vllm serve "$MODEL_PATH" --port "$VLLM_PORT" \
            "${DP_COMMON_ARGS[@]}" \
            > "$lf" 2>&1 < /dev/null &
        echo $! > "$pf"
        echo "dp-head: waiting for http://localhost:$VLLM_PORT/v1/models (up to ${VLLM_START_TIMEOUT}s)..."
        echo "         NOTE: the API only comes up after node2 joins the rendezvous."
        local rc=0
        wait_http "http://localhost:$VLLM_PORT/v1/models" "$VLLM_START_TIMEOUT" "$pf" || rc=$?
        if (( rc == 0 )); then
            echo "dp-head: up (pid $(cat "$pf"), log $lf)"
        elif (( rc == 2 )); then
            echo "ERROR: head died during startup; last log lines:" >&2
            tail -n 30 "$lf" >&2
            rm -f "$pf"
            return 1
        else
            echo "ERROR: head not healthy after ${VLLM_START_TIMEOUT}s (node2 never joined?); check $lf" >&2
            return 1
        fi
    else
        echo "dp-node2: starting rank 1 (GPU $GPU, headless, rendezvous $DP_ADDRESS:$DP_RPC_PORT)..."
        setsid nohup \
            vllm serve "$MODEL_PATH" --headless --data-parallel-start-rank 1 \
            "${DP_COMMON_ARGS[@]}" \
            > "$lf" 2>&1 < /dev/null &
        echo $! > "$pf"
        # headless rank has no API; consider it up if it survives early init
        local i=0
        while (( i < 30 )); do
            if ! alive "$pf"; then
                echo "ERROR: node2 died during startup; last log lines:" >&2
                tail -n 30 "$lf" >&2
                rm -f "$pf"
                return 1
            fi
            sleep 2; i=$((i + 2))
        done
        echo "dp-node2: running (pid $(cat "$pf"), log $lf) — rendezvous/model load continue; watch: $0 logs node2 -f"
    fi
}

dp_status() { # role
    local role=$1
    local pf=$RUN_DIR/dp-$role.pid
    if alive "$pf"; then
        echo "dp-$role: running (pid $(cat "$pf"))"
    else
        echo "dp-$role: stopped"
    fi
    if [[ $role == head ]]; then
        if curl -sf -o /dev/null "http://localhost:$VLLM_PORT/v1/models"; then
            echo "api: healthy at http://localhost:$VLLM_PORT/v1 (internal DP load balancer)"
        else
            echo "api: not responding on port $VLLM_PORT"
        fi
    fi
    if port_open "$LMCACHE_PORT"; then
        local chunk
        chunk=$(server_chunk_size)
        echo "local lmcache server: reachable on port $LMCACHE_PORT${chunk:+ (chunk-size $chunk; this setup needs $NEED_CHUNK)}"
    else
        echo "local lmcache server: NOT running ($COMMON_DIR/lmcache-ctl.sh start)"
    fi
    show_gpu
}

ROLE=${2:-}
case "${1:-}:$ROLE" in
    start:head|start:node2)     dp_start "$ROLE" ;;
    stop:head|stop:node2)       stop_one "dp-$ROLE" "$RUN_DIR/dp-$ROLE.pid" ;;
    restart:head|restart:node2) stop_one "dp-$ROLE" "$RUN_DIR/dp-$ROLE.pid"; dp_start "$ROLE" ;;
    status:head|status:node2)   dp_status "$ROLE" ;;
    logs:head|logs:node2)
        if [[ ${3:-} == "-f" ]]; then tail -f "$LOG_DIR/dp-$ROLE.log"; else tail -n 50 "$LOG_DIR/dp-$ROLE.log"; fi ;;
    *) print_usage; exit 1 ;;
esac
