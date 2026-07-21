#!/usr/bin/env bash
# p2p-demo.sh — cross-node P2P KV-cache sharing demo (homework Part 2, "Run P2P KV cache
# sharing across two physical nodes; create a case to show the benefit").
#
# Topology (two peers, each a full vLLM + local MP server, both P2P-enabled and
# registered to ONE coordinator):
#
#     coordinator (rtx-024:9300)  <-- HTTP discovery/heartbeat, never touches KV
#          ^                ^
#     node A (rtx-024)   node B (rtx-026)
#     vLLM :8002         vLLM :8002
#     MP server :5558    MP server :5558     <-- P2P: on local L1 miss, RDMA-read the
#     p2p :9400          p2p :9400               prefix straight from the peer's L1 (NIXL)
#
# The benefit case: warm a long prefix on node A only; then send the SAME prefix to node
# B.  B's local L1 misses, P2P finds it on A via the coordinator, and the transfer channel
# RDMA-reads it from A's L1 — so B skips prefill.  Compared against a never-seen prefix on
# B (cold recompute), B's TTFT for the shared prefix should be dramatically lower.
#
# ############################################################################
# ## PREREQUISITE (see ./check-rdma.sh):  NIXL runs over RDMA (RoCE here, mlx5_0).
# ## Both nodes have an ACTIVE RoCE link, BUT the `nixl` python package must be
# ## installed in BOTH venvs.  rtx-024 has nixl 1.3.1; rtx-026 does NOT yet:
# ##     ssh rtx-026 'uv pip install --python /home/bo/lmcache/.venv/bin/python nixl==1.3.1'
# ## Until then this demo cannot transfer.  UNTESTED end-to-end pending that install.
# ############################################################################
#
# Usage (run ON the relevant node):
#   # on rtx-024:
#   ./p2p-demo.sh coordinator            # start the coordinator (once, on node A)
#   ./p2p-demo.sh start A                # node A: P2P MP server + vLLM
#   # on rtx-026:
#   COORD_IP=172.16.176.27 ./p2p-demo.sh start B     # node B: P2P MP server + vLLM
#   # then, from node A:
#   ./p2p-demo.sh case                   # run the cross-node benefit scenario
#   ./p2p-demo.sh status <A|B> ; ./p2p-demo.sh stop <A|B|coordinator> ; ./p2p-demo.sh logs ...
#
# Env: COORD_IP (default 172.16.176.27=rtx-024), COORD_PORT(9300), PEER_B_IP(172.16.176.28),
#      MODEL, GPU, L1_SIZE_GB, GPU_MEM_UTIL, P2P_PORT(9400), P2P_LMCACHE_PORT(5558),
#      P2P_HTTP_PORT(8082), P2P_VLLM_PORT(8002), RDMA_NIC(enp41s0f0np0)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

COORD_IP=${COORD_IP:-172.16.176.27}
COORD_PORT=${COORD_PORT:-9300}
PEER_B_IP=${PEER_B_IP:-172.16.176.28}
RDMA_NIC=${RDMA_NIC:-enp41s0f0np0}
MODEL=${MODEL:-Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL}
L1_SIZE_GB=${L1_SIZE_GB:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
CHUNK=${LMCACHE_CHUNK_SIZE:-16}
P2P_PORT=${P2P_PORT:-9400}
P2P_LMCACHE_PORT=${P2P_LMCACHE_PORT:-5558}
P2P_HTTP_PORT=${P2P_HTTP_PORT:-8082}
P2P_VLLM_PORT=${P2P_VLLM_PORT:-8002}

# this node's IP on the RDMA NIC (what peers/coordinator reach us at)
SELF_IP=$(ip -4 addr show "$RDMA_NIC" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)

server_start() { # node-label
    local node=$1                                  # keep $node on its own `local` line:
    local pf="$RUN_DIR/p2p-server-$node.pid"        # a later RHS on the same line would
    local lf="$LOG_DIR/p2p-server-$node.log"        # expand $node BEFORE it is assigned
    if alive "$pf"; then echo "p2p-server-$node: already running (pid $(cat "$pf"))"; return 0; fi
    if port_open "$P2P_LMCACHE_PORT"; then echo "ERROR: port $P2P_LMCACHE_PORT busy" >&2; return 1; fi
    : "${SELF_IP:?could not resolve own IP on $RDMA_NIC}"
    # NOTE: bind 0.0.0.0, NOT localhost. Peers reach each other at ip:mq_port taken from the
    # coordinator record, so a localhost-bound MQ socket makes every P2P lookup RPC time out
    # ("P2P lookup submit to tcp://<ip>:<port> timed out; treating as a miss") -> silent 0 hits.
    echo "p2p-server-$node: starting MP server (l1 ${L1_SIZE_GB}GB, advertise $SELF_IP:$P2P_PORT, coord http://$COORD_IP:$COORD_PORT)"
    setsid nohup lmcache server \
        --host 0.0.0.0 --port "$P2P_LMCACHE_PORT" --http-port "$P2P_HTTP_PORT" \
        --l1-size-gb "$L1_SIZE_GB" --eviction-policy LRU --chunk-size "$CHUNK" \
        --l1-align-bytes 65536 \
        --p2p-advertise-url "$SELF_IP:$P2P_PORT" \
        --p2p-transfer-engine nixl \
        --coordinator-url "http://$COORD_IP:$COORD_PORT" \
        --coordinator-advertise-ip "$SELF_IP" \
        > "$lf" 2>&1 < /dev/null &
    echo $! > "$pf"
    if wait_port "$P2P_LMCACHE_PORT" 120; then
        sleep 3
        alive "$pf" || { echo "ERROR: p2p server died after opening port; last log:" >&2; tail -n 15 "$lf" >&2; rm -f "$pf"; return 1; }
        echo "p2p-server-$node: up (pid $(cat "$pf"), log $lf)"
    else
        echo "ERROR: p2p server didn't open $P2P_LMCACHE_PORT in 120s" >&2; tail -n 20 "$lf" >&2; return 1
    fi
}

vllm_start() { # node-label
    local node=$1                                  # keep $node on its own `local` line:
    local pf="$RUN_DIR/p2p-vllm-$node.pid"        # a later RHS on the same line would
    local lf="$LOG_DIR/p2p-vllm-$node.log"        # expand $node BEFORE it is assigned
    if alive "$pf"; then echo "p2p-vllm-$node: already running (pid $(cat "$pf"))"; return 0; fi
    if port_open "$P2P_VLLM_PORT"; then echo "ERROR: port $P2P_VLLM_PORT busy" >&2; return 1; fi
    export CUDA_VISIBLE_DEVICES=$GPU
    local kv="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":$P2P_LMCACHE_PORT,\"kv_load_failure_policy\":\"recompute\"}}"
    echo "p2p-vllm-$node: starting vLLM (GPU $GPU, api :$P2P_VLLM_PORT, server :$P2P_LMCACHE_PORT)"
    VLLM_SERVER_DEV_MODE=1 setsid nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL" "$MODEL_PATH" --port "$P2P_VLLM_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --kv-transfer-config "$kv" \
        > "$lf" 2>&1 < /dev/null &
    echo $! > "$pf"
    echo "p2p-vllm-$node: waiting for API (up to 600s)..."
    local rc=0; wait_http "http://localhost:$P2P_VLLM_PORT/v1/models" 600 "$pf" || rc=$?
    (( rc == 0 )) && echo "p2p-vllm-$node: up (pid $(cat "$pf"), log $lf)" || { echo "ERROR: vLLM not healthy; check $lf" >&2; return 1; }
}

coordinator_start() {
    local pf="$RUN_DIR/p2p-coordinator.pid" lf="$LOG_DIR/p2p-coordinator.log"
    if alive "$pf"; then echo "coordinator: already running (pid $(cat "$pf"))"; return 0; fi
    if port_open "$COORD_PORT"; then echo "ERROR: port $COORD_PORT busy" >&2; return 1; fi
    echo "coordinator: starting on 0.0.0.0:$COORD_PORT"
    setsid nohup lmcache coordinator --host 0.0.0.0 --port "$COORD_PORT" > "$lf" 2>&1 < /dev/null &
    echo $! > "$pf"
    if wait_port "$COORD_PORT" 30; then echo "coordinator: up (pid $(cat "$pf"), log $lf)"; else
        echo "ERROR: coordinator didn't open $COORD_PORT" >&2; tail -n 15 "$lf" >&2; return 1; fi
}

run_case() {
    echo "=== P2P benefit case: warm on A ($COORD_IP:$P2P_VLLM_PORT), read on B ($PEER_B_IP:$P2P_VLLM_PORT)"
    echo "    fleet members: $(curl -sf "http://$COORD_IP:$COORD_PORT/instances" 2>/dev/null || echo '(coordinator unreachable)')"
    OUT="$HERE/bench-results/p2p/case-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
    python3 "$HERE/p2p_case.py" \
        --node-a "http://$COORD_IP:$P2P_VLLM_PORT" \
        --node-b "http://$PEER_B_IP:$P2P_VLLM_PORT" \
        --model "$MODEL" --out "$OUT/case.json" 2>&1 | tee "$OUT/case.log"
    echo "=== results in $OUT"
}

status_node() { # node
    local node=$1
    for k in server vllm; do
        local pf="$RUN_DIR/p2p-$k-$node.pid"
        alive "$pf" && echo "p2p-$k-$node: running (pid $(cat "$pf"))" || echo "p2p-$k-$node: stopped"
    done
    curl -sf "http://localhost:$P2P_HTTP_PORT/status" 2>/dev/null | head -c 400 || true; echo
}

case "${1:-}:${2:-}" in
    coordinator:*) coordinator_start ;;
    start:A|start:B)   server_start "$2" && vllm_start "$2" ;;
    server:A|server:B) server_start "$2" ;;
    vllm:A|vllm:B)     vllm_start "$2" ;;
    case:*)            run_case ;;
    status:A|status:B) status_node "$2" ;;
    stop:coordinator)  stop_one p2p-coordinator "$RUN_DIR/p2p-coordinator.pid" ;;
    stop:A|stop:B)     stop_one "p2p-vllm-$2" "$RUN_DIR/p2p-vllm-$2.pid"; stop_one "p2p-server-$2" "$RUN_DIR/p2p-server-$2.pid" ;;
    logs:*)  f="$LOG_DIR/p2p-${2:-coordinator}.log"; [[ ${3:-} == -f ]] && tail -f "$f" || tail -n 50 "$f" ;;
    *) print_usage; exit 1 ;;
esac
