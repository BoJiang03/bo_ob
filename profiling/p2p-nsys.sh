#!/usr/bin/env bash
# p2p-nsys.sh — Part 3 task ③: nsys-measure P2P loading throughput across TWO REAL NODES.
#
# profiling/'s adaptation of l2_support/p2p-demo.sh (same topology: one coordinator on
# rtx-024, one "MP server + vLLM" peer per node; same case driver p2p_case.py, copied
# verbatim into this directory). Differences from the l2_support original:
#   * every subcommand is drivable FROM rtx-024 — node-B steps run over ssh, so one
#     terminal sees the whole experiment;
#   * both MP servers can start under `nsys launch -t cuda,nvtx` (NSYS=1). Collection
#     stays DEFERRED until `winopen`, keeping model load out of the window;
#   * p2p transfer ports default to 9402 (A) / 9401 (B): port 9400 on BOTH nodes is
#     held by a root docker-proxy (loopback-bound; NIXL on the NIC IP would coexist,
#     as experiment A proved, but a free number costs nothing);
#   * the MP server gets CUDA_VISIBLE_DEVICES pinned (the l2_support original let it
#     default; on 026 today that would land on someone's busy GPU0);
#   * REQUIRES the NVTX-annotated lmcache (fork PR #1, commit d5bdded7) in BOTH venvs,
#     else all NVTX tables come back empty. `install` builds v0.5.1+that-commit and
#     installs it on both nodes; `restore` puts stock lmcache==0.5.1 back.
#
# Orchestrated flow (ALL run from rtx-024, in this order):
#   ./p2p-nsys.sh check            # audit GPUs + ports on both nodes (read-only, no state)
#   ./p2p-nsys.sh install          # v0.5.1 + cherry-pick into both venvs (SLOW: CUDA build)
#   ./p2p-nsys.sh up               # syncs this dir to 026, then coordinator + A + B, NSYS=1
#   ./p2p-nsys.sh cap <tag>        # nsys windows on both nodes -> run P2P case -> close -> pull B rep
#   ./p2p-nsys.sh report <tag>     # NVTX range tables + CUDA memops + estimated GB/s per load
#   ./p2p-nsys.sh down             # stop everything on both nodes, verify GPUs idle
#   ./p2p-nsys.sh restore          # reinstall stock lmcache==0.5.1 in both venvs
#
# Per-node primitives (what the orchestrator itself calls, locally or over ssh):
#   coordinator | server <A|B> | vllm <A|B> | start <A|B> | stop <A|B|coordinator>
#   status <A|B> | logs <name> [-f] | sync | winopen <tag> <A|B> | winclose
#
# Env: A_GPU(0) B_GPU(1) PREFIX_TOKENS(6500) REPEATS(4) SEED_BASE(4242: MUST change
#      between caps on a live fleet, or B hits its own L1 and no P2P happens)
#      NIC_DEV(mlx5_0) NIC_METRICS(false: needs OFED) BYTES_PER_TOKEN(147456: Qwen3-8B
#      bf16 = 36 layers x 2(K,V) x 8 kv-heads x 128 head-dim x 2B) SESS(p2pprof)
#      NSYS_TRACE(cuda,nvtx) COORD_IP(172.16.176.27) COORD_PORT(9300)
#      PEER_B_IP(172.16.176.28) B_P2P_PORT(9401) MODEL(Qwen3-8B) L1_SIZE_GB(8)
#      GPU_MEM_UTIL(0.3) P2P_PORT(9402) P2P_LMCACHE_PORT(5558) P2P_HTTP_PORT(8082)
#      P2P_VLLM_PORT(8002) RDMA_NIC(enp41s0f0np0)
#      ANNOT_SRC_A(/data1/bo/LMCache-annot-051) ANNOT_SRC_B(/home/bo/LMCache-annot-051)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
SELF="$HERE/p2p-nsys.sh"

COORD_IP=${COORD_IP:-172.16.176.27}
COORD_PORT=${COORD_PORT:-9300}
PEER_B_IP=${PEER_B_IP:-172.16.176.28}
RDMA_NIC=${RDMA_NIC:-enp41s0f0np0}
MODEL=${MODEL:-Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL}
L1_SIZE_GB=${L1_SIZE_GB:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
CHUNK=${LMCACHE_CHUNK_SIZE:-16}
P2P_PORT=${P2P_PORT:-9402}
P2P_LMCACHE_PORT=${P2P_LMCACHE_PORT:-5558}
P2P_HTTP_PORT=${P2P_HTTP_PORT:-8082}
P2P_VLLM_PORT=${P2P_VLLM_PORT:-8002}

A_GPU=${A_GPU:-0}
B_GPU=${B_GPU:-1}
B_P2P_PORT=${B_P2P_PORT:-9401}
SESS=${SESS:-p2pprof}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
PREFIX_TOKENS=${PREFIX_TOKENS:-6500}
REPEATS=${REPEATS:-4}
BYTES_PER_TOKEN=${BYTES_PER_TOKEN:-147456}
ANNOT_SRC_A=${ANNOT_SRC_A:-/data1/bo/LMCache-annot-051}
ANNOT_SRC_B=${ANNOT_SRC_B:-/home/bo/LMCache-annot-051}
ANNOT_COMMIT=${ANNOT_COMMIT:-d5bdded7}
UV="$HOME/.local/bin/uv"

SELF_IP=$(ip -4 addr show "$RDMA_NIC" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)

on_b() { # run a command line in ~/onbording/profiling on node B, login shell (zsh-safe)
    ssh "$PEER_B_IP" "bash -lc 'cd \$HOME/onbording/profiling && $*'"
}

gpu_idle() { # gpu-index [remote]  -> ok if <5000 MiB used
    # Threshold guards against strangers' workloads (tens of GB), while our OWN
    # MP server already on the card (~700 MiB CUDA context, more under nsys) must
    # not trip the gate when vLLM starts second on the same GPU.
    local used
    if [[ ${2:-} == remote ]]; then
        used=$(ssh "$PEER_B_IP" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1" | tr -d ' \r')
    else
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    fi
    [[ -n $used && $used -lt 5000 ]]
}

# ---------------- per-node primitives (adapted from l2_support/p2p-demo.sh) ------------

server_start() { # node-label
    local node=$1
    local pf="$RUN_DIR/p2p-server-$node.pid"
    local lf="$LOG_DIR/p2p-server-$node.log"
    if alive "$pf"; then echo "p2p-server-$node: already running (pid $(cat "$pf"))"; return 0; fi
    if port_open "$P2P_LMCACHE_PORT"; then echo "ERROR: port $P2P_LMCACHE_PORT busy" >&2; return 1; fi
    : "${SELF_IP:?could not resolve own IP on $RDMA_NIC}"
    gpu_idle "$GPU" || { echo "ERROR: GPU $GPU is not idle; refusing to start" >&2; return 1; }
    # Optional nsys wrapper. Trace set is FIXED here at launch; winopen/-close only
    # open/close the collection window (nsys start rejects -t, 2025.5.2 rule).
    local launcher=()
    if [[ ${NSYS:-0} == 1 ]]; then
        nsys shutdown --session="$SESS" >/dev/null 2>&1 || true
        launcher=(nsys launch --session-new="$SESS" -t "$NSYS_TRACE")
        echo "p2p-server-$node: launching under nsys (session $SESS, trace $NSYS_TRACE, collection deferred)"
    fi
    echo "p2p-server-$node: starting MP server (GPU $GPU, l1 ${L1_SIZE_GB}GB, advertise $SELF_IP:$P2P_PORT, coord http://$COORD_IP:$COORD_PORT)"
    # bind 0.0.0.0, NOT localhost: peers dial the coordinator-recorded ip:mq_port, and a
    # loopback-bound MQ makes every P2P lookup RPC time out into a silent miss.
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup "${launcher[@]}" lmcache server \
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
    local node=$1
    local pf="$RUN_DIR/p2p-vllm-$node.pid"
    local lf="$LOG_DIR/p2p-vllm-$node.log"
    if alive "$pf"; then echo "p2p-vllm-$node: already running (pid $(cat "$pf"))"; return 0; fi
    if port_open "$P2P_VLLM_PORT"; then echo "ERROR: port $P2P_VLLM_PORT busy" >&2; return 1; fi
    gpu_idle "$GPU" || { echo "ERROR: GPU $GPU is not idle; refusing to start" >&2; return 1; }
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

status_node() { # node
    local node=$1
    for k in server vllm; do
        local pf="$RUN_DIR/p2p-$k-$node.pid"
        alive "$pf" && echo "p2p-$k-$node: running (pid $(cat "$pf"))" || echo "p2p-$k-$node: stopped"
    done
    curl -sf "http://localhost:$P2P_HTTP_PORT/status" 2>/dev/null | head -c 400 || true; echo
}

winopen() { # tag node  -- open the deferred collection window on THIS node
    local tag=${1:?winopen <tag> <A|B>} node=${2:?winopen <tag> <A|B>}
    # The RDMA wire transfer is invisible to CUDA tracing (NIC DMA between pinned
    # host buffers, no CUDA API). nsys' own fix, --nic-metrics, needs the NVIDIA
    # OFED driver (absent on these boxes, and installing one is a host-wide change
    # we don't make) -- so it stays opt-in, and cap instead samples the mlx5 sysfs
    # byte counters itself (nic-B.csv) for the direct wire-bandwidth signal.
    local nicflag=()
    [[ ${NIC_METRICS:-false} == true ]] && nicflag=(--nic-metrics=true)
    nsys start --session="$SESS" ${nicflag[@]+"${nicflag[@]}"} \
        -o "$RUN_DIR/p2p-$tag-$node.nsys-rep" -f true
    echo "nsys window open ($node) -> $RUN_DIR/p2p-$tag-$node.nsys-rep"
}

winclose() { nsys stop --session="$SESS"; echo "nsys window closed"; }

# ---------------- orchestrated subcommands (run from rtx-024) --------------------------

check() {
    local ok=0
    echo "== node A (local, GPU $A_GPU) =="
    gpu_idle "$A_GPU" && echo "  GPU $A_GPU: idle" || { echo "  GPU $A_GPU: BUSY"; ok=1; }
    for p in "$COORD_PORT" "$P2P_LMCACHE_PORT" "$P2P_VLLM_PORT" "$P2P_HTTP_PORT" "$P2P_PORT"; do
        port_open "$p" && { echo "  port $p: TAKEN"; ok=1; } || echo "  port $p: free"
    done
    echo "== node B ($PEER_B_IP, GPU $B_GPU) =="
    gpu_idle "$B_GPU" remote && echo "  GPU $B_GPU: idle" || { echo "  GPU $B_GPU: BUSY"; ok=1; }
    for p in "$P2P_LMCACHE_PORT" "$P2P_VLLM_PORT" "$P2P_HTTP_PORT" "$B_P2P_PORT"; do
        ssh "$PEER_B_IP" "bash -lc '(exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null'" \
            && { echo "  port $p: TAKEN"; ok=1; } || echo "  port $p: free"
    done
    echo "== annotated lmcache present? =="
    for side in A B; do
        local probe='python -c "import inspect; from lmcache import utils; print(\"annotated\" if \"iscoroutinefunction\" in inspect.getsource(utils._lmcache_nvtx_annotate) else \"PLAIN\")"'
        if [[ $side == A ]]; then r=$("$VENV/bin/python" -c 'import inspect; from lmcache import utils; print("annotated" if "iscoroutinefunction" in inspect.getsource(utils._lmcache_nvtx_annotate) else "PLAIN")')
        else r=$(ssh "$PEER_B_IP" "/home/bo/lmcache/.venv/bin/python -c 'import inspect; from lmcache import utils; print(\"annotated\" if \"iscoroutinefunction\" in inspect.getsource(utils._lmcache_nvtx_annotate) else \"PLAIN\")'"); fi
        echo "  $side: $r"; [[ $r == annotated ]] || ok=1
    done
    (( ok == 0 )) && echo "CHECK PASS" || { echo "CHECK FAIL (fix the lines above; 'install' fixes PLAIN)"; return 1; }
}

install() {
    if [[ ! -d $ANNOT_SRC_A ]]; then
        echo "install: cloning v0.5.1 + cherry-pick $ANNOT_COMMIT -> $ANNOT_SRC_A"
        git clone --no-hardlinks /data1/bo/LMCache "$ANNOT_SRC_A"
        git -C "$ANNOT_SRC_A" checkout -b p2p-annot-0.5.1 v0.5.1
        git -C "$ANNOT_SRC_A" cherry-pick "$ANNOT_COMMIT"
    fi
    echo "install: building into A venv $VENV (CUDA ext build, several minutes)"
    "$UV" pip install --python "$VENV/bin/python" "$ANNOT_SRC_A"
    echo "install: rsync source to B and build there"
    rsync -a --delete "$ANNOT_SRC_A/" "$PEER_B_IP:$ANNOT_SRC_B/"
    ssh "$PEER_B_IP" "\$HOME/.local/bin/uv pip install --python /home/bo/lmcache/.venv/bin/python $ANNOT_SRC_B"
    echo "install: done; run 'check' to verify both sides say 'annotated'"
}

restore() {
    echo "restore: stock lmcache==0.5.1 into both venvs"
    "$UV" pip install --python "$VENV/bin/python" lmcache==0.5.1
    ssh "$PEER_B_IP" "\$HOME/.local/bin/uv pip install --python /home/bo/lmcache/.venv/bin/python lmcache==0.5.1"
}

sync() {
    ssh "$PEER_B_IP" "mkdir -p \$HOME/onbording/profiling"
    rsync -a "$HERE/p2p-nsys.sh" "$HERE/p2p_case.py" "$HERE/_common.sh" "$PEER_B_IP:onbording/profiling/"
    echo "sync: profiling scripts -> $PEER_B_IP:~/onbording/profiling/"
}

up() {
    check
    sync
    coordinator_start
    GPU=$A_GPU NSYS=1 "$SELF" server A
    GPU=$A_GPU "$SELF" vllm A
    on_b "COORD_IP=$COORD_IP GPU=$B_GPU P2P_PORT=$B_P2P_PORT NSYS=1 ./p2p-nsys.sh server B"
    on_b "COORD_IP=$COORD_IP GPU=$B_GPU ./p2p-nsys.sh vllm B"
    # Final fleet verification: every failure mode above prints its ERROR into a lot
    # of scroll; this is the one loud line that says the whole fleet is really usable.
    local fleet_ok=1
    curl -sf "http://localhost:$P2P_VLLM_PORT/v1/models" >/dev/null || { echo "up: FAIL vLLM A not answering :$P2P_VLLM_PORT" >&2; fleet_ok=0; }
    curl -sf "http://$PEER_B_IP:$P2P_VLLM_PORT/v1/models" >/dev/null || { echo "up: FAIL vLLM B not answering $PEER_B_IP:$P2P_VLLM_PORT" >&2; fleet_ok=0; }
    local n
    n=$(curl -sf "http://$COORD_IP:$COORD_PORT/instances" 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["instances"]))' 2>/dev/null || echo 0)
    [[ $n == 2 ]] || { echo "up: FAIL coordinator sees $n instances, want 2" >&2; fleet_ok=0; }
    (( fleet_ok )) && echo "up: FLEET OK (vLLM A+B answering, 2 instances registered) — ready for cap" \
                   || { echo "up: FLEET INCOMPLETE — fix before cap" >&2; return 1; }
}

cap() {
    local tag=${1:?usage: p2p-nsys.sh cap <tag>}
    local out="$HERE/bench-results/p2p-nsys/$tag"; mkdir -p "$out"
    winopen "$tag" A
    # B's `nsys start` intermittently fails with "Configuring is not allowed in this
    # state" right after another session-state change; one short-delay retry clears
    # it. If B still fails, close A's window before bailing (no dangling windows).
    if ! on_b "./p2p-nsys.sh winopen $tag B"; then
        echo "cap: winopen B failed once, retrying in 5s..." >&2
        sleep 5
        on_b "./p2p-nsys.sh winopen $tag B" || { winclose || true; echo "ERROR: winopen B failed twice" >&2; return 1; }
    fi
    # Sample B's RDMA HCA byte counters (userspace sysfs, no OFED needed) at ~50Hz
    # for the direct wire-bandwidth signal. Counter unit is 4-byte words (IB spec).
    local nicf="$out/nic-B.csv" nicpid=
    ssh "$PEER_B_IP" 'while :; do echo "$(date +%s.%N) $(cat /sys/class/infiniband/'"${NIC_DEV:-mlx5_0}"'/ports/1/counters/port_rcv_data)"; sleep 0.02; done' > "$nicf" 2>/dev/null &
    nicpid=$!
    echo "cap: running P2P case (warm on A, read on B; $REPEATS prefixes x $PREFIX_TOKENS tokens)"
    local rc=0
    # SEED_BASE must CHANGE between caps against the same live fleet: prefixes from
    # an earlier run already live in B's L1, and a repeat would hit locally (0 P2P).
    python3 "$HERE/p2p_case.py" \
        --node-a "http://localhost:$P2P_VLLM_PORT" \
        --node-b "http://$PEER_B_IP:$P2P_VLLM_PORT" \
        --model "$MODEL" --prefix-tokens "$PREFIX_TOKENS" --repeats "$REPEATS" \
        --seed-base "${SEED_BASE:-4242}" \
        --out "$out/case.json" 2>&1 | tee "$out/case.log" || rc=$?
    # ALWAYS close both windows, even when the case blew up -- a dangling session
    # makes the next winopen fail with "session already started".
    [[ -n $nicpid ]] && { kill "$nicpid" 2>/dev/null || true; wait "$nicpid" 2>/dev/null || true; }
    winclose || true
    on_b "./p2p-nsys.sh winclose" || true
    (( rc == 0 )) || { echo "ERROR: case failed (rc=$rc); windows closed, reps are garbage" >&2; return "$rc"; }
    scp "$PEER_B_IP:onbording/profiling/run/p2p-$tag-B.nsys-rep" "$RUN_DIR/"
    echo "cap done: $RUN_DIR/p2p-$tag-{A,B}.nsys-rep + $out/case.json"
}

report() {
    local tag=${1:?usage: p2p-nsys.sh report <tag>}
    for side in A B; do
        local rep="$RUN_DIR/p2p-$tag-$side.nsys-rep"
        [[ -f $rep ]] || { echo "missing $rep" >&2; continue; }
        echo "################ side $side: NVTX start/end (async P2P: batched_get_non_blocking...) ################"
        nsys stats --force-export=true --report nvtx_startend_sum "$rep" 2>/dev/null
        echo "################ side $side: NVTX push/pop (sync: controller handlers, adapter) ################"
        nsys stats --force-export=true --report nvtx_pushpop_sum "$rep" 2>/dev/null
        echo "################ side $side: CUDA mem ops by time ################"
        nsys stats --force-export=true --report cuda_gpu_mem_time_sum "$rep" 2>/dev/null
    done
    # Direct wire bandwidth from the sysfs counter samples (if cap collected them):
    # groups consecutive >0.5 GB/s samples into bursts = individual P2P transfers.
    local nicf="$HERE/bench-results/p2p-nsys/$tag/nic-B.csv"
    [[ -s $nicf ]] && python3 - "$nicf" <<'NICEOF'
import sys
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 2]
samp = [(float(t), int(c) * 4) for t, c in rows]  # counter unit = 4-byte words
bursts, cur = [], None
for (t0, c0), (t1, c1) in zip(samp, samp[1:]):
    rate = (c1 - c0) / (t1 - t0)
    if rate > 0.5e9:
        cur = cur or [t0, c0, t1, c1]; cur[2:] = [t1, c1]
    elif cur:
        bursts.append(cur); cur = None
if cur: bursts.append(cur)
print("\n== RDMA wire bandwidth (B-side HCA rcv counter, 50Hz sysfs samples) ==")
for i, (t0, c0, t1, c1) in enumerate(bursts):
    gb, dur = (c1 - c0) / 2**30, t1 - t0
    print(f"   burst {i}: {gb:5.2f} GiB in {dur*1e3:7.1f} ms  -> {(c1-c0)/dur/1e9:5.2f} GB/s")
if not bursts: print("   (no burst above 0.5 GB/s found)")
NICEOF
    # Estimated per-load throughput: bytes(prefix) / avg duration of the B-side
    # batched_get_non_blocking range. Bytes are computed, not measured -- states its math.
    local brep="$RUN_DIR/p2p-$tag-B.nsys-rep"
    [[ -f $brep ]] && PREFIX_TOKENS="$PREFIX_TOKENS" BYTES_PER_TOKEN="$BYTES_PER_TOKEN" python3 - "$brep" <<'PYEOF'
import csv, io, os, subprocess, sys
rep = sys.argv[1]
try:
    out = subprocess.run(
        ["nsys", "stats", "--force-export=true", "--report", "nvtx_startend_sum",
         "--format", "csv", "--output", "-", rep],
        capture_output=True, text=True, timeout=300).stdout
    rows = [r for r in csv.DictReader(io.StringIO(out.split("\n\n")[-1].strip()))
            if "batched_get_non_blocking" in (r.get("Range") or "")]
    if not rows:
        raise RuntimeError("no batched_get_non_blocking range rows")
    r = rows[0]
    n = int(r["Instances"]); avg_ns = float(r["Avg (ns)"])
    tokens = int(os.environ["PREFIX_TOKENS"]); bpt = int(os.environ["BYTES_PER_TOKEN"])
    gbs = tokens * bpt / avg_ns  # bytes / ns == GB/s
    print(f"\n== estimated P2P load throughput (B side) ==")
    print(f"   {n} loads, avg range {avg_ns/1e6:.1f} ms; "
          f"assumed bytes/load = {tokens} tok x {bpt} B = {tokens*bpt/2**30:.2f} GiB")
    print(f"   => ~{gbs:.2f} GB/s per load (compare: exp-A RoCE 5-10 GB/s, H2D ceiling ~32 GB/s)")
except Exception as e:
    print(f"\n(throughput estimate skipped: {e}; do the math from the tables above)")
PYEOF
}

down() {
    on_b "./p2p-nsys.sh stop B" || true
    "$SELF" stop A || true
    "$SELF" stop coordinator || true
    nsys shutdown --session="$SESS" >/dev/null 2>&1 || true
    ssh "$PEER_B_IP" "bash -lc 'nsys shutdown --session=$SESS >/dev/null 2>&1'" || true
    echo "== GPU state after teardown =="
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$A_GPU"
    ssh "$PEER_B_IP" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $B_GPU"
}

case "${1:-}:${2:-}" in
    check:*)           check ;;
    install:*)         install ;;
    restore:*)         restore ;;
    sync:*)            sync ;;
    up:*)              up ;;
    cap:*)             shift; cap "$@" ;;
    report:*)          shift; report "$@" ;;
    down:*)            down ;;
    coordinator:*)     coordinator_start ;;
    start:A|start:B)   server_start "$2" && vllm_start "$2" ;;
    server:A|server:B) server_start "$2" ;;
    vllm:A|vllm:B)     vllm_start "$2" ;;
    status:A|status:B) status_node "$2" ;;
    stop:coordinator)  stop_one p2p-coordinator "$RUN_DIR/p2p-coordinator.pid" ;;
    stop:A|stop:B)     stop_one "p2p-vllm-$2" "$RUN_DIR/p2p-vllm-$2.pid"; stop_one "p2p-server-$2" "$RUN_DIR/p2p-server-$2.pid" ;;
    winopen:*)         shift; winopen "$@" ;;
    winclose:*)        winclose ;;
    logs:*)  f="$LOG_DIR/p2p-${2:-coordinator}.log"; [[ ${3:-} == -f ]] && tail -f "$f" || tail -n 50 "$f" ;;
    *) print_usage; exit 1 ;;
esac
