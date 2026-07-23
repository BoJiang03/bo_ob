#!/usr/bin/env bash
# copy-nsys.sh — Part 3 task ④: nsys-measure the H2D/D2H KV copy throughput, one script.
#
# Single-node sibling of p2p-nsys.sh (task ③), wrapping the same building blocks nsys.sh
# drives by hand: lmcache-ctl.sh + Qwen3-8B.sh bring the fleet up (the MP server under
# `nsys launch`, collection DEFERRED), workload.sh generates the store/reset/reload
# traffic. Differences from nsys.sh:
#   * full lifecycle: check (read-only audit) / up / cap / report / down — the fleet
#     stays up across caps; each cap only opens/closes an nsys collection window;
#   * report COMPUTES the answer instead of dumping raw `nsys stats` tables:
#     per-direction op-level GB/s, per-burst WALL-clock GB/s + copy-engine busy%,
#     stream layout, NVTX range shares — all from the exported sqlite;
#   * report also accepts legacy reps (run/<tag>.nsys-rep, e.g. `report srv_h2d_1`
#     re-analyzes the 07-21 capture) alongside its own run/copy-<tag>.nsys-rep.
# No install/restore here: stock lmcache 0.5.1 already carries NVTX annotations on the
# server-side copy path (LMCacheDrivenTransferModule.store/retrieve, allocator) — the
# task-② fork build is only needed for the async P2P path (p2p-nsys.sh).
#
# Orchestrated flow:
#   ./copy-nsys.sh check           # GPU + ports + nsys + venv audit (read-only)
#   ./copy-nsys.sh up              # lmcache server under nsys (deferred) + vLLM
#   ./copy-nsys.sh cap <tag>       # open window -> workload.sh -> close (repeatable)
#   ./copy-nsys.sh report <tag>    # computed H2D/D2H throughput analysis
#   ./copy-nsys.sh down            # stop vLLM + server, kill nsys session, show GPU
# Primitives: status | logs <server|vllm> [-f] | winopen <tag> | winclose
#
# Env: GPU(7, the profiling block 5559/8002/8082; VLLM_GPU/LMCACHE_GPU per role),
#      NSYS_TARGET(server: where the copy runs in MP mode | vllm: submit side only),
#      NSYS_TRACE(cuda,nvtx), SESS(copyprof), GAP_MS(50: memcpy gap splitting bursts),
#      MIN_BURST_MB(100: hide warmup dribbles in the burst table), plus all WL_*
#      workload knobs (WL_PROMPTS 16, WL_PREFIX_WORDS 1200, WL_ROUNDS 3, WL_SETTLE 3).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

SESS=${SESS:-copyprof}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_TARGET=${NSYS_TARGET:-server}
MODEL_SCRIPT="$HERE/Qwen3-8B.sh"
GAP_MS=${GAP_MS:-50}
MIN_BURST_MB=${MIN_BURST_MB:-100}

gpu_idle() { # gpu-index -> ok if <5000 MiB used (same threshold rationale as p2p-nsys.sh)
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [[ -n $used && $used -lt 5000 ]]
}

check() {
    local ok=0 gpu="${VLLM_GPU:-$GPU}"
    echo "== GPU $gpu =="
    gpu_idle "$gpu" && echo "  idle" || { echo "  BUSY ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu"))"; ok=1; }
    echo "== ports (must be free before up; taken-by-us is fine between up and down) =="
    for p in "$LMCACHE_PORT" 8002 8082; do
        port_open "$p" && echo "  port $p: TAKEN" || echo "  port $p: free"
    done
    echo "== tools =="
    echo "  nsys: $(command -v nsys) ($(nsys --version 2>/dev/null | head -1))"
    local lm
    lm=$("$VENV/bin/python" -c 'import lmcache, importlib.metadata as m; print(m.version("lmcache"))' 2>/dev/null) \
        && echo "  lmcache: $lm (venv $VENV)" || { echo "  lmcache: NOT importable in $VENV"; ok=1; }
    "$VENV/bin/python" -c 'from lmcache.utils import _lmcache_nvtx_annotate' 2>/dev/null \
        && echo "  NVTX annotate helper: present" || { echo "  NVTX annotate helper: MISSING"; ok=1; }
    (( ok == 0 )) && echo "CHECK PASS" || { echo "CHECK FAIL (fix the lines above)"; return 1; }
}

up() {
    local gpu="${VLLM_GPU:-$GPU}"
    gpu_idle "$gpu" || { echo "ERROR: GPU $gpu is not idle; refusing to start" >&2; return 1; }
    nsys shutdown --session="$SESS" >/dev/null 2>&1 || true
    local launch="nsys launch --session-new=$SESS -t $NSYS_TRACE"
    echo "up: target=$NSYS_TARGET under [$launch] (collection deferred until cap)"
    if [[ $NSYS_TARGET == server ]]; then
        NSYS_PREFIX="$launch" LMCACHE_GPU="$gpu" "$HERE/lmcache-ctl.sh" start
        VLLM_GPU="$gpu" "$MODEL_SCRIPT" start
    else
        LMCACHE_GPU="$gpu" "$HERE/lmcache-ctl.sh" start
        NSYS_PREFIX="$launch" VLLM_GPU="$gpu" "$MODEL_SCRIPT" start
    fi
    # One loud line that the whole stack is really usable (mirrors p2p-nsys.sh up).
    local fleet_ok=1
    curl -sf "http://localhost:8002/v1/models" >/dev/null || { echo "up: FAIL vLLM not answering :8002" >&2; fleet_ok=0; }
    port_open "$LMCACHE_PORT" || { echo "up: FAIL lmcache server ZMQ port $LMCACHE_PORT closed" >&2; fleet_ok=0; }
    nsys sessions list 2>/dev/null | grep -q "$SESS" || { echo "up: FAIL nsys session $SESS not live" >&2; fleet_ok=0; }
    (( fleet_ok )) && echo "up: FLEET OK (vLLM answering, server up, nsys session armed) — ready for cap" \
                   || { echo "up: FLEET INCOMPLETE — fix before cap" >&2; return 1; }
}

winopen() { # tag
    local tag=${1:?winopen <tag>}
    nsys start --session="$SESS" -o "$RUN_DIR/copy-$tag.nsys-rep" -f true
    echo "nsys window open -> $RUN_DIR/copy-$tag.nsys-rep"
}

winclose() { nsys stop --session="$SESS"; echo "nsys window closed"; }

cap() {
    local tag=${1:?usage: copy-nsys.sh cap <tag>}
    local out="$HERE/bench-results/copy-nsys/$tag"; mkdir -p "$out"
    winopen "$tag"
    echo "cap: driving workload.sh (store/reset/reload; WL_PROMPTS=${WL_PROMPTS:-16} WL_ROUNDS=${WL_ROUNDS:-3} WL_PREFIX_WORDS=${WL_PREFIX_WORDS:-1200})"
    local rc=0
    "$HERE/workload.sh" 2>&1 | tee "$out/workload.log" || rc=$?
    # ALWAYS close the window — a dangling session makes the next winopen fail.
    winclose || true
    (( rc == 0 )) || { echo "ERROR: workload failed (rc=$rc); window closed, rep is garbage" >&2; return "$rc"; }
    echo "cap done: $RUN_DIR/copy-$tag.nsys-rep + $out/workload.log"
}

report() {
    local tag=${1:?usage: copy-nsys.sh report <tag>}
    local rep="$RUN_DIR/copy-$tag.nsys-rep"
    [[ -f $rep ]] || rep="$RUN_DIR/$tag.nsys-rep"   # legacy captures (nsys.sh / 07-21)
    [[ -f $rep ]] || { echo "no report: $RUN_DIR/{copy-$tag,$tag}.nsys-rep" >&2; return 1; }
    local db="${rep%.nsys-rep}.sqlite"
    if [[ ! -f $db || $rep -nt $db ]]; then
        echo "report: exporting $rep -> sqlite" >&2
        nsys export --type sqlite --force-overwrite=true -o "$db" "$rep" >/dev/null 2>&1
    fi
    GAP_MS="$GAP_MS" MIN_BURST_MB="$MIN_BURST_MB" python3 - "$db" <<'PYEOF'
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
gap_ns = float(os.environ["GAP_MS"]) * 1e6
min_b = float(os.environ["MIN_BURST_MB"]) * 1e6
KIND = {1: "H2D (load,  L1->GPU)", 2: "D2H (store, GPU->L1)"}

tabs = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "CUPTI_ACTIVITY_KIND_MEMCPY" not in tabs:
    sys.exit("no CUPTI_ACTIVITY_KIND_MEMCPY table — was this captured with -t cuda?")

print("== per-direction copy summary (op level: Σ bytes / Σ per-op device time) ==")
for k in (2, 1):
    n, byt, opsum = db.execute(
        "SELECT COUNT(*), SUM(bytes), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        f"WHERE copyKind={k}").fetchone()
    if not n:
        print(f"   {KIND[k]}: none"); continue
    size, cnt = db.execute(
        "SELECT bytes, COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        f"WHERE copyKind={k} GROUP BY bytes ORDER BY 2 DESC LIMIT 1").fetchone()
    print(f"   {KIND[k]}: {n} copies, {byt/2**30:.2f} GiB, op-sum {opsum/1e6:.0f} ms"
          f" -> {byt/opsum:.1f} GB/s ; dominant size {size/2**20:.2f} MiB x{cnt}"
          f" ({opsum/n/1e3:.0f} us/copy avg)")

print("\n== stream layout (which directions share a copy stream) ==")
streams = {}
for k, sid, n in db.execute(
        "SELECT copyKind, streamId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE copyKind IN (1,2) GROUP BY 1,2"):
    streams.setdefault(sid, []).append((k, n))
for sid, kinds in sorted(streams.items()):
    names = ", ".join(f"{KIND[k].split()[0]} x{n}" for k, n in kinds)
    print(f"   stream {sid}: {names}")
if any(len(k) > 1 for k in streams.values()):
    print("   -> BOTH directions on one stream: a store flush and a load can never overlap")

print(f"\n== bursts (memcpys <{os.environ['GAP_MS']}ms apart; wall GB/s = what a request"
      f" experiences, busy% = copy-engine duty cycle) ==")
for k in (2, 1):
    rows = db.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                      f"WHERE copyKind={k} ORDER BY start").fetchall()
    bursts = []
    for s, e, b in rows:
        if bursts and s - bursts[-1][1] < gap_ns:
            bursts[-1][1] = max(bursts[-1][1], e); bursts[-1][2] += b
            bursts[-1][3] += e - s; bursts[-1][4] += 1
        else:
            bursts.append([s, e, b, e - s, 1])
    big = [x for x in bursts if x[2] > min_b]
    print(f"   {KIND[k]}: {len(big)} bursts >{os.environ['MIN_BURST_MB']}MB"
          f" ({len(bursts)-len(big)} smaller hidden)")
    walls, busys = [], []
    for s, e, b, op, n in big:
        wall = e - s
        walls.append(b / wall); busys.append(op / wall)
        if len(walls) <= 8:
            print(f"      {b/1e9:5.2f} GB, {n:5d} copies, wall {wall/1e6:7.1f} ms"
                  f" -> {b/wall:5.1f} GB/s wall, busy {op/wall*100:4.1f}%")
    if walls:
        walls.sort(); busys.sort(); m = len(walls) // 2
        print(f"      median: {walls[m]:.1f} GB/s wall, busy {busys[m]*100:.0f}%"
              f"  (idle gaps = per-copy submit overhead, allocator, batching)")

if "NVTX_EVENTS" in tabs:
    print("\n== NVTX push/pop ranges (server-side copy path) ==")
    rows = db.execute(
        "SELECT COALESCE(s.value, e.text) nm, COUNT(*), SUM(e.end-e.start), AVG(e.end-e.start) "
        "FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id "
        "WHERE e.end IS NOT NULL AND e.eventType = 59 GROUP BY nm ORDER BY 3 DESC").fetchall()
    total = sum(r[2] for r in rows) or 1
    for nm, n, tot, avg in rows:
        print(f"   {nm[:52]:52s} n={n:5d} total {tot/1e6:8.1f} ms  avg {avg/1e6:6.2f} ms"
              f"  ({tot/total*100:4.1f}% of range time)")
PYEOF
}

down() {
    "$MODEL_SCRIPT" stop || true
    "$HERE/lmcache-ctl.sh" stop || true
    nsys shutdown --session="$SESS" >/dev/null 2>&1 || true
    echo "== GPU state after teardown =="
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "${VLLM_GPU:-$GPU}"
}

case ${1:-} in
    check)    check ;;
    up)       up ;;
    cap)      shift; cap "$@" ;;
    report)   shift; report "$@" ;;
    down)     down ;;
    status)   "$MODEL_SCRIPT" status ;;
    logs)     case ${2:-server} in
                  vllm)   "$MODEL_SCRIPT" logs "${3:-}" ;;
                  server) "$HERE/lmcache-ctl.sh" logs "${3:-}" ;;
                  *)      print_usage; exit 1 ;;
              esac ;;
    winopen)  shift; winopen "$@" ;;
    winclose) winclose ;;
    *) print_usage; exit 1 ;;
esac
