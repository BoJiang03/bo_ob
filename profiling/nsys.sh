#!/usr/bin/env bash
# nsys.sh — capture & analyze an nsys timeline of vLLM + LMCache under workload.
#
#   cap <tag>     Bring up the lmcache server and vLLM, with vLLM under `nsys launch`
#                 (profiler injected, collection DEFERRED). Once vLLM is healthy,
#                 `nsys start` opens the collection window, workload.sh drives H2D/D2H
#                 (and P2P) traffic, then `nsys stop` writes run/<tag>.nsys-rep.
#                 Deferring collection until the server is healthy keeps model-load
#                 noise out of the window and avoids guessing a fixed --delay.
#
#   report <tag>  nsys stats: CUDA MemOps by time & size (H2D/D2H copy throughput)
#                 plus NVTX push/pop and start/end range summaries.
#
# NVTX needs no privilege here (pure userspace); CUDA MemOps come from CUPTI, which
# also needs no perf_event access -- so this runs fine under perf_event_paranoid=4.
#
# Config: VLLM_GPU(0), NSYS_TRACE(cuda,nvtx), SESS(lmcprof), KEEP_UP=1 (leave vLLM up),
#         plus all WL_* workload knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
SESS=${SESS:-lmcprof}
MODEL_SCRIPT="$HERE/Qwen3-8B.sh"

cap() {
    local tag=${1:?usage: nsys.sh cap <tag>}
    local out="$RUN_DIR/$tag.nsys-rep"
    # WHICH process to profile. In MP mode the real KV H2D/D2H copy (to_gpu/from_gpu)
    # runs in the lmcache SERVER, not vLLM -- so 'server' is the default for Task ④.
    # 'vllm' instead captures the client-side adapter submit/lookup overhead.
    local target=${NSYS_TARGET:-server}
    local gpu=${VLLM_GPU:-0}
    local launch="nsys launch --session-new=$SESS -t $NSYS_TRACE"  # trace fixed at launch
    nsys shutdown --session="$SESS" >/dev/null 2>&1 || true

    echo "nsys: cap target=$target (session=$SESS, trace=$NSYS_TRACE)"
    if [[ $target == server ]]; then
        NSYS_PREFIX="$launch" LMCACHE_GPU="$gpu" "$HERE/lmcache-ctl.sh" start
        VLLM_GPU="$gpu" "$MODEL_SCRIPT" start || { echo "ERROR: vLLM failed to start" >&2; return 1; }
    else
        LMCACHE_GPU="$gpu" "$HERE/lmcache-ctl.sh" start
        NSYS_PREFIX="$launch" VLLM_GPU="$gpu" "$MODEL_SCRIPT" start \
            || { echo "ERROR: vLLM under nsys failed to start" >&2; return 1; }
    fi

    echo "nsys: opening collection window -> $out"
    nsys start --session="$SESS" -o "$out" -f true   # -o/-f only; trace was set at launch
    "$HERE/workload.sh"
    echo "nsys: closing collection window"
    nsys stop --session="$SESS"

    if [[ ${KEEP_UP:-0} != 1 ]]; then "$MODEL_SCRIPT" stop; "$HERE/lmcache-ctl.sh" stop; fi
    if [[ -f $out ]]; then
        echo "cap done: $out ($(du -h "$out" | cut -f1))"
    else
        echo "WARN: expected report $out not found; check 'nsys sessions list'" >&2
        return 1
    fi
}

report() {
    local tag=${1:?usage: nsys.sh report <tag>}
    local rep="$RUN_DIR/$tag.nsys-rep"
    [[ -f $rep ]] || { echo "no report: $rep" >&2; return 1; }
    echo "########## H2D / D2H copy — by TIME ##########"
    nsys stats --force-export=true --report cuda_gpu_mem_time_sum "$rep" 2>/dev/null
    echo "########## H2D / D2H copy — by SIZE ##########"
    nsys stats --force-export=true --report cuda_gpu_mem_size_sum "$rep" 2>/dev/null
    echo "########## NVTX push/pop ranges (sync annotations: to_gpu/from_gpu, L2 adapter) ##########"
    nsys stats --force-export=true --report nvtx_pushpop_sum "$rep" 2>/dev/null
    echo "########## NVTX start/end ranges (async annotations: P2P coroutines) ##########"
    nsys stats --force-export=true --report nvtx_startend_sum "$rep" 2>/dev/null
}

case ${1:-} in
    cap)    shift; cap "$@" ;;
    report) shift; report "$@" ;;
    *) echo "usage: nsys.sh {cap|report} <tag>"; exit 1 ;;
esac
