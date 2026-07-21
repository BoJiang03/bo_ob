#!/usr/bin/env bash
# l2-gain.sh — demonstrate the end-to-end performance gain of the L2 tier (homework
# Part 2, "Try one of the L2 adapters and demonstrate the performance gain of L2").
#
# Runs the SAME populate->reset->measure scenario for several server configs, LMCache on
# every time; only the L2 tier differs.  Arms (env ARMS, default "no-l2 fs fs_native"):
#   no-l2      : small L1, no L2            -> the L1-overflow recomputes (prefill)
#   fs         : small L1 + fs L2 (disk)    -> overflow reloads from pure-Python disk
#   fs_native  : small L1 + fs_native L2    -> overflow reloads from C++/page-cache disk
#
# The scenario (l2_scenario.py) fills LMCache with a working set >> L1, RESETS vLLM's GPU
# prefix cache, then measures re-read TTFT — so the re-read is served by LMCache's tiers,
# not vLLM's own cache.  Whether L2 *wins* depends on regime: L2 helps only when recompute
# (prefill) costs more than loading the KV back.  Long contexts make prefill expensive, so
# the default doc length is large.  fs_native (fast) beats recompute more easily than fs.
#
# Single GPU (default 7), self-contained in l2_support/.
#
# Usage:  ./l2-gain.sh
#         ARMS="no-l2 fs_native" NUM_DOCS=8 DOC_TOKENS=20000 ./l2-gain.sh
#         KEEP_UP=1 ./l2-gain.sh
# Env: MODEL ARMS NUM_DOCS DOC_TOKENS L1_SIZE_GB GPU_MEM_UTIL GPU L2_BASE OUT_ROOT KEEP_UP

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

MODEL=${MODEL:-Qwen3-8B}
ARMS=${ARMS:-"no-l2 fs fs_native"}
NUM_DOCS=${NUM_DOCS:-8}
DOC_TOKENS=${DOC_TOKENS:-20000}                 # long -> expensive prefill -> L2 can win
SEED=${SEED:-1000}
L1_SIZE_GB=${L1_SIZE_GB:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}             # bigger GPU pool: fits the long contexts
CHUNK=${LMCACHE_CHUNK_SIZE:-16}
L2_BASE=${L2_BASE:-/data1/bo/l2cache-demo}
VLLM_PORT=${VLLM_PORT:-8001}
HTTP_PORT=${LMCACHE_HTTP_PORT:-8081}
OUT_ROOT=${OUT_ROOT:-$HERE/bench-results}
CMP_DIR=${OUT_DIR:-$OUT_ROOT/$(date +%Y%m%d-%H%M%S)-$MODEL-l2gain}
mkdir -p "$CMP_DIR"

export GPU_MEM_UTIL LMCACHE_CHUNK_SIZE=$CHUNK VLLM_PORT

echo "=== l2-gain: $MODEL / arms=[$ARMS] / ${NUM_DOCS} docs x ${DOC_TOKENS} tok / L1 ${L1_SIZE_GB}GB / gpu-mem $GPU_MEM_UTIL"
echo "=== output -> $CMP_DIR"
gpu_mem_warn 40000 "GPU $GPU busy — vLLM (mem-util $GPU_MEM_UTIL) may OOM"

"$HERE/$MODEL.sh" stop || true
"$HERE/lmcache-ctl.sh" stop || true

adapter_json() { # arm -> prints the L2 adapter JSON ("" for no-l2)
    case $1 in
        no-l2)     echo "" ;;
        fs)        echo "{\"type\":\"fs\",\"base_path\":\"$L2_BASE/fs\"}" ;;
        fs_native) echo "{\"type\":\"fs_native\",\"base_path\":\"$L2_BASE/fs_native\",\"num_workers\":4}" ;;
        *) echo "UNKNOWN" ;;
    esac
}

run_arm() { # arm
    local arm=$1 l2json; l2json=$(adapter_json "$arm")
    [[ $l2json == UNKNOWN ]] && { echo "!! unknown arm $arm — skipping" >&2; return 0; }
    local dir="$CMP_DIR/$arm"; mkdir -p "$dir"
    echo; echo "=== [$arm] server L1=${L1_SIZE_GB}GB${l2json:+, L2=$l2json}"
    "$HERE/$MODEL.sh" stop || true
    "$HERE/lmcache-ctl.sh" stop || true
    if [[ -n $l2json ]]; then
        local base; base=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['base_path'])" "$l2json")
        rm -rf "$base"; mkdir -p "$base"
        CHUNK_SIZE=$CHUNK L1_SIZE_GB=$L1_SIZE_GB L2_ADAPTER="$l2json" "$HERE/lmcache-ctl.sh" start
    else
        CHUNK_SIZE=$CHUNK L1_SIZE_GB=$L1_SIZE_GB "$HERE/lmcache-ctl.sh" start
    fi
    "$HERE/$MODEL.sh" start
    python3 "$HERE/l2_scenario.py" \
        --base-url "http://localhost:$VLLM_PORT" --model "$MODEL" \
        --num-docs "$NUM_DOCS" --doc-tokens "$DOC_TOKENS" --seed "$SEED" \
        --reset-url "http://localhost:$VLLM_PORT/reset_prefix_cache" \
        --label "$arm" --out "$dir/scenario.json" 2>&1 | tee "$dir/scenario.log"
    curl -sf "http://localhost:$HTTP_PORT/metrics" -o "$dir/lmcache-metrics.txt" 2>/dev/null || true
    if [[ -n $l2json ]]; then
        local base; base=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['base_path'])" "$l2json")
        echo "    $arm L2 on-disk footprint: $(du -sh "$base" 2>/dev/null | cut -f1)"
    fi
}

for arm in $ARMS; do run_arm "$arm"; done

if [[ ${KEEP_UP:-0} != 1 ]]; then
    "$HERE/$MODEL.sh" stop || true
    "$HERE/lmcache-ctl.sh" stop || true
fi

echo
python3 - "$CMP_DIR" "$ARMS" <<'EOF' | tee "$CMP_DIR/summary.txt"
import json, os, sys
d = sys.argv[1]; arms = sys.argv[2].split()
data = {}
for a in arms:
    p = os.path.join(d, a, 'scenario.json')
    if os.path.exists(p): data[a] = json.load(open(p))
if not data:
    print("no scenario results found"); sys.exit(1)
any_v = next(iter(data.values()))
print(f"\n=== L2 gain (post-reset re-read TTFT): {os.path.basename(d)}")
print(f"working set: {any_v['num_docs']} docs x {any_v['doc_tokens']} tok "
      f"= {any_v['total_prompt_tokens']} prompt tokens; reset={any_v['reset']}")
base = data.get('no-l2')
hdr = f"{'arm':<12}{'mean TTFT':>12}{'median':>10}{'p90':>10}"
if base: hdr += f"{'vs no-l2':>10}"
print("\n" + hdr); print('-'*len(hdr))
for a in arms:
    if a not in data: continue
    t = data[a]['ttft_ms']
    line = f"{a:<12}{t['mean']:>10.1f}ms{t['median']:>8.1f}ms{t['p90']:>8.1f}ms"
    if base:
        g = base['ttft_ms']['mean']/t['mean'] if t['mean'] else float('nan')
        line += f"{g:>9.2f}x"
    print(line)
print("\n(vs no-l2 >1 => that L2 adapter served the L1-overflow faster than recompute)")
EOF
echo "=== results in $CMP_DIR"
