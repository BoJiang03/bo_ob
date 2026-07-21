#!/usr/bin/env bash
# workload.sh — deterministic traffic that exercises the LMCache copy paths.
#
# Each round: send M distinct long-prefix prompts (COLD -> vLLM prefills, KV is
# stored to LMCache = D2H copy), then POST /reset_prefix_cache to wipe vLLM's own
# GPU prefix cache, then re-send the SAME M prompts (WARM -> LMCache hit, KV is
# loaded back into the GPU = H2D copy). Run this while nsys.sh / flame.sh captures.
#
# Config via env: VLLM_PORT, MODEL_NAME, WL_PROMPTS, WL_PREFIX_WORDS, WL_ROUNDS, WL_MAXTOK
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME=${MODEL_NAME:-Qwen3-8B}
VLLM_PORT=${VLLM_PORT:-8002}
BASE="http://localhost:$VLLM_PORT"
WL_PROMPTS=${WL_PROMPTS:-16}            # distinct prompts per round
WL_PREFIX_WORDS=${WL_PREFIX_WORDS:-1200} # ~tokens of prefix per prompt (KV volume knob)
WL_ROUNDS=${WL_ROUNDS:-3}              # store/reset/load cycles
WL_MAXTOK=${WL_MAXTOK:-8}             # generation length; we care about prefill KV, keep small
WL_SETTLE=${WL_SETTLE:-3}             # seconds to let async D2H stores flush before reset/reload

api() { curl -sf -m 180 "$@"; }

reset_prefix_cache() {
    if api -X POST "$BASE/reset_prefix_cache" -o /dev/null; then
        echo "  [reset_prefix_cache ok]"
    else
        echo "  [reset_prefix_cache FAILED — vLLM must run with VLLM_SERVER_DEV_MODE=1]" >&2
    fi
}

# build a ~WL_PREFIX_WORDS-word prompt whose content is DISTINCT per id (seeded random,
# so prompts don't share a prefix -- otherwise vLLM's own prefix cache dedups them and
# nothing but the first is really prefilled/stored). Same id -> same prompt -> LMCache hit.
make_prompt() {
    python3 -c "import random,sys
random.seed(sys.argv[1])
print('doc'+sys.argv[1]+' '+' '.join('w%d'%random.randint(0,99999) for _ in range($WL_PREFIX_WORDS)))" "$1"
}

send() { # prompt
    api "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL_NAME" --arg p "$1" --argjson n "$WL_MAXTOK" \
              '{model:$m, prompt:$p, max_tokens:$n, temperature:0}')" -o /dev/null
}

echo "workload: $WL_PROMPTS prompts × $WL_ROUNDS rounds (~$WL_PREFIX_WORDS toks each) -> $BASE [$MODEL_NAME]"
api "$BASE/v1/models" -o /dev/null || { echo "ERROR: $BASE not healthy" >&2; exit 1; }

t0=$(date +%s)
for ((r = 1; r <= WL_ROUNDS; r++)); do
    echo "== round $r/$WL_ROUNDS =="
    echo " store phase (cold -> D2H):"
    for ((k = 1; k <= WL_PROMPTS; k++)); do send "$(make_prompt "$r-$k")"; done
    echo "  stored $WL_PROMPTS prompts; settling ${WL_SETTLE}s for async D2H to flush"
    sleep "$WL_SETTLE"
    reset_prefix_cache
    echo " load phase (warm -> H2D):"
    for ((k = 1; k <= WL_PROMPTS; k++)); do send "$(make_prompt "$r-$k")"; done
    echo "  reloaded $WL_PROMPTS prompts"
    sleep "$WL_SETTLE"
done
echo "workload: done in $(( $(date +%s) - t0 ))s"
