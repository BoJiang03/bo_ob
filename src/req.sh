#!/usr/bin/env bash
# req.sh — poke the debug vLLM instance (vllm-dbg.sh) while stepping LMCache in PyCharm.
#
# Usage:
#   ./req.sh send [text]    # POST /v1/completions (streaming) with a FIXED prompt —
#                           # identical token ids every run, so lmcache chunk hashes
#                           # match across sends. Generates ONE token and reports TTFT
#                           # (request sent -> first streamed token) + cached-token count.
#                           # Optional [text] replaces the fixed prompt.
#   ./req.sh clear_cache    # POST /reset_prefix_cache — wipe vLLM's own KV/prefix cache.
#                           # LMCache keeps its copy, so the next `send` of the same
#                           # prompt exercises LOOKUP-hit -> RETRIEVE instead of compute.
#
# Debug loop: send (STORE, D2H) -> clear_cache -> send (RETRIEVE, H2D) -> step in PyCharm.
#
# Env: VLLM_PORT (8003), MODEL (Qwen3-8B), REPEAT (fixed-prompt paragraphs, default 12,
#      ~55 tokens each — raise it to cross more 16-token chunks), MAX_TOKENS (1)

set -euo pipefail

VLLM_PORT=${VLLM_PORT:-8003}
MODEL=${MODEL:-Qwen3-8B}
REPEAT=${REPEAT:-12}
MAX_TOKENS=${MAX_TOKENS:-1}
BASE="http://localhost:$VLLM_PORT"

print_usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; }

fixed_prompt() {
    local i
    for i in $(seq 1 "$REPEAT"); do
        printf 'Paragraph %d: the quick brown fox jumps over the lazy dog while the cache warms up, storing key value pairs chunk by chunk into pinned host memory for later retrieval. ' "$i"
    done
}

send() {
    local prompt=${1:-$(fixed_prompt)}
    curl -sf -o /dev/null "$BASE/v1/models" \
        || { echo "ERROR: vLLM not reachable at $BASE (./vllm-dbg.sh status)" >&2; exit 1; }
    PROMPT="$prompt" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" python3 - "$BASE" <<'EOF'
import json, os, sys, time, urllib.request
base = sys.argv[1]
payload = json.dumps({
    "model": os.environ["MODEL"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()
req = urllib.request.Request(base + "/v1/completions", data=payload,
                             headers={"Content-Type": "application/json"})
ttft = None
text = []
usage = {}
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=600) as r:   # generous: breakpoints pause the server
    for raw in r:                                     # SSE: lines of "data: {...}"
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        if chunk.get("choices") and chunk["choices"][0].get("text"):
            if ttft is None:
                ttft = time.monotonic() - t0
            text.append(chunk["choices"][0]["text"])
        if chunk.get("usage"):
            usage = chunk["usage"]
total = time.monotonic() - t0
details = usage.get("prompt_tokens_details") or {}
print(f"TTFT             : {ttft*1000:.0f} ms" if ttft is not None else "TTFT             : n/a (no token streamed)")
print(f"total            : {total*1000:.0f} ms")
print(f"prompt_tokens    : {usage.get('prompt_tokens')}")
print(f"cached_tokens    : {details.get('cached_tokens', 'n/a')}  (vLLM prefix cache + LMCache hits)")
print(f"completion_tokens: {usage.get('completion_tokens')}")
print(f"text             : {''.join(text)!r}")
EOF
}

clear_cache() {
    # requires the instance to run with VLLM_SERVER_DEV_MODE=1 (vllm-dbg.sh does)
    if curl -sf -X POST "$BASE/reset_prefix_cache" -o /dev/null; then
        echo "vLLM prefix cache cleared (lmcache side untouched)"
    else
        echo "ERROR: reset_prefix_cache failed — is vLLM up, and started via vllm-dbg.sh (VLLM_SERVER_DEV_MODE=1)?" >&2
        exit 1
    fi
}

case ${1:-} in
    send)        shift; send "$@" ;;
    clear_cache) clear_cache ;;
    *)           print_usage; exit 1 ;;
esac
