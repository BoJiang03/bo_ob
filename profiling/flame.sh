#!/usr/bin/env bash
# flame.sh — CPU flamegraph of the vLLM (or lmcache-server) process via py-spy.
#   gil  = only samples where a thread holds the GIL   (py-spy --gil)
#   wall = include idle / blocked threads, wall-clock  (py-spy --idle)
# On this box perf_event_paranoid=4 and ptrace_scope=1 block perf/bcc and plain
# attach, so py-spy runs under `sudo -n` (root bypasses Yama). py-spy reads the
# target via process_vm_readv and resolves Python frames itself -- no perf maps,
# no PYTHONPERFSUPPORT needed. --subprocesses follows vLLM's EngineCore children.
#
# Usage: ./flame.sh {vllm|server|<pid>} [tag]
# Config: DURATION, RATE, MODES(gil,wall), PYSPY(abs path or name), VLLM_PORT
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

FLAME_DIR="$HERE/flames"; mkdir -p "$FLAME_DIR"
DURATION=${DURATION:-30}
RATE=${RATE:-99}
MODES=${MODES:-gil,wall}
TARGET=${1:-vllm}
TAG=${2:-$(date +%H%M%S)}

# resolve py-spy to an absolute path (sudo env_reset drops PATH)
PYSPY_ABS=$(command -v "${PYSPY:-py-spy}" 2>/dev/null || true)
[[ -x $PYSPY_ABS ]] || { echo "ERROR: py-spy not found (set PYSPY=/abs/path/py-spy)" >&2; exit 1; }

case "$TARGET" in
    vllm)   PID=$(cat "$RUN_DIR/Qwen3-8B.pid" 2>/dev/null || true) ;;
    server) PID=$(cat "$RUN_DIR/lmcache-server.pid" 2>/dev/null || true) ;;
    ''|*[!0-9]*) echo "ERROR: target must be vllm|server|<pid>" >&2; exit 1 ;;
    *)      PID=$TARGET ;;
esac
[[ -n ${PID:-} ]] && kill -0 "$PID" 2>/dev/null || { echo "ERROR: target '$TARGET' has no live pid" >&2; exit 1; }

echo "flame: target=$TARGET pid=$PID modes=$MODES duration=${DURATION}s rate=${RATE}Hz"
for mode in ${MODES//,/ }; do
    case "$mode" in
        gil)  extra=(--gil) ;;
        wall) extra=(--idle) ;;
        *)    echo "  skip unknown mode '$mode'"; continue ;;
    esac
    out="$FLAME_DIR/${TARGET}_${mode}_${TAG}.svg"
    echo "  py-spy $mode -> $out"
    if sudo -n env "PATH=$(dirname "$PYSPY_ABS"):/usr/bin:/bin" "$PYSPY_ABS" record \
            --pid "$PID" --subprocesses --format flamegraph --rate "$RATE" \
            --duration "$DURATION" --output "$out" "${extra[@]}"; then
        sudo -n chown "$(id -u):$(id -g)" "$out" 2>/dev/null || true
        echo "    ok ($(du -h "$out" 2>/dev/null | cut -f1))"
    else
        echo "    py-spy $mode FAILED" >&2
    fi
done
echo "flame: done -> $FLAME_DIR"
