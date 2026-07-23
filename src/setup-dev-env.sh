#!/usr/bin/env bash
# setup-dev-env.sh — build the code-reading dev env: fresh vllm + LMCache clones pinned to
# the exact versions the serving harnesses run (vllm v0.25.1, lmcache v0.5.1), editable-
# installed into a dedicated venv. The clones live HERE in onbording/src/ as physical dirs
# (~1 GiB, gitignored); only the heavy venv (9+ GB) stays on /data1 (/ is ~93% full).
#
# Creates:
#   onbording/src/vllm       vllm @ v0.25.1   — editable, VLLM_USE_PRECOMPILED=1
#                            (binary kernels fetched prebuilt; only the Python layer is
#                            editable — which is the layer we read and edit)
#   onbording/src/lmcache    LMCache @ v0.5.1 — editable, compiles lmcache.c_ops locally
#   /data1/bo/dev/venv       uv venv, python 3.12
#
# Usage:
#   ./setup-dev-env.sh              # all steps in order; every step is idempotent
#   ./setup-dev-env.sh <step>       # one of: clone | venv | install | verify
#
# Zero GPU; safe to run anytime. Never touches /home/bo/lmcache/.venv — the serving
# harnesses stay on that venv. Also distinct from /data1/bo/LMCache (the dev-branch
# annotation clone backing code_structure/'s file:line refs) and from
# /data1/bo/venvs/lmcache-profiling.
#
# Caveat: don't start python with onbording/src/ as the cwd — the vllm/lmcache dirs
# here shadow the installed packages via sys.path[0] (python -P avoids this).

set -euo pipefail

UV="$HOME/.local/bin/uv"
PYTHON_VERSION=3.12

DEV_ROOT=/data1/bo/dev
VENV="$DEV_ROOT/venv"
PY="$VENV/bin/python"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"   # clones live next to this script

VLLM_REPO=https://github.com/vllm-project/vllm.git
VLLM_TAG=v0.25.1
VLLM_DIR="$SRC_DIR/vllm"

LMC_REPO=https://github.com/LMCache/LMCache.git
LMC_TAG=v0.5.1
LMC_DIR="$SRC_DIR/lmcache"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"          # nvcc for the lmcache c_ops build

log() { printf '\n=== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
print_usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; }

clone_one() {  # <repo-url> <dir> <tag>
    local repo=$1 dir=$2 tag=$3
    if [[ -d "$dir/.git" ]]; then
        log "clone: $dir already exists — fetching tags"
        git -C "$dir" fetch --tags --quiet
    else
        log "clone: $repo -> $dir (full history, so git blame/log work)"
        git clone "$repo" "$dir"
    fi
    git -C "$dir" rev-parse -q --verify "refs/tags/$tag^{commit}" >/dev/null \
        || die "no tag $tag in $dir; last tags: $(git -C "$dir" tag | sort -V | tail -5 | tr '\n' ' ')"
    # local branch (not detached HEAD) so reading-session edits/commits have a home
    git -C "$dir" switch -q -C "local/$tag" "$tag"
    log "clone: $dir @ $(git -C "$dir" log --oneline -1)"
}

step_clone() {
    clone_one "$VLLM_REPO" "$VLLM_DIR" "$VLLM_TAG"
    clone_one "$LMC_REPO"  "$LMC_DIR"  "$LMC_TAG"
}

step_venv() {
    [[ -x $UV ]] || die "uv not found at $UV"
    mkdir -p "$DEV_ROOT"
    if [[ -x $PY ]]; then
        log "venv: $VENV already exists ($("$PY" --version))"
    else
        log "venv: creating $VENV (python $PYTHON_VERSION)"
        "$UV" venv --python "$PYTHON_VERSION" "$VENV"
    fi
}

step_install() {
    [[ -x $PY ]] || die "venv missing — run the venv step first"
    [[ -d $VLLM_DIR ]] || die "clones missing — run the clone step first"

    log "install: vllm editable (VLLM_USE_PRECOMPILED=1 — fetches release kernels, no local CUDA build)"
    VLLM_USE_PRECOMPILED=1 "$UV" pip install --python "$PY" -e "$VLLM_DIR"

    log "install: lmcache build deps, then editable (compiles c_ops; nvcc: $(command -v nvcc || echo MISSING))"
    "$UV" pip install --python "$PY" setuptools wheel ninja cmake packaging
    "$UV" pip install --python "$PY" --no-build-isolation -e "$LMC_DIR"

    log "install: nixl==1.3.1 (matches the serving venv; keeps the P2P/NIXL adapters importable)"
    "$UV" pip install --python "$PY" 'nixl==1.3.1'
}

step_verify() {
    log "verify: imports resolve to the editable clones"
    # -P: keep cwd out of sys.path — the vllm/lmcache dirs next to this script
    # would otherwise shadow the installed packages as namespace packages
    "$PY" -P - "$VLLM_DIR" "$LMC_DIR" <<'EOF'
import sys
vllm_dir, lmc_dir = sys.argv[1], sys.argv[2]
import vllm, lmcache
ok = True
for name, mod, want in (("vllm", vllm, vllm_dir), ("lmcache", lmcache, lmc_dir)):
    # editable/namespace packages may have __file__ = None; fall back to __path__
    path = mod.__file__ or next(iter(getattr(mod, "__path__", [])), None) or "<unknown>"
    good = path.startswith(want + "/")
    ok &= good
    print(f"{name:9} {mod.__version__:12} {path}  {'OK' if good else '!! not from ' + want}")
import lmcache.c_ops  # compiled extension must import
print("c_ops     OK")
sys.exit(0 if ok else 1)
EOF
    log "verify: clone states"
    git -C "$VLLM_DIR" log --oneline -1
    git -C "$LMC_DIR"  log --oneline -1
    log "all good — activate with: source $VENV/bin/activate"
}

step="${1:-all}"
case "$step" in
    all)     step_clone; step_venv; step_install; step_verify ;;
    clone)   step_clone ;;
    venv)    step_venv ;;
    install) step_install ;;
    verify)  step_verify ;;
    *)       print_usage; exit 1 ;;
esac
