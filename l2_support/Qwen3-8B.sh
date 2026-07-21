#!/usr/bin/env bash
# Qwen3-8B.sh — vLLM instance for Qwen3-8B (dense model) bound to the shared LMCache
# server, per https://docs.lmcache.ai/getting_started/quickstart.html
#
# The filename IS the model name: this script serves /dev/shm/models/Qwen3-8B.
# To manage another model, copy it to <Model-Name>.sh (and give it its own VLLM_PORT).
# `start` auto-starts the LMCache server (with this model's chunk-size) if it isn't up;
# `stop` stops only this vLLM instance — the server and its cache stay alive.
#
# Usage:
#   ./Qwen3-8B.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection:
#   GPU=N        — global default, applies to this instance AND an auto-started server
#   VLLM_GPU=N   — this vLLM instance only (e.g. second instance on another card)
#   LMCACHE_GPU  — consumed by lmcache-ctl.sh if `start` has to auto-start the server
# e.g. second instance sharing the GPU-7 server:  VLLM_GPU=4 VLLM_PORT=8001 ./Qwen3-8B.sh start
#
# Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, LMCACHE_PORT, LMCACHE_CHUNK_SIZE,
#                   VENV, VLLM_START_TIMEOUT

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-$GPU}   # instance GPU: VLLM_GPU beats the global GPU default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL_NAME}
VLLM_PORT=${VLLM_PORT:-8001}   # l2_support default (setup uses 8000)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-600}
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-16}    # dense model: chunk-size is a free choice
EXTRA_VLLM_ARGS=()

model_dispatch "$@"
