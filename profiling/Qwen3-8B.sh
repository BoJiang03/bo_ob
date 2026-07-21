#!/usr/bin/env bash
# Qwen3-8B.sh — vLLM instance for Qwen3-8B (dense model) bound to the shared LMCache
# server. COPY of setup/Qwen3-8B.sh for Part 3 (profiling); only VLLM_PORT default
# differs (8002 vs setup's 8000). Serves /dev/shm/models/Qwen3-8B.
#
# Usage:
#   ./Qwen3-8B.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection:
#   GPU=N        — global default, applies to this instance AND an auto-started server
#   VLLM_GPU=N   — this vLLM instance only (e.g. GPU 0 on rtx-026)
# Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, LMCACHE_PORT, LMCACHE_CHUNK_SIZE,
#                   VENV, VLLM_START_TIMEOUT, NSYS_PREFIX (set by nsys.sh)

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-$GPU}   # instance GPU: VLLM_GPU beats the global GPU default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL_NAME}
VLLM_PORT=${VLLM_PORT:-8002}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-600}
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-16}    # dense model: chunk-size is a free choice
EXTRA_VLLM_ARGS=()

model_dispatch "$@"
