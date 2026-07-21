#!/usr/bin/env bash
# Qwen3.5-9B.sh — vLLM instance for Qwen3.5-9B (linear-attention / hybrid-mamba model)
# bound to the shared LMCache server, per https://docs.lmcache.ai/recipes/qwen3_5.html
#
# Recipe requirements baked in below:
#   --mamba-cache-mode align       mandatory for mamba prefix caching (marked experimental)
#   --enable-prefix-caching
#   LMCache chunk-size == vLLM unified block size = 528 for this model
#     (probed 2026-07-21 from vllm 0.25.1 startup log: "Setting attention block size to
#      528 tokens to ensure that attention page size is >= mamba page size";
#      recipe reference values: Qwen3.5-0.8B=544, Qwen3.6-27B=784)
#   --max-num-batched-tokens = 2*528-1 = 1055 (exactly N would serialize prefill/decode)
#   --max-num-seqs lowered: each decode seq needs one mamba cache block; at mem-util 0.4
#     only ~985 blocks exist and the vLLM default of 1024 aborts startup
# Recipe caveats: generation is NOT bit-exact between cached and fresh runs; cache cannot
# be shared across engines with different attention backends.
#
# Usage:
#   ./Qwen3.5-9B.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection: GPU=N (global) / VLLM_GPU=N (this instance) / LMCACHE_GPU=N (server)
# Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, MAX_NUM_SEQS, UNIFIED_BLOCK,
#                   LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-$GPU}   # instance GPU: VLLM_GPU beats the global GPU default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-/dev/shm/models/$MODEL_NAME}
VLLM_PORT=${VLLM_PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-600}
UNIFIED_BLOCK=${UNIFIED_BLOCK:-528}     # vLLM's unified (attention/mamba) block size for this model
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-$UNIFIED_BLOCK}
EXTRA_VLLM_ARGS=(
    --enable-prefix-caching
    --mamba-cache-mode align
    --max-num-batched-tokens $((2 * UNIFIED_BLOCK - 1))
    --max-num-seqs "$MAX_NUM_SEQS"
)

model_dispatch "$@"
