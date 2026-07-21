#!/usr/bin/env bash
# DeepSeek-V4-Flash.sh — vLLM instance for DeepSeek-V4-Flash (hybrid model: compressed
# MLA latents + sparse-attention indexers) bound to the shared LMCache server,
# per https://docs.lmcache.ai/recipes/deepseek_v4_flash.html
#
# *** UNTESTED ON THIS NODE ***  It needs all 8 GPUs (TP=8) — currently used by others —
# and the weights are not in /dev/shm/models; the default MODEL_PATH is the HF id, which
# would download hundreds of GB on first start. Faithful to the recipe; validate first.
#
# Recipe requirements baked in below:
#   --kv-cache-dtype fp8_ds_mla / --tokenizer-mode deepseek_v4   mandatory for this model
#   --enable-expert-parallel        distributes MoE experts across TP ranks
#   --tensor-parallel-size 8        adjust TP_SIZE to your hardware
#   vLLM version: use a tagged release, NOT the dev branch (fp4 MoE expert misdispatching)
#   lmcache server: recipe uses default chunk-size 256 and --l1-size-gb 100
#     (pass L1_SIZE_GB=100 when this script auto-starts the server)
# The interleaved KV cache groups (fp8/uint8 MLA latents vs float32 indexers) are handled
# by LMCache automatically — no extra config beyond the dtype flag.
#
# Usage:
#   ./DeepSeek-V4-Flash.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection: defaults to all 8 GPUs; GPU=list (global) / VLLM_GPU=list (this instance)
# Other env config: VLLM_PORT, MODEL_PATH, TP_SIZE, GPU_MEM_UTIL, LMCACHE_PORT,
#                   LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT

set -euo pipefail
USER_GPU=${GPU:-}   # capture before _common.sh applies its single-GPU default
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-${USER_GPU:-0,1,2,3,4,5,6,7}}   # TP=8 needs the whole node by default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-deepseek-ai/$MODEL_NAME}   # HF id — set MODEL_PATH to local weights
VLLM_PORT=${VLLM_PORT:-8000}
TP_SIZE=${TP_SIZE:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-1800}      # big model: weights load slowly
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-256}               # recipe uses lmcache's default chunk-size
EXTRA_VLLM_ARGS=(
    --tensor-parallel-size "$TP_SIZE"
    --enable-expert-parallel
    --kv-cache-dtype fp8_ds_mla
    --tokenizer-mode deepseek_v4
    --trust-remote-code
)

model_dispatch "$@"
