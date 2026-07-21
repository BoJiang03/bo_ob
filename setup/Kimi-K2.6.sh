#!/usr/bin/env bash
# Kimi-K2.6.sh — vLLM instance for moonshotai/Kimi-K2.6 (1T-param MoE, 32B active,
# native multimodal, DeepSeek-V3-style MLA attention) bound to the shared LMCache server.
#
# *** NEW RECIPE *** — https://docs.lmcache.ai/recipes/ has no Kimi entry yet.
# Derived from the DeepSeek-V4-Flash recipe by keeping the MoE/MLA parts and dropping
# the V4-only ones; rationale (from the model's config.json + vllm 0.25.1 registry):
#   - vLLM arch KimiK25ForConditionalGeneration wraps a DeepseekV3ForCausalLM text tower
#     (model_type kimi_k2, MLA with kv_lora_rank 512, 61 layers, 384 experts / 8 active)
#   - standard MLA KV caching: LMCache stores the per-layer compressed latents
#     (~70KB/token — far smaller than GQA models). NO --kv-cache-dtype fp8_ds_mla and
#     NO --tokenizer-mode deepseek_v4: those exist for DeepSeek-V4's sparse-attention
#     indexer, which K2.6 does not have.
#   - --trust-remote-code: tokenizer/processor ship custom code (tiktoken-based)
#   - native low-bit checkpoint is ~554 GiB -> --tensor-parallel-size 8 on 8x96GB
#     (~69 GiB weights per GPU) with --enable-expert-parallel to spread the 384 experts
#   - chunk-size 256: uniform-attention MLA default, same as the DeepSeek recipe
#   - context is 256K native; set MAX_MODEL_LEN to trade KV pool headroom for length
#
# Usage:
#   ./Kimi-K2.6.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection: needs the whole node by default; GPU=list (global) / VLLM_GPU=list
# Other env config: VLLM_PORT, MODEL_PATH, TP_SIZE, GPU_MEM_UTIL, MAX_MODEL_LEN,
#                   LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT

set -euo pipefail
USER_GPU=${GPU:-}   # capture before _common.sh applies its single-GPU default
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-${USER_GPU:-0,1,2,3,4,5,6,7}}   # TP=8 needs the whole node by default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-/data1/bo/models/$MODEL_NAME}
VLLM_PORT=${VLLM_PORT:-8000}
TP_SIZE=${TP_SIZE:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-3600}      # 554 GiB of weights load slowly
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-256}
EXTRA_VLLM_ARGS=(
    --tensor-parallel-size "$TP_SIZE"
    --enable-expert-parallel
    --trust-remote-code
)
[[ -n ${MAX_MODEL_LEN:-} ]] && EXTRA_VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

model_dispatch "$@"
