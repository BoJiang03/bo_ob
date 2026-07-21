#!/usr/bin/env bash
# Qwen3.6-35B-A3B.sh — vLLM instance for Qwen3.6-35B-A3B (hybrid linear-attention MoE,
# 35B total / 3B active) bound to the shared LMCache server,
# per https://docs.lmcache.ai/recipes/qwen3_5.html (Qwen3.5/3.6 series recipe)
#
# Recipe requirements baked in below (same family rules as Qwen3.5-9B.sh):
#   --mamba-cache-mode align       mandatory for mamba prefix caching (experimental)
#   --enable-prefix-caching
#   LMCache chunk-size == vLLM unified block size (UNIFIED_BLOCK below)
#     Recipe reference values: Qwen3.5-0.8B=544, Qwen3.6-27B=784. The 35B-A3B value is
#     NOT documented: probed 2026-07-21 on vllm 0.25.1 -> 1056 (same at TP=1 and TP=2).
#     Verify on new vllm versions from the log line
#     "Setting attention block size to N tokens ..." and set UNIFIED_BLOCK accordingly.
#   --max-num-batched-tokens = 2N-1 (exactly N would serialize prefill/decode)
#   --max-num-seqs lowered: each decode seq needs one mamba cache block
# MoE notes: 3B active params, single-GPU friendly (66 GiB bf16 weights); no
# expert-parallel flags needed at TP=1.
# Recipe caveats: generation is NOT bit-exact between cached and fresh runs; cache
# cannot be shared across engines with different attention backends.
#
# Usage:
#   ./Qwen3.6-35B-A3B.sh {start|stop|restart|status|logs [-f]}
#
# GPU selection: GPU=N (global) / VLLM_GPU=N (this instance) / LMCACHE_GPU=N (server)
# Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, MAX_NUM_SEQS, UNIFIED_BLOCK,
#                   LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU=${VLLM_GPU:-$GPU}   # instance GPU: VLLM_GPU beats the global GPU default
export CUDA_VISIBLE_DEVICES=$GPU

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_PATH=${MODEL_PATH:-/data1/bo/models/$MODEL_NAME}
VLLM_PORT=${VLLM_PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}       # 66 GiB weights on a 96GB card needs most of it
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-900}
UNIFIED_BLOCK=${UNIFIED_BLOCK:-1056}    # probed; see header
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
NEED_CHUNK=${LMCACHE_CHUNK_SIZE:-$UNIFIED_BLOCK}
EXTRA_VLLM_ARGS=(
    --enable-prefix-caching
    --mamba-cache-mode align
    --max-num-batched-tokens $((2 * UNIFIED_BLOCK - 1))
    --max-num-seqs "$MAX_NUM_SEQS"
)
# optional TP: 66 GiB of weights don't leave KV room on a single shared GPU
[[ -n ${TP_SIZE:-} ]] && EXTRA_VLLM_ARGS+=(--tensor-parallel-size "$TP_SIZE")

model_dispatch "$@"
