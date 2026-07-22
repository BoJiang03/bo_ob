#!/usr/bin/env python3
"""Kimi-K2.6.py — vLLM instance for moonshotai/Kimi-K2.6 (1T-param MoE, 32B active,
native multimodal, DeepSeek-V3-style MLA attention) bound to the shared LMCache server.

Python port of Kimi-K2.6.sh (pilot) — behavior-identical; the .sh stays until verified.

*** NEW RECIPE *** — https://docs.lmcache.ai/recipes/ has no Kimi entry yet.
Derived from the DeepSeek-V4-Flash recipe by keeping the MoE/MLA parts and dropping
the V4-only ones; rationale (from the model's config.json + vllm 0.25.1 registry):
  - vLLM arch KimiK25ForConditionalGeneration wraps a DeepseekV3ForCausalLM text tower
    (model_type kimi_k2, MLA with kv_lora_rank 512, 61 layers, 384 experts / 8 active)
  - standard MLA KV caching: LMCache stores the per-layer compressed latents
    (~70KB/token — far smaller than GQA models). NO --kv-cache-dtype fp8_ds_mla and
    NO --tokenizer-mode deepseek_v4: those exist for DeepSeek-V4's sparse-attention
    indexer, which K2.6 does not have.
  - --trust-remote-code: tokenizer/processor ship custom code (tiktoken-based)
  - native low-bit checkpoint is ~554 GiB -> --tensor-parallel-size 8 on 8x96GB
    (~69 GiB weights per GPU) with --enable-expert-parallel to spread the 384 experts
  - chunk-size 256: uniform-attention MLA default, same as the DeepSeek recipe
  - context is 256K native; set MAX_MODEL_LEN to trade KV pool headroom for length

Usage:
  ./Kimi-K2.6.py {start|stop|restart|status|logs [-f]}

GPU selection: needs the whole node by default; GPU=list (global) / VLLM_GPU=list
Other env config: VLLM_PORT, MODEL_PATH, TP_SIZE, GPU_MEM_UTIL, MAX_MODEL_LEN,
                  LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

# TP=8 needs the whole node by default (GPU read from env, NOT _common's single-GPU default)
gpu = os.environ.get("VLLM_GPU") or os.environ.get("GPU") or "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

name = Path(__file__).stem
tp_size = os.environ.get("TP_SIZE", "8")
extra = [
    "--tensor-parallel-size", tp_size,
    "--enable-expert-parallel",
    "--trust-remote-code",
]
if os.environ.get("MAX_MODEL_LEN"):
    extra += ["--max-model-len", os.environ["MAX_MODEL_LEN"]]

cfg = c.ModelConfig(
    model_name=name,
    model_path=os.environ.get("MODEL_PATH", f"/data1/bo/models/{name}"),
    vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
    gpu=gpu,
    gpu_mem_util=os.environ.get("GPU_MEM_UTIL", "0.9"),
    start_timeout=int(os.environ.get("VLLM_START_TIMEOUT", "3600")),   # 554 GiB loads slowly
    need_chunk=int(os.environ.get("LMCACHE_CHUNK_SIZE", "256")),
    extra_vllm_args=extra,
)

c.model_dispatch(cfg, sys.argv[1:])
