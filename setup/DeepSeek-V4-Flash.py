#!/usr/bin/env python3
"""DeepSeek-V4-Flash.py — vLLM instance for DeepSeek-V4-Flash (hybrid model: compressed
MLA latents + sparse-attention indexers) bound to the shared LMCache server,
per https://docs.lmcache.ai/recipes/deepseek_v4_flash.html

Python port of DeepSeek-V4-Flash.sh (pilot) — behavior-identical; the .sh stays until verified.

*** UNTESTED ON THIS NODE ***  Faithful to the recipe; validate first.
Weights are fp8, ~148 GiB (local copy under /data1/bo/models). The recipe's reference
setup is TP=8; on a partially free node TP_SIZE=2 fits (74 GiB weights + KV per GPU).

Recipe requirements baked in below:
  --kv-cache-dtype fp8_ds_mla / --tokenizer-mode deepseek_v4   mandatory for this model
  --enable-expert-parallel        distributes MoE experts across TP ranks
  --tensor-parallel-size 8        adjust TP_SIZE to your hardware
  vLLM version: use a tagged release, NOT the dev branch (fp4 MoE expert misdispatching)
  lmcache server: recipe uses default chunk-size 256 and --l1-size-gb 100
    (pass L1_SIZE_GB=100 when this script auto-starts the server)
The interleaved KV cache groups (fp8/uint8 MLA latents vs float32 indexers) are handled
by LMCache automatically — no extra config beyond the dtype flag.

Usage:
  ./DeepSeek-V4-Flash.py {start|stop|restart|status|logs [-f]}

GPU selection: defaults to all 8 GPUs; GPU=list (global) / VLLM_GPU=list (this instance)
Other env config: VLLM_PORT, MODEL_PATH, TP_SIZE, GPU_MEM_UTIL, LMCACHE_PORT,
                  LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT
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
cfg = c.ModelConfig(
    model_name=name,
    model_path=os.environ.get("MODEL_PATH", f"/data1/bo/models/{name}"),
    vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
    gpu=gpu,
    gpu_mem_util=os.environ.get("GPU_MEM_UTIL", "0.9"),
    start_timeout=int(os.environ.get("VLLM_START_TIMEOUT", "1800")),   # big model: slow load
    need_chunk=int(os.environ.get("LMCACHE_CHUNK_SIZE", "256")),   # recipe uses lmcache default
    extra_vllm_args=[
        "--tensor-parallel-size", tp_size,
        "--enable-expert-parallel",
        "--kv-cache-dtype", "fp8_ds_mla",
        "--tokenizer-mode", "deepseek_v4",
        "--trust-remote-code",
    ],
)

c.model_dispatch(cfg, sys.argv[1:])
