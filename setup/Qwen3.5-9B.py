#!/usr/bin/env python3
"""Qwen3.5-9B.py — vLLM instance for Qwen3.5-9B (linear-attention / hybrid-mamba model)
bound to the shared LMCache server, per https://docs.lmcache.ai/recipes/qwen3_5.html

Python port of Qwen3.5-9B.sh (pilot) — behavior-identical; the .sh stays until verified.

Recipe requirements baked in below:
  --mamba-cache-mode align       mandatory for mamba prefix caching (marked experimental)
  --enable-prefix-caching
  LMCache chunk-size == vLLM unified block size = 528 for this model
    (probed 2026-07-21 from vllm 0.25.1 startup log: "Setting attention block size to
     528 tokens to ensure that attention page size is >= mamba page size";
     recipe reference values: Qwen3.5-0.8B=544, Qwen3.6-27B=784)
  --max-num-batched-tokens = 2*528-1 = 1055 (exactly N would serialize prefill/decode)
  --max-num-seqs lowered: each decode seq needs one mamba cache block; at mem-util 0.4
    only ~985 blocks exist and the vLLM default of 1024 aborts startup
Recipe caveats: generation is NOT bit-exact between cached and fresh runs; cache cannot
be shared across engines with different attention backends.

Usage:
  ./Qwen3.5-9B.py {start|stop|restart|status|logs [-f]}

GPU selection: GPU=N (global) / VLLM_GPU=N (this instance) / LMCACHE_GPU=N (server)
Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, MAX_NUM_SEQS, UNIFIED_BLOCK,
                  LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV, VLLM_START_TIMEOUT
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

gpu = os.environ.get("VLLM_GPU") or c.GPU   # instance GPU: VLLM_GPU beats the global default
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

name = Path(__file__).stem
# vLLM's unified (attention/mamba) block size for this model
unified_block = int(os.environ.get("UNIFIED_BLOCK", "528"))
max_num_seqs = os.environ.get("MAX_NUM_SEQS", "256")
cfg = c.ModelConfig(
    model_name=name,
    model_path=os.environ.get("MODEL_PATH", f"/dev/shm/models/{name}"),
    vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
    gpu=gpu,
    gpu_mem_util=os.environ.get("GPU_MEM_UTIL", "0.4"),
    start_timeout=int(os.environ.get("VLLM_START_TIMEOUT", "600")),
    need_chunk=int(os.environ.get("LMCACHE_CHUNK_SIZE", str(unified_block))),
    extra_vllm_args=[
        "--enable-prefix-caching",
        "--mamba-cache-mode", "align",
        "--max-num-batched-tokens", str(2 * unified_block - 1),
        "--max-num-seqs", max_num_seqs,
    ],
)

c.model_dispatch(cfg, sys.argv[1:])
