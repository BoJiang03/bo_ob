#!/usr/bin/env python3
"""Qwen3.6-35B-A3B.py — vLLM instance for Qwen3.6-35B-A3B (hybrid linear-attention MoE,
35B total / 3B active) bound to the shared LMCache server,
per https://docs.lmcache.ai/recipes/qwen3_5.html (Qwen3.5/3.6 series recipe)

Python port of Qwen3.6-35B-A3B.sh (pilot) — behavior-identical; the .sh stays until verified.

Recipe requirements baked in below (same family rules as Qwen3.5-9B.py):
  --mamba-cache-mode align       mandatory for mamba prefix caching (experimental)
  --enable-prefix-caching
  LMCache chunk-size == vLLM unified block size (UNIFIED_BLOCK below)
    Recipe reference values: Qwen3.5-0.8B=544, Qwen3.6-27B=784. The 35B-A3B value is
    NOT documented: probed 2026-07-21 on vllm 0.25.1 -> 1056 (same at TP=1 and TP=2).
    Verify on new vllm versions from the log line
    "Setting attention block size to N tokens ..." and set UNIFIED_BLOCK accordingly.
  --max-num-batched-tokens = 2N-1 (exactly N would serialize prefill/decode)
  --max-num-seqs lowered: each decode seq needs one mamba cache block
MoE notes: 3B active params, single-GPU friendly (66 GiB bf16 weights); no
expert-parallel flags needed at TP=1.
Recipe caveats: generation is NOT bit-exact between cached and fresh runs; cache
cannot be shared across engines with different attention backends.

Usage:
  ./Qwen3.6-35B-A3B.py {start|stop|restart|status|logs [-f]}

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
unified_block = int(os.environ.get("UNIFIED_BLOCK", "1056"))   # probed; see header
max_num_seqs = os.environ.get("MAX_NUM_SEQS", "256")
extra = [
    "--enable-prefix-caching",
    "--mamba-cache-mode", "align",
    "--max-num-batched-tokens", str(2 * unified_block - 1),
    "--max-num-seqs", max_num_seqs,
]
# optional TP: 66 GiB of weights don't leave KV room on a single shared GPU
if os.environ.get("TP_SIZE"):
    extra += ["--tensor-parallel-size", os.environ["TP_SIZE"]]

cfg = c.ModelConfig(
    model_name=name,
    model_path=os.environ.get("MODEL_PATH", f"/data1/bo/models/{name}"),
    vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
    gpu=gpu,
    gpu_mem_util=os.environ.get("GPU_MEM_UTIL", "0.9"),   # 66 GiB weights on a 96GB card
    start_timeout=int(os.environ.get("VLLM_START_TIMEOUT", "900")),
    need_chunk=int(os.environ.get("LMCACHE_CHUNK_SIZE", str(unified_block))),
    extra_vllm_args=extra,
)

c.model_dispatch(cfg, sys.argv[1:])
