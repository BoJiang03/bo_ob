#!/usr/bin/env python3
"""Qwen3-8B.py — vLLM instance for Qwen3-8B (dense model) bound to the shared LMCache
server, per https://docs.lmcache.ai/getting_started/quickstart.html

Python port of Qwen3-8B.sh (pilot) — behavior-identical; the .sh stays until verified.

The filename IS the model name: this script serves /dev/shm/models/Qwen3-8B.
To manage another model, copy it to <Model-Name>.py (and give it its own VLLM_PORT).
`start` auto-starts the LMCache server (with this model's chunk-size) if it isn't up;
`stop` stops only this vLLM instance — the server and its cache stay alive.

Usage:
  ./Qwen3-8B.py {start|stop|restart|status|logs [-f]}

GPU selection:
  GPU=N        — global default, applies to this instance AND an auto-started server
  VLLM_GPU=N   — this vLLM instance only (e.g. second instance on another card)
  LMCACHE_GPU  — consumed by lmcache-ctl if `start` has to auto-start the server
e.g. second instance sharing the GPU-7 server:  VLLM_GPU=4 VLLM_PORT=8001 ./Qwen3-8B.py start

Other env config: VLLM_PORT, MODEL_PATH, GPU_MEM_UTIL, LMCACHE_PORT, LMCACHE_CHUNK_SIZE,
                  VENV, VLLM_START_TIMEOUT
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

gpu = os.environ.get("VLLM_GPU") or c.GPU   # instance GPU: VLLM_GPU beats the global default
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

name = Path(__file__).stem
cfg = c.ModelConfig(
    model_name=name,
    model_path=os.environ.get("MODEL_PATH", f"/dev/shm/models/{name}"),
    vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
    gpu=gpu,
    gpu_mem_util=os.environ.get("GPU_MEM_UTIL", "0.4"),
    start_timeout=int(os.environ.get("VLLM_START_TIMEOUT", "600")),
    need_chunk=int(os.environ.get("LMCACHE_CHUNK_SIZE", "16")),   # dense model: free choice
    extra_vllm_args=[],
)

c.model_dispatch(cfg, sys.argv[1:])
