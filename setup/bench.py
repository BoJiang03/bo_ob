#!/usr/bin/env python3
"""bench.py — benchmark a vLLM+LMCache instance with `lmcache bench engine`,
per https://docs.lmcache.ai/getting_started/benchmarking.html and
    https://docs.lmcache.ai/cli/bench.html

Python port of bench.sh (pilot) — behavior-identical; the .sh stays until verified.
Drives the .py model scripts (e.g. ./Qwen3-8B.py), so the whole py family runs together.

Ensures the model instance is up (via its per-model script, which also brings up the
LMCache server), then runs the chosen workload against it.

The key knob is KV_VOLUME (--kv-cache-volume, GB): the benchmark derives its document
set from this working-set size. If the volume fits within the cache tiers (vLLM GPU KV
pool + lmcache server L1), everything hits after warmup; if it exceeds capacity, LRU
eviction kicks in and cache misses appear.

Usage:
  ./bench.py [extra `lmcache bench engine` args...]
e.g.
  ./bench.py                                # long-doc-qa over a 10GB working set
  KV_VOLUME=60 ./bench.py                   # overflow GPU pool + 20GB L1 -> misses
  WORKLOAD=multi-round-chat ./bench.py --mrc-qps 2 --mrc-duration 120
  MODEL=Qwen3.5-9B ./bench.py               # bench another managed model
  NO_LMCACHE=1 TOKENS_PER_GB=7281 ./bench.py   # baseline: vLLM without LMCache

NO_LMCACHE=1 starts the instance without the LMCache connector (plain vLLM baseline,
still with vLLM's own GPU prefix cache). There is no lmcache server to auto-detect
tokens-per-GB from, so TOKENS_PER_GB is required — take config.tokens_per_gb_kvcache
from any previous with-LMCache bench_summary.json of the same model.

Env config: MODEL (default Qwen3-8B; needs a matching ./<MODEL>.py), WORKLOAD,
  KV_VOLUME (GB), VLLM_PORT, LMCACHE_HTTP_PORT, OUT_ROOT, OUT_DIR,
  NO_LMCACHE, TOKENS_PER_GB
Results (CSV + JSON summary) land in bench-results/<timestamp>-<model>-<workload>-<vol>/
"""

import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

MODEL = os.environ.get("MODEL", "Qwen3-8B")
# the MP server registers models by path, so --model must be the exact local path the
# model script serves (see _common.py); look in the known weight roots
MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    for d in ("/dev/shm/models", "/data1/bo/models"):
        if os.path.exists(f"{d}/{MODEL}"):
            MODEL_PATH = f"{d}/{MODEL}"
            break
if not MODEL_PATH:
    sys.exit(f"ERROR: no local weights found for {MODEL} — set MODEL_PATH")

WORKLOAD = os.environ.get("WORKLOAD", "long-doc-qa")
KV_VOLUME = os.environ.get("KV_VOLUME", "10")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
LMCACHE_HTTP_PORT = os.environ.get("LMCACHE_HTTP_PORT", "8080")   # tokens-per-GB autodetect
OUT_ROOT = os.environ.get("OUT_ROOT", str(c.COMMON_DIR / "bench-results"))

NO_LMCACHE = os.environ.get("NO_LMCACHE", "0")
TOKENS_PER_GB = os.environ.get("TOKENS_PER_GB", "")

model_script = c.COMMON_DIR / f"{MODEL}.py"
if not os.access(model_script, os.X_OK):
    sys.exit(f"ERROR: no model script {model_script}")

cache_args = ["--lmcache-url", f"http://localhost:{LMCACHE_HTTP_PORT}"]
if NO_LMCACHE == "1":
    if not TOKENS_PER_GB:
        print("ERROR: NO_LMCACHE=1 needs TOKENS_PER_GB (no lmcache server to auto-detect from;",
              file=sys.stderr)
        print("       take config.tokens_per_gb_kvcache from a with-LMCache bench_summary.json)",
              file=sys.stderr)
        sys.exit(1)
    cache_args = ["--tokens-per-gb-kvcache", TOKENS_PER_GB]

# idempotent: no-op if the instance is already running — but it does NOT check which
# mode the running instance was started in; bench-compare.py stops it between modes
env = os.environ.copy()
env["NO_LMCACHE"] = NO_LMCACHE
c.run_or_die([sys.executable, str(model_script), "start"], env=env)

suffix = "-nolmcache" if NO_LMCACHE == "1" else ""
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.environ.get("OUT_DIR") or f"{OUT_ROOT}/{ts}-{MODEL}-{WORKLOAD}-{KV_VOLUME}GB{suffix}"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

lmcache_state = "no" if NO_LMCACHE == "1" else "yes"
print(f"bench: workload={WORKLOAD} model={MODEL} kv-cache-volume={KV_VOLUME}GB "
      f"lmcache={lmcache_state} -> {OUT_DIR}")
c.run_or_die(["lmcache", "bench", "engine",
              "--engine-url", f"http://localhost:{VLLM_PORT}",
              *cache_args,
              "--model", MODEL_PATH,
              "--workload", WORKLOAD,
              "--kv-cache-volume", KV_VOLUME,
              "--no-interactive",
              "--json",
              "--output-dir", OUT_DIR,
              *sys.argv[1:]])
print(f"bench: results in {OUT_DIR}")
