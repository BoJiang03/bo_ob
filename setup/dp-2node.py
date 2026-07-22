#!/usr/bin/env python3
"""dp-2node.py — 2-node data-parallel deployment of a MoE model: one DP rank per node,
each rank bound to a LOCAL LMCache server.

Python port of dp-2node.sh (pilot) — behavior-identical; the .sh stays until verified.

Topology (why one lmcache server per node): LMCache MP mode shares KV tensors with
vLLM via CUDA IPC, which only works within one machine — a vLLM on another node
cannot attach to this node's server. Cross-node KV sharing is LMCache's distributed
tier (P2P / coordinator), separate from MP mode.

  rtx-1 (head, owns the API):   ./dp-2node.py start head
  rtx-2 (secondary, headless):  ./dp-2node.py start node2

The head runs vLLM's internal DP load balancer: requests to :$VLLM_PORT are routed
across both ranks. --enable-expert-parallel shards the MoE experts over the DP
ranks (wide-EP) — the reason DP deployments exist for MoE models. Both nodes must
use the same model, chunk-size, and PYTHONHASHSEED=0 (from _common.py).

Single-box simulation also works: run both roles on one machine with
DP_ADDRESS=127.0.0.1 and a different VLLM_GPU per role.

Usage:
  ./dp-2node.py {start|stop|restart|status} <head|node2>
  ./dp-2node.py logs <head|node2> [-f]

Env config:
  DP_ADDRESS   head node's reachable IP (default 172.16.176.27 = rtx-1 LAN)
  DP_RPC_PORT  DP rendezvous port on the head (default 13345)
  DP_MODEL     model name (default Qwen3-30B-A3B); weights auto-located in
               /dev/shm/models or /data1/bo/models, or set MODEL_PATH
  VLLM_GPU     GPU index for this node's rank (default 7)
  NCCL_IFACE   NIC for NCCL/GLOO cross-node traffic (default enp41s0f0np0);
               without it NCCL may pick docker0 and hang
  VLLM_PORT, GPU_MEM_UTIL, LMCACHE_PORT, LMCACHE_CHUNK_SIZE, VENV,
  VLLM_START_TIMEOUT
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

GPU = os.environ.get("VLLM_GPU", "7")
# DP engines re-derive their device from the local DP rank and stomp any
# CUDA_VISIBLE_DEVICES pin — rank 0 then always grabs physical GPU 0, which on a
# shared box is usually someone else's. --device-ids (below) is vLLM's supported
# way to pin DP ranks to physical GPUs; CUDA_VISIBLE_DEVICES must stay unset or
# the physical id in --device-ids can no longer be resolved.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

MODEL_NAME = os.environ.get("DP_MODEL", "Qwen3-30B-A3B")
MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    for d in ("/dev/shm/models", "/data1/bo/models"):
        if os.path.exists(f"{d}/{MODEL_NAME}"):
            MODEL_PATH = f"{d}/{MODEL_NAME}"
            break
if not MODEL_PATH:
    sys.exit(f"ERROR: no local weights found for {MODEL_NAME} — set MODEL_PATH")

DP_ADDRESS = os.environ.get("DP_ADDRESS", "172.16.176.27")
DP_RPC_PORT = os.environ.get("DP_RPC_PORT", "13345")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8100"))
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.8")
VLLM_START_TIMEOUT = int(os.environ.get("VLLM_START_TIMEOUT", "900"))
NEED_CHUNK = int(os.environ.get("LMCACHE_CHUNK_SIZE", "256"))   # Qwen3 MoE recipe default

# cross-node NCCL/GLOO must use the LAN NIC, not docker0/lo
NCCL_IFACE = os.environ.get("NCCL_IFACE", "enp41s0f0np0")
os.environ["NCCL_SOCKET_IFNAME"] = NCCL_IFACE
os.environ["GLOO_SOCKET_IFNAME"] = NCCL_IFACE

# how long the head waits at the rendezvous for the other rank (vLLM default is
# 900s, after which the head half-dies while still listening — a trap). 4h lets
# a pre-launched head stand by until a GPU window opens on the other node.
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "14400")
# ...and the torch TCPStore rendezvous has its OWN 1800s default ("Timed out
# ... 1/2 clients joined") -- raised via the two --*distributed-timeout-seconds
# args in DP_COMMON_ARGS below.
DP_WAIT_SECONDS = os.environ.get("DP_WAIT_SECONDS", "14400")

KV_CFG = json.dumps({
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.host": "tcp://localhost",
        "lmcache.mp.port": c.LMCACHE_PORT,
    },
})
DP_COMMON_ARGS = [
    "--served-model-name", MODEL_NAME, MODEL_PATH,
    "--device-ids", GPU,
    "--distributed-timeout-seconds", DP_WAIT_SECONDS,
    "--cpu-distributed-timeout-seconds", DP_WAIT_SECONDS,
    "--gpu-memory-utilization", GPU_MEM_UTIL,
    "--enable-expert-parallel",
    "--data-parallel-size", "2",
    "--data-parallel-size-local", "1",
    "--data-parallel-address", DP_ADDRESS,
    "--data-parallel-rpc-port", DP_RPC_PORT,
    "--kv-transfer-config", KV_CFG,
]


def files(role):
    return c.RUN_DIR / f"dp-{role}.pid", c.LOG_DIR / f"dp-{role}.log"


def dp_start(role):
    pf, lf = files(role)
    if c.alive(pf):
        print(f"dp-{role}: already running (pid {c.read_pid(pf)})")
        return True
    if not c.ensure_server(NEED_CHUNK, MODEL_NAME):
        return False
    c.gpu_mem_warn(GPU, 40000, f"DP rank (mem-util {GPU_MEM_UTIL}) may OOM on top of it")

    if role == "head":
        if c.port_open(VLLM_PORT):
            print(f"ERROR: port {VLLM_PORT} already in use — not starting", file=sys.stderr)
            return False
        print(f"dp-head: starting rank 0 (GPU {GPU}, api :{VLLM_PORT}, rendezvous {DP_ADDRESS}:{DP_RPC_PORT})...")
        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"
        c.launch_detached(["vllm", "serve", MODEL_PATH, "--port", str(VLLM_PORT),
                           *DP_COMMON_ARGS], lf, pf, env=env)
        print(f"dp-head: waiting for http://localhost:{VLLM_PORT}/v1/models (up to {VLLM_START_TIMEOUT}s)...")
        print("         NOTE: the API only comes up after node2 joins the rendezvous.")
        rc = c.wait_http(f"http://localhost:{VLLM_PORT}/v1/models", VLLM_START_TIMEOUT, pf)
        if rc == 0:
            print(f"dp-head: up (pid {c.read_pid(pf)}, log {lf})")
            return True
        if rc == 2:
            print("ERROR: head died during startup; last log lines:", file=sys.stderr)
            c.tail_log(lf, 30, stream=sys.stderr)
            pf.unlink(missing_ok=True)
            return False
        print(f"ERROR: head not healthy after {VLLM_START_TIMEOUT}s (node2 never joined?); check {lf}",
              file=sys.stderr)
        return False

    print(f"dp-node2: starting rank 1 (GPU {GPU}, headless, rendezvous {DP_ADDRESS}:{DP_RPC_PORT})...")
    c.launch_detached(["vllm", "serve", MODEL_PATH, "--headless",
                       "--data-parallel-start-rank", "1", *DP_COMMON_ARGS], lf, pf)
    # headless rank has no API; consider it up if it survives early init
    for _ in range(15):
        if not c.alive(pf):
            print("ERROR: node2 died during startup; last log lines:", file=sys.stderr)
            c.tail_log(lf, 30, stream=sys.stderr)
            pf.unlink(missing_ok=True)
            return False
        time.sleep(2)
    print(f"dp-node2: running (pid {c.read_pid(pf)}, log {lf}) — rendezvous/model load continue; "
          f"watch: {sys.argv[0]} logs node2 -f")
    return True


def dp_status(role):
    pf, _ = files(role)
    if c.alive(pf):
        print(f"dp-{role}: running (pid {c.read_pid(pf)})")
    else:
        print(f"dp-{role}: stopped")
    if role == "head":
        if c.http_ok(f"http://localhost:{VLLM_PORT}/v1/models"):
            print(f"api: healthy at http://localhost:{VLLM_PORT}/v1 (internal DP load balancer)")
        else:
            print(f"api: not responding on port {VLLM_PORT}")
    if c.port_open(c.LMCACHE_PORT):
        chunk = c.server_chunk_size()
        extra = f" (chunk-size {chunk}; this setup needs {NEED_CHUNK})" if chunk else ""
        print(f"local lmcache server: reachable on port {c.LMCACHE_PORT}{extra}")
    else:
        print(f"local lmcache server: NOT running ({c.COMMON_DIR}/lmcache-ctl.py start)")
    c.show_gpu(GPU)


def main(argv):
    cmd = argv[0] if len(argv) > 0 else ""
    role = argv[1] if len(argv) > 1 else ""
    if role not in ("head", "node2"):
        c.print_usage()
        sys.exit(1)
    pf, lf = files(role)
    if cmd == "start":
        sys.exit(0 if dp_start(role) else 1)
    elif cmd == "stop":
        c.stop_one(f"dp-{role}", pf)
    elif cmd == "restart":
        c.stop_one(f"dp-{role}", pf)
        sys.exit(0 if dp_start(role) else 1)
    elif cmd == "status":
        dp_status(role)
    elif cmd == "logs":
        c.follow_or_tail(lf, len(argv) > 2 and argv[2] == "-f")
    else:
        c.print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
