#!/usr/bin/env python3
"""lmcache-ctl.py — start / stop / manage the shared LMCache MP server on this node.

Python port of lmcache-ctl.sh (pilot) — behavior-identical, shares run/ and
run/lmcache-server.conf with the .sh scripts; the .sh stays canonical until verified.

vLLM instances are managed separately by the per-model scripts (e.g. ./Qwen3-8B.py),
which connect to this server and auto-start it when needed — passing the chunk-size
their model requires. The running server's config is recorded in run/lmcache-server.conf
so model scripts can refuse to attach to a server with a mismatched chunk-size.
Stop the server only when no instance is using it — its cache (L1 CPU memory) dies with it.

Usage:
  ./lmcache-ctl.py {start|stop|restart|status|logs [-f]}

GPU visibility: the MP server resolves the device UUIDs that vLLM instances send
when registering their KV caches — it MUST be able to see every GPU any attached
vLLM instance runs on, or registration dies with "Device UUID ... not found".
Its own idle GPU footprint is small (~0.5GB), so the default is to leave it
unrestricted (sees all GPUs). Set LMCACHE_GPU only if you really want to pin it.

Config via env, e.g.: CHUNK_SIZE=528 L1_SIZE_GB=40 ./lmcache-ctl.py start
  LMCACHE_GPU (default: all GPUs visible), LMCACHE_PORT, L1_SIZE_GB, CHUNK_SIZE, VENV
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c

GPU = os.environ.get("LMCACHE_GPU", "")   # empty = unrestricted (server sees all GPUs)
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    GPU = "all"
L1_SIZE_GB = os.environ.get("L1_SIZE_GB", "20")   # CPU pinned-memory cache size
CHUNK_SIZE = os.environ.get("CHUNK_SIZE", "16")   # must match what the attached models need
                                                  # (dense: free choice; mamba/linear-attention
                                                  # models: must equal vLLM's unified block size)
HTTP_PORT = os.environ.get("LMCACHE_HTTP_PORT", "8080")   # mgmt/metrics HTTP port; if taken by a
                                                          # foreign process the whole server dies
                                                          # seconds after "up" (bind Errno 98 kills
                                                          # it AFTER the ZMQ port passed the check)
PIDFILE = c.RUN_DIR / "lmcache-server.pid"
LOGFILE = c.LOG_DIR / "lmcache-server.log"


def start():
    if c.alive(PIDFILE):
        print(f"lmcache server: already running (pid {c.read_pid(PIDFILE)}, chunk-size {c.server_chunk_size()})")
        return True
    if c.port_open(c.LMCACHE_PORT):
        print(f"ERROR: port {c.LMCACHE_PORT} already in use by another process — not starting",
              file=sys.stderr)
        return False
    print(f"lmcache server: starting on port {c.LMCACHE_PORT} (GPU {GPU}, L1 {L1_SIZE_GB}GB, chunk {CHUNK_SIZE})...")
    c.launch_detached(
        ["lmcache", "server",
         "--host", "localhost", "--port", str(c.LMCACHE_PORT), "--http-port", HTTP_PORT,
         "--l1-size-gb", L1_SIZE_GB, "--eviction-policy", "LRU", "--chunk-size", CHUNK_SIZE],
        LOGFILE, PIDFILE)
    if not c.wait_port(c.LMCACHE_PORT, 120):
        print(f"ERROR: lmcache server did not open port {c.LMCACHE_PORT} in 120s; last log lines:",
              file=sys.stderr)
        c.tail_log(LOGFILE, 20, stream=sys.stderr)
        return False
    time.sleep(3)   # the HTTP mgmt server binds after the ZMQ port; a bind failure kills the process
    if not c.alive(PIDFILE):
        print(f"ERROR: lmcache server died right after opening port {c.LMCACHE_PORT} "
              f"(HTTP port {HTTP_PORT} taken?); last log lines:", file=sys.stderr)
        c.tail_log(LOGFILE, 10, stream=sys.stderr)
        PIDFILE.unlink(missing_ok=True)
        return False
    c.SERVER_CONF.write_text(
        f"CHUNK_SIZE={CHUNK_SIZE}\nL1_SIZE_GB={L1_SIZE_GB}\nGPU={GPU}\nPORT={c.LMCACHE_PORT}\n")
    print(f"lmcache server: up (pid {c.read_pid(PIDFILE)}, log {LOGFILE})")
    return True


def status():
    if c.alive(PIDFILE):
        state = f"port {c.LMCACHE_PORT} open" if c.port_open(c.LMCACHE_PORT) \
            else f"port {c.LMCACHE_PORT} NOT responding"
        print(f"lmcache server: running (pid {c.read_pid(PIDFILE)}, {state}, "
              f"chunk-size {c.server_chunk_size()}, L1 {c.server_conf_get('L1_SIZE_GB')}GB)")
    elif c.port_open(c.LMCACHE_PORT):
        print(f"lmcache server: no pidfile, but port {c.LMCACHE_PORT} is in use (foreign process?)")
    else:
        print("lmcache server: stopped")
    c.show_gpu(GPU)


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "start":
        sys.exit(0 if start() else 1)
    elif cmd == "stop":
        c.stop_one("lmcache-server", PIDFILE)
        c.SERVER_CONF.unlink(missing_ok=True)
    elif cmd == "restart":
        c.stop_one("lmcache-server", PIDFILE)
        c.SERVER_CONF.unlink(missing_ok=True)
        sys.exit(0 if start() else 1)
    elif cmd == "status":
        status()
    elif cmd == "logs":
        c.follow_or_tail(LOGFILE, len(argv) > 1 and argv[1] == "-f")
    else:
        c.print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
