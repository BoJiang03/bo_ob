"""_common.py — shared config, helpers, and the generic vLLM-instance engine.

Python port of _common.sh (pilot). Behavior mirrors the .sh 1:1; run/, logs/ and
run/lmcache-server.conf are shared with the .sh scripts, so the two families are
interoperable — each can see and stop what the other started. The .sh files stay
canonical until the .py versions are verified against them on a GPU window.

Imported by lmcache-ctl.py and the per-model scripts; not executable.
A per-model script only needs to build a ModelConfig (MODEL_NAME, MODEL_PATH,
VLLM_PORT, GPU_MEM_UTIL, NEED_CHUNK, EXTRA_VLLM_ARGS...) and call
model_dispatch(cfg, sys.argv[1:]).
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
VENV = os.environ.get("VENV", "/home/bo/lmcache/.venv")
GPU = os.environ.get("GPU", "7")    # global default GPU (shared box: 0-6 used by others);
                                    # each script may override per role, then pins CUDA_VISIBLE_DEVICES itself
LMCACHE_PORT = int(os.environ.get("LMCACHE_PORT", "5556"))  # 5555 is taken by someone else on this node

LOG_DIR = COMMON_DIR / "logs"
RUN_DIR = COMMON_DIR / "run"
SERVER_CONF = RUN_DIR / "lmcache-server.conf"   # written by lmcache-ctl on start
LOG_DIR.mkdir(exist_ok=True)
RUN_DIR.mkdir(exist_ok=True)

# MP mode requires identical hashing across all processes; ninja on PATH for flashinfer JIT
os.environ["PATH"] = f"{VENV}/bin:/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["PYTHONHASHSEED"] = "0"


def read_pid(pidfile):
    try:
        return int(Path(pidfile).read_text().strip())
    except (OSError, ValueError):
        return None


# Popen objects for processes THIS process launched: a dead child stays a zombie
# (kill -0 still succeeds) until reaped, so pid_alive must poll() it first.
# The sh version never had this problem — bash auto-reaps background children.
_children = {}


def pid_alive(pid):
    proc = _children.get(pid)
    if proc is not None and proc.poll() is not None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:   # ESRCH: gone; EPERM: not ours — either way not our instance
        return False


def alive(pidfile):
    pid = read_pid(pidfile)
    return pid is not None and pid_alive(pid)


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            return True
    except OSError:
        return False


def wait_port(port, timeout_s):
    for _ in range(int(timeout_s)):
        if port_open(port):
            return True
        time.sleep(1)
    return False


def http_ok(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def wait_http(url, timeout_s, pidfile):
    """Returns 0 = healthy, 1 = timeout, 2 = process died while we waited."""
    waited = 0
    while waited < timeout_s:
        if http_ok(url):
            return 0
        if not alive(pidfile):
            return 2
        time.sleep(2)
        waited += 2
    return 1


def _smi_lines(query, gpu, units=False):
    fmt = "csv,noheader" if units else "csv,noheader,nounits"
    cmd = ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"]
    if str(gpu) != "all":
        cmd += ["-i", str(gpu)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    except OSError:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def gpu_mem_warn(gpu, threshold_mib, message):
    """$gpu may be a comma list or "all"; checks the busiest card."""
    used = []
    for ln in _smi_lines("memory.used", gpu):
        try:
            used.append(int(ln))
        except ValueError:
            pass
    if used and max(used) > threshold_mib:
        print(f"WARN: GPU {gpu} already has {max(used)} MiB in use — {message}")


def show_gpu(gpu):
    for ln in _smi_lines("index,memory.used,memory.total", gpu, units=True):
        print(f"GPU {ln}")


def server_conf_get(key):
    """Reads one KEY=value from the running server's conf; None if unknown."""
    try:
        for line in SERVER_CONF.read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
    except OSError:
        pass
    return None


def server_chunk_size():
    return server_conf_get("CHUNK_SIZE")


def kill_tree(pid, sig):
    # setsid made pid a group leader — signalling the group takes the whole tree
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def stop_one(name, pidfile):
    pidfile = Path(pidfile)
    pid = read_pid(pidfile)
    if pid is None or not pid_alive(pid):
        print(f"{name}: not running")
        pidfile.unlink(missing_ok=True)
        return True
    print(f"{name}: stopping (pid {pid})...")
    kill_tree(pid, signal.SIGTERM)
    for _ in range(30):
        if not pid_alive(pid):
            break
        time.sleep(1)
    if pid_alive(pid):
        print(f"{name}: still alive after 30s, force killing")
        kill_tree(pid, signal.SIGKILL)
    pidfile.unlink(missing_ok=True)
    print(f"{name}: stopped")
    return True


def print_usage():
    """Prints the executed script's module docstring (its header)."""
    import __main__
    print((__main__.__doc__ or "").strip("\n"))


def tail_log(logfile, n, stream=None):
    subprocess.run(["tail", "-n", str(n), str(logfile)],
                   stdout=stream if stream is not None else None,
                   stderr=subprocess.DEVNULL)


def follow_or_tail(logfile, follow):
    try:
        if follow:
            subprocess.call(["tail", "-f", str(logfile)])
        else:
            subprocess.call(["tail", "-n", "50", str(logfile)])
    except KeyboardInterrupt:
        pass


def launch_detached(cmd, logfile, pidfile, env=None):
    """setsid + nohup equivalent: own session, log truncated, pid recorded."""
    with open(logfile, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                env=env if env is not None else os.environ.copy())
    _children[proc.pid] = proc
    Path(pidfile).write_text(f"{proc.pid}\n")
    return proc.pid


def run_or_die(cmd, env=None):
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        sys.exit(rc)


def ensure_server(need_chunk, who):
    """Auto-starts or validates the lmcache server for a consumer needing need_chunk."""
    if port_open(LMCACHE_PORT):
        chunk = server_chunk_size()
        if chunk and chunk.isdigit() and int(chunk) != int(need_chunk):
            print(f"ERROR: running lmcache server uses chunk-size {chunk} but {who} needs {need_chunk}.",
                  file=sys.stderr)
            print(f"       Restart it with: CHUNK_SIZE={need_chunk} {COMMON_DIR}/lmcache-ctl.py restart  (wipes its cache)",
                  file=sys.stderr)
            print("       or point this instance at a second server via LMCACHE_PORT.", file=sys.stderr)
            return False
        if not chunk:
            print(f"WARN: server on port {LMCACHE_PORT} not started by lmcache-ctl — cannot verify chunk-size (need {need_chunk})")
        return True
    print(f"{who}: lmcache server not up, starting it (chunk-size {need_chunk})...")
    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(need_chunk)
    return subprocess.run([sys.executable, str(COMMON_DIR / "lmcache-ctl.py"), "start"],
                          env=env).returncode == 0


# ---------- generic vLLM model-instance engine ----------

@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    vllm_port: int
    gpu: str
    gpu_mem_util: str
    start_timeout: int
    need_chunk: int
    extra_vllm_args: list = field(default_factory=list)

    @property
    def pidfile(self):
        return RUN_DIR / f"{self.model_name}.pid"

    @property
    def logfile(self):
        return LOG_DIR / f"{self.model_name}.log"


def model_start(cfg):
    if alive(cfg.pidfile):
        print(f"{cfg.model_name}: already running (pid {read_pid(cfg.pidfile)})")
        return True
    if cfg.model_path.startswith("/") and not os.path.exists(cfg.model_path):
        print(f"ERROR: model path {cfg.model_path} does not exist", file=sys.stderr)
        return False
    kv_args = []
    if os.environ.get("NO_LMCACHE", "0") == "1":
        print(f"{cfg.model_name}: NO_LMCACHE=1 — baseline mode, running WITHOUT LMCache")
    else:
        if not ensure_server(cfg.need_chunk, cfg.model_name):
            return False
        kv_cfg = json.dumps({
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": LMCACHE_PORT,
            },
        })
        kv_args = ["--kv-transfer-config", kv_cfg]
    if port_open(cfg.vllm_port):
        print(f"ERROR: port {cfg.vllm_port} already in use by another process — not starting",
              file=sys.stderr)
        return False
    gpu_mem_warn(cfg.gpu, 55000, f"vLLM (mem-util {cfg.gpu_mem_util}) may OOM on top of it")
    bound = f", bound to lmcache on port {LMCACHE_PORT}" if kv_args else ""
    print(f"{cfg.model_name}: starting vLLM on port {cfg.vllm_port} (GPU {cfg.gpu}, mem-util {cfg.gpu_mem_util}){bound}...")
    # VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache (needed for cache benchmarks)
    # both names accepted by the API: the short name for humans, the path for
    # `lmcache bench engine` (the MP server registers models under their path)
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    cmd = ["vllm", "serve", cfg.model_path,
           "--served-model-name", cfg.model_name, cfg.model_path,
           "--port", str(cfg.vllm_port),
           "--gpu-memory-utilization", str(cfg.gpu_mem_util),
           *[str(a) for a in cfg.extra_vllm_args],
           *kv_args]
    launch_detached(cmd, cfg.logfile, cfg.pidfile, env=env)
    print(f"{cfg.model_name}: waiting for http://localhost:{cfg.vllm_port}/v1/models (up to {cfg.start_timeout}s)...")
    rc = wait_http(f"http://localhost:{cfg.vllm_port}/v1/models", cfg.start_timeout, cfg.pidfile)
    if rc == 0:
        print(f"{cfg.model_name}: up (pid {read_pid(cfg.pidfile)}, log {cfg.logfile})")
        return True
    if rc == 2:
        print("ERROR: vLLM process died during startup; last log lines:", file=sys.stderr)
        tail_log(cfg.logfile, 30, stream=sys.stderr)
        cfg.pidfile.unlink(missing_ok=True)
        return False
    print(f"ERROR: vLLM not healthy after {cfg.start_timeout}s; check {cfg.logfile}", file=sys.stderr)
    return False


def model_status(cfg):
    if alive(cfg.pidfile):
        pid = read_pid(cfg.pidfile)
        if http_ok(f"http://localhost:{cfg.vllm_port}/v1/models"):
            print(f"{cfg.model_name}: running (pid {pid}), API healthy at http://localhost:{cfg.vllm_port}/v1")
        else:
            print(f"{cfg.model_name}: running (pid {pid}), but API on port {cfg.vllm_port} not responding (still starting, or wedged)")
    elif port_open(cfg.vllm_port):
        print(f"{cfg.model_name}: stopped (but port {cfg.vllm_port} is in use by another process)")
    else:
        print(f"{cfg.model_name}: stopped")
    if port_open(LMCACHE_PORT):
        chunk = server_chunk_size()
        extra = f" (chunk-size {chunk}; this model needs {cfg.need_chunk})" if chunk else ""
        print(f"lmcache server: reachable on port {LMCACHE_PORT}{extra}")
    else:
        print(f"lmcache server: NOT running ({COMMON_DIR}/lmcache-ctl.py start)")
    show_gpu(cfg.gpu)


def model_dispatch(cfg, argv):
    cmd = argv[0] if argv else ""
    if cmd == "start":
        sys.exit(0 if model_start(cfg) else 1)
    elif cmd == "stop":
        stop_one(cfg.model_name, cfg.pidfile)
    elif cmd == "restart":
        stop_one(cfg.model_name, cfg.pidfile)
        sys.exit(0 if model_start(cfg) else 1)
    elif cmd == "status":
        model_status(cfg)
    elif cmd == "logs":
        follow_or_tail(cfg.logfile, len(argv) > 1 and argv[1] == "-f")
    else:
        print_usage()
        sys.exit(1)
