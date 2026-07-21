#!/usr/bin/env bash
# check-rdma.sh — zero-GPU feasibility probe for the P2P KV-cache-sharing demo (Task 3).
#
# P2P's data plane is NIXL over RDMA (RoCE/InfiniBand). Before spending GPU windows on a
# 2-node demo, this checks the hard prerequisites on BOTH nodes:
#   1. an ACTIVE RDMA link (mlx5 RoCE) on the cross-node NIC
#   2. the nixl python package importable in the venv
#   3. same subnet + passwordless ssh reachability
# It touches no GPU and starts nothing — pure inspection. Prints a go/no-go verdict.
#
# Usage:  ./check-rdma.sh              # probe this node + REMOTE_HOST (default rtx-026)
#         REMOTE_HOST=172.16.176.28 ./check-rdma.sh
# Env: REMOTE_HOST, VENV, RDMA_NIC

set -uo pipefail                        # no -e: run every probe even if some fail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST=${REMOTE_HOST:-172.16.176.28}
VENV=${VENV:-/home/bo/lmcache/.venv}
RDMA_NIC=${RDMA_NIC:-enp41s0f0np0}
OUT_DIR=${OUT_DIR:-$HERE/bench-results/p2p}
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/check-rdma-$(date +%Y%m%d-%H%M%S).txt"

# identical probe run on each node (self-defaults VENV/NIC so it works over `ssh bash -s`)
read -r -d '' PROBE <<'PROBE_EOF' || true
VENV=${VENV:-/home/bo/lmcache/.venv}
NIC=${NIC:-enp41s0f0np0}
echo "host       : $(hostname)"
echo "nic $NIC IP: $(ip -4 addr show "$NIC" 2>/dev/null | awk '/inet /{print $2}' | head -1)"
echo "rdma link  :"
rdma link 2>/dev/null | sed 's/^/    /' || echo "    (no rdma tool)"
echo "uverbs dev : $(ls /dev/infiniband/uverbs* 2>/dev/null | tr '\n' ' ' || echo none)"
echo "nixl       : $("$VENV/bin/python" -c 'import nixl, importlib.metadata as m; print("OK", m.version("nixl"))' 2>&1 | tail -1)"
echo "ucx        : $(command -v ucx_info >/dev/null 2>&1 && ucx_info -v 2>/dev/null | head -1 || echo 'ucx_info not found')"
echo "ACTIVE_RDMA_LINKS=$(rdma link 2>/dev/null | grep -c 'state ACTIVE')"
PROBE_EOF

run_local()  { VENV="$VENV" NIC="$RDMA_NIC" bash -c "$PROBE" 2>&1; }
run_remote() { timeout 20 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 \
                 "$REMOTE_HOST" "VENV='$VENV' NIC='$RDMA_NIC' bash -s" <<<"$PROBE" 2>&1; }

{
echo "==================== P2P RDMA feasibility probe ===================="
echo "date: $(date -Is)   remote: $REMOTE_HOST   nic: $RDMA_NIC"
echo
echo "-------- LOCAL ($(hostname)) --------"
LOCAL_OUT=$(run_local); echo "$LOCAL_OUT"
echo
echo "-------- REMOTE ($REMOTE_HOST) --------"
REMOTE_OUT=$(run_remote); REMOTE_RC=$?
if (( REMOTE_RC != 0 )) && [[ -z $REMOTE_OUT ]]; then
    echo "(ssh to $REMOTE_HOST failed rc=$REMOTE_RC — passwordless ssh not set up?)"
fi
echo "$REMOTE_OUT"
echo
echo "-------- cross-node reachability --------"
ping -c1 -W2 "$REMOTE_HOST" >/dev/null 2>&1 && echo "ping $REMOTE_HOST: OK" || echo "ping $REMOTE_HOST: FAIL"

# ---- verdict ----
la=$(grep -oP 'ACTIVE_RDMA_LINKS=\K[0-9]+' <<<"$LOCAL_OUT"  | head -1); la=${la:-0}
ra=$(grep -oP 'ACTIVE_RDMA_LINKS=\K[0-9]+' <<<"$REMOTE_OUT" | head -1); ra=${ra:-0}
ln=$(grep -c '^nixl *: OK' <<<"$LOCAL_OUT")
rn=$(grep -c '^nixl *: OK' <<<"$REMOTE_OUT")
echo
echo "==================== VERDICT ===================="
printf '  %-26s local=%s  remote=%s\n' "ACTIVE RDMA link(s)" "$la" "$ra"
printf '  %-26s local=%s  remote=%s\n' "nixl importable"     "$([[ $ln -ge 1 ]] && echo yes || echo NO)" "$([[ $rn -ge 1 ]] && echo yes || echo NO)"
echo
if (( la >= 1 && ra >= 1 )) && [[ $ln -ge 1 && $rn -ge 1 ]]; then
    echo "  => GO: both nodes have active RoCE + nixl. P2P demo can run."
elif (( la >= 1 && ra >= 1 )); then
    echo "  => ALMOST: RoCE present on both, but nixl missing on a node."
    echo "     Install into that node's venv (uv, per Bo's rule), matching local nixl:"
    echo "       uv pip install --python $VENV/bin/python nixl==$("$VENV/bin/python" -c 'import importlib.metadata as m;print(m.version("nixl"))' 2>/dev/null)"
    echo "     then re-run this probe."
else
    echo "  => NO-GO (yet): missing an ACTIVE RDMA link on a node — P2P transfer channel needs RoCE/IB."
fi
echo "================================================="
} | tee "$REPORT"
echo
echo "report saved: $REPORT"
