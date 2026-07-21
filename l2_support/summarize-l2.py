#!/usr/bin/env python3
"""summarize-l2.py — render a side-by-side table from `lmcache bench l2` JSON outputs.

Usage: summarize-l2.py <results_dir>

Reads every <adapter>.json in the dir (each the --format json --output of one
`lmcache bench l2` run) and prints a Markdown comparison of the Store / Load / Lookup
metrics.  Called by bench-l2.sh, but works standalone on any such directory.
"""
import glob
import json
import os
import sys


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  (skip {name}: {e})", file=sys.stderr)
            continue
        metrics = data.get("metrics", {})
        ops = {}
        for k, v in metrics.items():
            if k.startswith("op_") and isinstance(v, dict):
                ops[v.get("operation", k)] = v
        if ops:
            runs.append((name, metrics.get("config", {}), ops))
    return runs


def cell(op, key, fmt="{:.1f}"):
    if op is None or key not in op:
        return "-"
    try:
        return fmt.format(op[key])
    except (ValueError, TypeError):
        return str(op[key])


def md_table(header, rows):
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        print("usage: summarize-l2.py <results_dir>", file=sys.stderr)
        sys.exit(1)
    results_dir = sys.argv[1]
    runs = load_runs(results_dir)
    if not runs:
        print(f"no benchmark JSON found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = runs[0][1]
    lines = []
    lines.append("# L2 adapter comparison (`lmcache bench l2`)\n")
    lines.append(
        "workload: num_keys={num_keys} in_flight={in_flight} data_size_kb={data_size_kb} "
        "rounds={measurement_rounds} (warmup {warmup_rounds}) "
        "lookup_max_hit_rate={lookup_max_hit_rate}\n".format(**{
            "num_keys": cfg.get("num_keys", "?"),
            "in_flight": cfg.get("in_flight", "?"),
            "data_size_kb": cfg.get("data_size_kb", "?"),
            "measurement_rounds": cfg.get("measurement_rounds", "?"),
            "warmup_rounds": cfg.get("warmup_rounds", "?"),
            "lookup_max_hit_rate": cfg.get("lookup_max_hit_rate", "?"),
        })
    )

    # Store — throughput + latency
    header = ["adapter", "store MB/s", "store p50 ms", "store p99 ms", "store ops/s"]
    rows = []
    for name, _cfg, ops in runs:
        o = ops.get("Store")
        rows.append([name, cell(o, "throughput_avg_mbps"), cell(o, "duration_p50_ms", "{:.2f}"),
                     cell(o, "duration_p99_ms", "{:.2f}"), cell(o, "ops_per_sec_avg", "{:.0f}")])
    lines.append("## Store (L1 -> L2)\n")
    lines.append(md_table(header, rows) + "\n")

    # Load — throughput + latency
    header = ["adapter", "load MB/s", "load p50 ms", "load p99 ms", "load ops/s"]
    rows = []
    for name, _cfg, ops in runs:
        o = ops.get("Load")
        rows.append([name, cell(o, "throughput_avg_mbps"), cell(o, "duration_p50_ms", "{:.2f}"),
                     cell(o, "duration_p99_ms", "{:.2f}"), cell(o, "ops_per_sec_avg", "{:.0f}")])
    lines.append("## Load (L2 -> L1)\n")
    lines.append(md_table(header, rows) + "\n")

    # Lookup — latency + ops/s + hit rate
    header = ["adapter", "lookup p50 ms", "lookup p99 ms", "lookup ops/s", "hits/total"]
    rows = []
    for name, _cfg, ops in runs:
        o = ops.get("Lookup")
        hits = "-"
        if o is not None:
            hits = f"{o.get('total_success', '?')}/{o.get('total_keys', '?')}"
        rows.append([name, cell(o, "duration_p50_ms", "{:.2f}"), cell(o, "duration_p99_ms", "{:.2f}"),
                     cell(o, "ops_per_sec_avg", "{:.0f}"), hits])
    lines.append("## Lookup (query + lock)\n")
    lines.append(md_table(header, rows) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
