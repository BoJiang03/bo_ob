# L2 adapter comparison (`lmcache bench l2`)

workload: num_keys=32 in_flight=4 data_size_kb=256 rounds=5 (warmup 2) lookup_max_hit_rate=1.0

## Store (L1 -> L2)

| adapter   | store MB/s | store p50 ms | store p99 ms | store ops/s |
|-----------|------------|--------------|--------------|-------------|
| fs        | 637.2      | 50.49        | 51.72        | 2549        |
| fs_native | 10450.3    | 3.17         | 3.34         | 41801       |
| resp      | 2960.1     | 10.77        | 11.45        | 11840       |

## Load (L2 -> L1)

| adapter   | load MB/s | load p50 ms | load p99 ms | load ops/s |
|-----------|-----------|-------------|-------------|------------|
| fs        | 1657.8    | 19.23       | 20.33       | 6631       |
| fs_native | 25883.5   | 1.23        | 1.28        | 103534     |
| resp      | 7004.9    | 4.51        | 4.70        | 28020      |

## Lookup (query + lock)

| adapter   | lookup p50 ms | lookup p99 ms | lookup ops/s | hits/total |
|-----------|---------------|---------------|--------------|------------|
| fs        | 8.35          | 8.81          | 15287        | 640/640    |
| fs_native | 0.23          | 0.25          | 542235       | 640/640    |
| resp      | 1.32          | 1.61          | 117668       | 640/640    |

