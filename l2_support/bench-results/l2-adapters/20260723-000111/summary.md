# L2 adapter comparison (`lmcache bench l2`)

workload: num_keys=32 in_flight=4 data_size_kb=2304 rounds=5 (warmup 2) lookup_max_hit_rate=1.0

## Store (L1 -> L2)

| adapter   | store MB/s | store p50 ms | store p99 ms | store ops/s |
|-----------|------------|--------------|--------------|-------------|
| fs        | 5760.9     | 49.30        | 53.68        | 2560        |
| fs_native | 30943.5    | 9.25         | 9.73         | 13753       |
| resp      | 3720.5     | 76.66        | 81.76        | 1654        |

## Load (L2 -> L1)

| adapter   | load MB/s | load p50 ms | load p99 ms | load ops/s |
|-----------|-----------|-------------|-------------|------------|
| fs        | 9440.9    | 30.40       | 31.25       | 4196       |
| fs_native | 43957.5   | 6.57        | 6.59        | 19537      |
| resp      | 4064.6    | 66.41       | 94.27       | 1806       |

## Lookup (query + lock)

| adapter   | lookup p50 ms | lookup p99 ms | lookup ops/s | hits/total |
|-----------|---------------|---------------|--------------|------------|
| fs        | 6.68          | 6.88          | 19080        | 640/640    |
| fs_native | 0.29          | 0.29          | 450327       | 640/640    |
| resp      | 1.04          | 1.18          | 121490       | 640/640    |

