# L2 adapter comparison (`lmcache bench l2`)

workload: num_keys=32 in_flight=4 data_size_kb=256 rounds=5 (warmup 2) lookup_max_hit_rate=1.0

## Store (L1 -> L2)

| adapter   | store MB/s | store p50 ms | store p99 ms | store ops/s |
|-----------|------------|--------------|--------------|-------------|
| fs        | 686.6      | 47.38        | 49.83        | 2747        |
| fs_native | 8370.1     | 3.91         | 4.24         | 33480       |
| resp      | 3279.6     | 9.52         | 11.06        | 13118       |

## Load (L2 -> L1)

| adapter   | load MB/s | load p50 ms | load p99 ms | load ops/s |
|-----------|-----------|-------------|-------------|------------|
| fs        | 1240.3    | 25.85       | 26.20       | 4961       |
| fs_native | 34320.4   | 0.93        | 0.94        | 137282     |
| resp      | 5533.6    | 5.93        | 6.43        | 22135      |

## Lookup (query + lock)

| adapter   | lookup p50 ms | lookup p99 ms | lookup ops/s | hits/total |
|-----------|---------------|---------------|--------------|------------|
| fs        | 6.85          | 7.44          | 18566        | 640/640    |
| fs_native | 0.33          | 0.35          | 381134       | 640/640    |
| resp      | 0.95          | 1.05          | 149228       | 640/640    |

