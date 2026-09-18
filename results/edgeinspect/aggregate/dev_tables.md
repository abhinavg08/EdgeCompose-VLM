# EdgeInspect results (dev)

## Macro over categories (mean over reference seeds)

| k | retention | n_seeds | f1 | f1_std | precision | recall | f1_max | auroc | f1_structural | f1_logical | auroc_structural | auroc_logical | pred_anomalous_ratio | ttft_ms_p50 | total_latency_ms_p50 | vram_mb | visual_tokens_after | feasible_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 1 | 0.781 | — | 0.658 | 0.963 | 0.800 | 0.624 | 0.644 | 0.655 | 0.745 | 0.502 | 0.975 | 689.7 | 946.2 | 4792.5 | 502.000 | True |
| 1 | 1.000 | 1 | 0.768 | — | 0.652 | 0.938 | 0.825 | 0.670 | 0.644 | 0.632 | 0.787 | 0.552 | 0.958 | 905.1 | 1157.8 | 4918.5 | 1005.000 | True |
| 4 | 0.500 | 1 | 0.781 | — | 0.658 | 0.963 | 0.800 | 0.515 | 0.644 | 0.655 | 0.660 | 0.370 | 0.975 | 1791.1 | 2041.9 | 5468.5 | 1255.000 | True |
| 4 | 1.000 | 1 | 0.775 | — | 0.655 | 0.950 | 0.817 | 0.688 | 0.644 | 0.644 | 0.807 | 0.568 | 0.967 | 2812.0 | 3052.4 | 6130.5 | 2512.500 | True |

## Pareto-efficient (feasible) configurations on f1

| k | retention | f1 | total_latency_ms_p50 | vram_mb | visual_tokens_after | front_q_lat | front_q_vram | front_q_tokens | front_q_lat_vram |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.781 | 946.2 | 4792.5 | 502.000 | True | True | True | True |

## AUFC (area under F1-max vs log2 k)

| retention | aufc_f1_max | ks |
|---|---|---|
| 0.500 | 0.800 | 1/4 |
| 1.000 | 0.821 | 1/4 |

## Per category (mean over seeds)

| category | k | retention | f1 | f1_max | auroc |
|---|---|---|---|---|---|
| pushpins | 1 | 0.500 | 0.800 | 0.800 | 0.624 |
| pushpins | 1 | 1.000 | 0.800 | 0.825 | 0.670 |
| pushpins | 4 | 0.500 | 0.800 | 0.800 | 0.515 |
| pushpins | 4 | 1.000 | 0.800 | 0.817 | 0.688 |
| splicing_connectors | 1 | 0.500 | 0.763 | — | — |
| splicing_connectors | 1 | 1.000 | 0.737 | — | — |
| splicing_connectors | 4 | 0.500 | 0.763 | — | — |
| splicing_connectors | 4 | 1.000 | 0.750 | — | — |



## Max reference count within 8 GB VRAM (isolated probe, pushpins)

| attention | retention | max_k_within_vram | footprint_at_max_k_mb | latency_at_max_k_ms | first_failing_k | latency_first_failing_ms |
|---|---|---|---|---|---|---|
| sdpa | 0.2 | 16 | 7384.5 | 5456.6 | 24 | 16416.9 |
| sdpa | 0.5 | 12 | 7632.5 | 5591.9 | 16 | 16303.4 |
| sdpa | 0.8 | 8 | 7392.5 | 4889.7 | 12 | 21380.6 |
| sdpa | 1.0 | 4 | 6130.5 | 2951.8 | 8 | 6369.7 |

## Latency: interleaved grid (all categories) vs isolated probe (pushpins)

| k | retention | total_latency_ms_p50 | latency_ms_median | device_footprint_mb | status | interleaved_over_isolated |
|---|---|---|---|---|---|---|
| 1 | 0.50 | 946.2 | 929.2 | 4792.5 | ok | 1.02 |
| 1 | 1.00 | 1157.8 | 1103.5 | 4918.5 | ok | 1.05 |
| 4 | 0.50 | 2041.9 | 1960.0 | 5468.5 | ok | 1.04 |
| 4 | 1.00 | 3052.4 | 2951.8 | 6130.5 | ok | 1.03 |


## Isolated memory scaling (all points)

| category | k | retention | status | error_message | peak_allocated_mb | peak_reserved_mb | context_overhead_mb | device_footprint_mb | latency_ms_median | ttft_ms_median | visual_tokens_before | visual_tokens_after | prefill_seq_len | max_pixels_tokens | attention_backend | device_total_mb | exceeds_vram |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pushpins | 1 | 1.0 | ok | — | 3553.1 | 3818.0 | 1100.5 | 4918.5 | 1103.5 | 855.2 | 986 | 986 | 1136 | 512 | sdpa | 8187.5 | False |
| pushpins | 2 | 1.0 | ok | — | 3800.8 | 4192.0 | 1100.5 | 5292.5 | 1684.6 | 1445.9 | 1479 | 1479 | 1638 | 512 | sdpa | 8187.5 | False |
| pushpins | 4 | 1.0 | ok | — | 4535.1 | 5030.0 | 1100.5 | 6130.5 | 2951.8 | 2720.4 | 2465 | 2465 | 2642 | 512 | sdpa | 8187.5 | False |
| pushpins | 8 | 1.0 | over_vram | — | 6767.9 | 7534.0 | 1100.5 | 8634.5 | 6369.7 | 6096.4 | 4437 | 4437 | 4650 | 512 | sdpa | 8187.5 | True |
| pushpins | 12 | 1.0 | over_vram | — | 10177.9 | 11174.0 | 1100.5 | 12274.5 | 104392.9 | 104044.5 | 6409 | 6409 | 6662 | 512 | sdpa | 8187.5 | True |
| pushpins | 1 | 0.8 | ok | — | 3466.3 | 3816.0 | 1100.5 | 4916.5 | 1043.1 | 769.0 | 986 | 740 | 890 | 512 | sdpa | 8187.5 | False |
| pushpins | 2 | 0.8 | ok | — | 3624.3 | 4044.0 | 1100.5 | 5144.5 | 1432.6 | 1189.9 | 1479 | 1110 | 1269 | 512 | sdpa | 8187.5 | False |
| pushpins | 4 | 0.8 | ok | — | 4058.7 | 4624.0 | 1100.5 | 5724.5 | 2415.1 | 2179.0 | 2465 | 1850 | 2027 | 512 | sdpa | 8187.5 | False |
| pushpins | 8 | 0.8 | ok | — | 5427.4 | 6292.0 | 1100.5 | 7392.5 | 4889.7 | 4644.6 | 4437 | 3330 | 3543 | 512 | sdpa | 8187.5 | False |
| pushpins | 12 | 0.8 | over_vram | — | 7428.8 | 8600.0 | 1100.5 | 9700.5 | 21380.6 | 21092.0 | 6409 | 4810 | 5063 | 512 | sdpa | 8187.5 | True |
| pushpins | 16 | 0.8 | over_vram | — | 10092.9 | 11542.0 | 1100.5 | 12642.5 | 111015.3 | 110669.7 | 8381 | 6290 | 6583 | 512 | sdpa | 8187.5 | True |
| pushpins | 1 | 0.5 | ok | — | 3442.1 | 3692.0 | 1100.5 | 4792.5 | 929.2 | 683.4 | 986 | 492 | 642 | 512 | sdpa | 8187.5 | False |
| pushpins | 2 | 0.5 | ok | — | 3527.6 | 3938.0 | 1100.5 | 5038.5 | 1269.9 | 1029.6 | 1479 | 738 | 897 | 512 | sdpa | 8187.5 | False |
| pushpins | 4 | 0.5 | ok | — | 3704.5 | 4368.0 | 1100.5 | 5468.5 | 1960.0 | 1726.0 | 2465 | 1230 | 1407 | 512 | sdpa | 8187.5 | False |
| pushpins | 8 | 0.5 | ok | — | 4373.8 | 5284.0 | 1100.5 | 6384.5 | 3661.8 | 3432.7 | 4437 | 2214 | 2427 | 512 | sdpa | 8187.5 | False |
| pushpins | 12 | 0.5 | ok | — | 5352.7 | 6532.0 | 1100.5 | 7632.5 | 5591.9 | 5361.3 | 6409 | 3198 | 3451 | 512 | sdpa | 8187.5 | False |
| pushpins | 16 | 0.5 | over_vram | — | 6601.7 | 8342.0 | 1100.5 | 9442.5 | 16303.4 | 16039.9 | 8381 | 4182 | 4475 | 512 | sdpa | 8187.5 | True |
| pushpins | 24 | 0.5 | over_vram | — | 10013.2 | 12460.0 | 1100.5 | 13560.5 | 138679.9 | 138317.4 | 12325 | 6150 | 6523 | 512 | sdpa | 8187.5 | True |
| pushpins | 1 | 0.2 | ok | — | 3442.1 | 3682.0 | 1100.5 | 4782.5 | 826.8 | 594.4 | 986 | 246 | 396 | 512 | sdpa | 8187.5 | False |
| pushpins | 2 | 0.2 | ok | — | 3527.6 | 3812.0 | 1100.5 | 4912.5 | 1102.1 | 870.3 | 1479 | 369 | 528 | 512 | sdpa | 8187.5 | False |
| pushpins | 4 | 0.2 | ok | — | 3704.5 | 4078.0 | 1100.5 | 5178.5 | 1677.1 | 1436.5 | 2465 | 615 | 792 | 512 | sdpa | 8187.5 | False |
| pushpins | 8 | 0.2 | ok | — | 4059.0 | 4586.0 | 1100.5 | 5686.5 | 2830.8 | 2586.4 | 4437 | 1107 | 1320 | 512 | sdpa | 8187.5 | False |
| pushpins | 12 | 0.2 | ok | — | 4416.6 | 5516.0 | 1100.5 | 6616.5 | 4136.2 | 3900.1 | 6409 | 1599 | 1852 | 512 | sdpa | 8187.5 | False |
| pushpins | 16 | 0.2 | ok | — | 4767.8 | 6284.0 | 1100.5 | 7384.5 | 5456.6 | 5212.7 | 8381 | 2091 | 2384 | 512 | sdpa | 8187.5 | False |
| pushpins | 24 | 0.2 | over_vram | — | 5473.8 | 8066.0 | 1100.5 | 9166.5 | 16416.9 | 16181.6 | 12325 | 3075 | 3448 | 512 | sdpa | 8187.5 | True |
