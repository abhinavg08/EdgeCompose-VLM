# EdgeCompose-VLM tables (n=200 per dataset)

## Table 1 - System configuration

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MB |
| System RAM | 15.7 GB |
| NVIDIA driver | 560.94 |
| CUDA (torch build) | 11.8 |
| PyTorch | 2.7.1+cu118 |
| Transformers | 4.57.1 |
| AutoAWQ / Triton | 0.2.9 / triton-windows 3.3.1.post21 |
| Python / OS | 3.9.24 / Windows-10-10.0.26200-SP0 |
| Model | Qwen/Qwen2.5-VL-3B-Instruct-AWQ |
| Quantization | AWQ INT4 (w4, g128) LLM; vision tower FP16 |
| Model weights resident | 3251 MB allocated after load |
| SDPA kernels available | {'flash': False, 'mem_efficient': True, 'math': True} |
| flash-attn usable | False |

## Table 2 - Main results (medians unless noted; quality with 95% bootstrap CI)

| dataset | config_name | quality | quality_ci_low | quality_ci_high | ttft_ms_p50 | ttft_ms_p95 | total_latency_ms_p50 | total_latency_ms_p95 | tokens_per_second_p50 | peak_allocated_mb_p50 | peak_reserved_mb_max | energy_j_median | visual_tokens_after_mean | n_ok | n_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| textvqa | C0_sdpa_r100 | 0.8 | 0.7 | 0.9 | 1100.0 | 1438.3 | 1297.4 | 2068.2 | 2.4 | 3508.8 | 7782.0 | 93.8 | 953.5 | 38 | 0 |
| textvqa | C1_sdpa_r75 | 0.8 | 0.7 | 0.9 | 1152.5 | 1414.0 | 1371.0 | 1901.3 | 2.2 | 3446.5 | 7782.0 | 103.3 | 715.1 | 38 | 0 |
| textvqa | C2_sdpa_r50 | 0.8 | 0.7 | 0.9 | 912.2 | 1105.1 | 1105.6 | 1637.8 | 2.7 | 3446.5 | 7782.0 | 79.8 | 477.0 | 38 | 0 |
| textvqa | C3_sdpa_r25 | 0.6 | 0.5 | 0.8 | 701.9 | 896.3 | 915.3 | 1401.8 | 2.9 | 3446.5 | 7782.0 | 62.3 | 238.4 | 38 | 0 |
| textvqa | U2_sdpa_uniform_r50 | 0.8 | 0.7 | 0.9 | 869.8 | 1070.6 | 1104.9 | 1692.5 | 2.7 | 3438.1 | 7782.0 | 77.8 | 477.4 | 39 | 0 |
| textvqa | U3_sdpa_uniform_r25 | 0.7 | 0.5 | 0.8 | 689.9 | 841.9 | 896.8 | 1439.2 | 3.6 | 3438.1 | 7782.0 | 61.3 | 238.7 | 39 | 0 |
| textvqa | E0_eager_r100 | 0.8 | 0.7 | 0.9 | 1381.3 | 1877.6 | 1576.9 | 2389.7 | 1.9 | 5227.5 | 7782.0 | 103.8 | 953.5 | 38 | 0 |
| textvqa | E1_eager_r75 | 0.8 | 0.7 | 0.9 | 1473.8 | 1909.3 | 1730.4 | 2356.7 | 1.9 | 5227.5 | 7782.0 | 109.4 | 715.1 | 38 | 0 |
| textvqa | E2_eager_r50 | 0.8 | 0.7 | 0.9 | 1293.7 | 1678.9 | 1536.1 | 2080.8 | 2.1 | 5227.5 | 7782.0 | 94.1 | 477.0 | 38 | 0 |
| textvqa | E3_eager_r25 | 0.6 | 0.5 | 0.8 | 1036.5 | 1351.7 | 1280.1 | 1895.0 | 2.3 | 5227.5 | 7782.0 | 76.6 | 238.4 | 38 | 0 |

Quality columns in Table 2 are fractions (shown with 1 decimal above; see CSV for full precision).

## Table 3 - Relative to C0 (SDPA, 100% tokens); paired bootstrap 95% CIs

| config_name | dataset | n_paired | quality_retention_pct | quality_delta | quality_delta_ci_low | quality_delta_ci_high | latency_reduction_pct | latency_reduction_ci_low | latency_reduction_ci_high | ttft_reduction_pct | ttft_reduction_ci_low | ttft_reduction_ci_high | prefill_reduction_pct | prefill_reduction_ci_low | prefill_reduction_ci_high | vision_reduction_pct | vision_reduction_ci_low | vision_reduction_ci_high | vram_reduction_pct | vram_reduction_ci_low | vram_reduction_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | textvqa | 38 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| C1_sdpa_r75 | textvqa | 38 | 101.97 | 0.02 | 0.00 | 0.05 | -5.68 | -16.88 | 5.76 | -4.77 | -23.26 | 6.53 | -3.85 | -51.90 | 20.99 | -1.46 | -4.75 | 4.81 | 1.78 | 1.63 | 1.86 |
| C2_sdpa_r50 | textvqa | 38 | 98.36 | -0.01 | -0.08 | 0.04 | 14.78 | 4.29 | 22.89 | 17.07 | 3.00 | 25.47 | 31.63 | 0.68 | 48.03 | 0.73 | -5.51 | 7.30 | 1.78 | 1.63 | 1.86 |
| C3_sdpa_r25 | textvqa | 38 | 77.05 | -0.18 | -0.31 | -0.07 | 29.45 | 21.92 | 38.33 | 36.19 | 21.92 | 42.37 | 62.57 | 45.45 | 71.46 | 1.51 | -6.17 | 7.73 | 1.78 | 1.63 | 1.86 |
| U2_sdpa_uniform_r50 | textvqa | 38 | 101.91 | 0.01 | -0.05 | 0.09 | 15.24 | 6.82 | 25.88 | 21.32 | 6.01 | 29.53 | 32.36 | 1.31 | 48.37 | 2.43 | -6.17 | 6.71 | 2.03 | 1.71 | 2.09 |
| U3_sdpa_uniform_r25 | textvqa | 38 | 82.10 | -0.15 | -0.28 | -0.05 | 31.85 | 23.55 | 40.86 | 37.65 | 26.72 | 45.83 | 63.32 | 47.14 | 71.93 | 0.67 | -5.72 | 4.95 | 2.03 | 1.71 | 2.09 |
| E0_eager_r100 | textvqa | 38 | 100.00 | 0.00 | 0.00 | 0.00 | -21.54 | -31.17 | -10.85 | -25.57 | -44.71 | -14.73 | 7.03 | -1.20 | 13.46 | -82.73 | -118.71 | -60.09 | -48.98 | -50.63 | -43.58 |
| E1_eager_r75 | textvqa | 38 | 100.00 | 0.00 | 0.00 | 0.00 | -33.38 | -50.36 | -13.31 | -33.98 | -66.19 | -13.85 | 3.60 | -40.84 | 27.13 | -95.37 | -120.30 | -65.49 | -48.98 | -50.63 | -43.58 |
| E2_eager_r50 | textvqa | 38 | 98.36 | -0.01 | -0.08 | 0.04 | -18.40 | -36.42 | 4.83 | -17.61 | -46.64 | 4.86 | 36.25 | 7.39 | 51.08 | -103.74 | -127.65 | -63.21 | -48.98 | -50.63 | -43.58 |
| E3_eager_r25 | textvqa | 38 | 77.05 | -0.18 | -0.31 | -0.07 | 1.33 | -15.72 | 21.81 | 5.77 | -25.18 | 21.50 | 64.70 | 49.03 | 73.03 | -81.70 | -118.47 | -60.45 | -48.98 | -50.63 | -43.58 |

## Table 4 - Pareto-efficient configurations

| dataset | config_name | quality | latency_p50_ms | ttft_p50_ms | peak_alloc_mb | energy_j | front_q_lat_mem | front_q_lat | front_q_mem | front_q_energy |
|---|---|---|---|---|---|---|---|---|---|---|
| textvqa | C1_sdpa_r75 | 0.818 | 1370.991 | 1152.513 | 3446.484 | 103.349 | True | True | True | True |
| textvqa | U2_sdpa_uniform_r50 | 0.818 | 1104.895 | 869.761 | 3438.135 | 77.838 | True | True | True | True |
| textvqa | U3_sdpa_uniform_r25 | 0.659 | 896.807 | 689.922 | 3438.135 | 61.288 | True | True | False | True |

## Stage-wise medians and shares

| dataset | config_name | preprocess_ms_p50 | vision_ms_p50 | compress_ms_p50 | prefill_ms_p50 | decode_ms_p50 | total_latency_ms_p50 | decode_ms_per_token_p50 | generated_tokens_p50 | prefill_seq_len_mean | frac_prefill_ge_1024 | preprocess_share | vision_share | compress_share | prefill_share | decode_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| textvqa | C0_sdpa_r100 | 23.08 | 409.10 | 2.12 | 637.00 | 239.19 | 1297.36 | 95.98 | 3.00 | 993.29 | 0.50 | 0.02 | 0.31 | 0.00 | 0.49 | 0.18 |
| textvqa | C1_sdpa_r75 | 25.51 | 415.07 | 39.42 | 661.53 | 236.94 | 1370.99 | 117.04 | 3.00 | 754.84 | 0.00 | 0.02 | 0.30 | 0.03 | 0.48 | 0.17 |
| textvqa | C2_sdpa_r50 | 29.85 | 406.12 | 39.54 | 435.53 | 218.31 | 1105.57 | 94.60 | 3.00 | 516.74 | 0.00 | 0.03 | 0.36 | 0.04 | 0.39 | 0.19 |
| textvqa | C3_sdpa_r25 | 24.22 | 402.94 | 39.92 | 238.43 | 187.59 | 915.31 | 80.75 | 3.00 | 278.21 | 0.00 | 0.03 | 0.45 | 0.04 | 0.27 | 0.21 |
| textvqa | U2_sdpa_uniform_r50 | 23.98 | 399.34 | 2.34 | 431.13 | 237.54 | 1104.89 | 71.52 | 3.00 | 517.26 | 0.00 | 0.02 | 0.36 | 0.00 | 0.39 | 0.22 |
| textvqa | U3_sdpa_uniform_r25 | 24.40 | 409.18 | 2.40 | 233.77 | 179.71 | 896.81 | 82.53 | 3.00 | 278.51 | 0.00 | 0.03 | 0.48 | 0.00 | 0.28 | 0.21 |
| textvqa | E0_eager_r100 | 21.57 | 747.54 | 2.13 | 592.25 | 231.91 | 1576.88 | 75.39 | 3.00 | 993.29 | 0.50 | 0.01 | 0.47 | 0.00 | 0.37 | 0.15 |
| textvqa | E1_eager_r75 | 28.45 | 799.23 | 40.49 | 614.06 | 237.92 | 1730.36 | 97.84 | 3.00 | 754.84 | 0.00 | 0.02 | 0.46 | 0.02 | 0.36 | 0.14 |
| textvqa | E2_eager_r50 | 28.70 | 833.50 | 40.25 | 406.07 | 215.95 | 1536.07 | 117.42 | 3.00 | 516.74 | 0.00 | 0.02 | 0.55 | 0.03 | 0.27 | 0.14 |
| textvqa | E3_eager_r25 | 21.64 | 743.32 | 39.97 | 224.85 | 175.86 | 1280.08 | 87.55 | 3.00 | 278.21 | 0.00 | 0.02 | 0.62 | 0.03 | 0.19 | 0.15 |

## Composition interaction (A = VisionZip r, B = SDPA vs eager, baseline = eager 100%)

| dataset | retention | n | metric | R_A | R_B | R_AB | R_A*R_B | I | I_ci_low | I_ci_high | verdict | R_additive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| textvqa | 0.750 | 38 | E2E latency | 1.097 | 0.823 | 0.869 | 0.903 | -0.033 | -0.133 | 0.063 | approximately independent | 0.920 |
| textvqa | 0.750 | 38 | TTFT | 1.067 | 0.796 | 0.834 | 0.850 | -0.015 | -0.107 | 0.118 | approximately independent | 0.863 |
| textvqa | 0.750 | 38 | prefill | 1.037 | 1.076 | 1.117 | 1.115 | 0.002 | -0.116 | 0.124 | approximately independent | 1.112 |
| textvqa | 0.750 | 38 | vision | 1.069 | 0.547 | 0.555 | 0.585 | -0.030 | -0.109 | 0.024 | approximately independent | 0.616 |
| textvqa | 0.750 | 38 | peak alloc | 1.000 | 0.671 | 0.659 | 0.671 | -0.012 | -0.012 | -0.011 | approximately independent | 0.671 |
| textvqa | 0.500 | 38 | E2E latency | 0.974 | 0.823 | 0.701 | 0.801 | -0.100 | -0.203 | 0.014 | approximately independent | 0.797 |
| textvqa | 0.500 | 38 | TTFT | 0.937 | 0.796 | 0.660 | 0.746 | -0.085 | -0.167 | 0.040 | approximately independent | 0.733 |
| textvqa | 0.500 | 38 | prefill | 0.686 | 1.076 | 0.735 | 0.737 | -0.002 | -0.080 | 0.074 | approximately independent | 0.761 |
| textvqa | 0.500 | 38 | vision | 1.115 | 0.547 | 0.543 | 0.610 | -0.067 | -0.189 | 0.024 | approximately independent | 0.662 |
| textvqa | 0.500 | 38 | peak alloc | 1.000 | 0.671 | 0.659 | 0.671 | -0.012 | -0.012 | -0.011 | approximately independent | 0.671 |
| textvqa | 0.250 | 38 | E2E latency | 0.812 | 0.823 | 0.580 | 0.668 | -0.087 | -0.193 | 0.017 | approximately independent | 0.635 |
| textvqa | 0.250 | 38 | TTFT | 0.750 | 0.796 | 0.508 | 0.598 | -0.089 | -0.182 | 0.021 | approximately independent | 0.547 |
| textvqa | 0.250 | 38 | prefill | 0.380 | 1.076 | 0.403 | 0.408 | -0.006 | -0.050 | 0.039 | approximately independent | 0.455 |
| textvqa | 0.250 | 38 | vision | 0.994 | 0.547 | 0.539 | 0.544 | -0.005 | -0.138 | 0.079 | approximately independent | 0.542 |
| textvqa | 0.250 | 38 | peak alloc | 1.000 | 0.671 | 0.659 | 0.671 | -0.012 | -0.012 | -0.011 | approximately independent | 0.671 |

## Theoretical FLOPs vs measured time (SDPA + VisionZip)

| dataset | config_name | retention | prefill_len | prefill_flops_ratio | prefill_time_ratio | ttft_flops_ratio | ttft_time_ratio | vit_share_of_flops_at_100 |
|---|---|---|---|---|---|---|---|---|
| textvqa | C1_sdpa_r75 | 0.750 | 754.842 | 0.755 | 1.039 | 0.872 | 1.048 | 0.476 |
| textvqa | C2_sdpa_r50 | 0.500 | 516.737 | 0.514 | 0.684 | 0.745 | 0.829 | 0.476 |
| textvqa | C3_sdpa_r25 | 0.250 | 278.211 | 0.275 | 0.374 | 0.620 | 0.638 | 0.476 |
| textvqa | C0_sdpa_r100 | 1.000 | 993.289 | 1.000 | 1.000 | 1.000 | 1.000 | 0.476 |
