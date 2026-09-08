# EdgeCompose-VLM tables (n=500 per dataset)

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

`peak_allocated_mb_p50` = per-query live-tensor peak (interleaved run); `device_footprint_mb` = isolated per-config peak reserved + CUDA context/other processes on the 10 largest images.

| dataset | config_name | quality | quality_ci_low | quality_ci_high | ttft_ms_p50 | ttft_ms_p95 | total_latency_ms_p50 | total_latency_ms_p95 | tokens_per_second_p50 | peak_allocated_mb_p50 | device_footprint_mb | energy_j_median | visual_tokens_after_mean | n_ok | n_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 0.848 | 0.818 | 0.878 | 459.4 | 558.6 | 529.7 | 671.5 | 3.776 | 3321.8 | 4600.5 | 39.4 | 353.444 | 500 | 0 |
| pope | C1_sdpa_r75 | 0.850 | 0.820 | 0.880 | 400.4 | 469.2 | 471.4 | 586.2 | 4.243 | 3320.9 | 4612.5 | 32.5 | 265.102 | 500 | 0 |
| pope | C2_sdpa_r50 | 0.850 | 0.820 | 0.878 | 321.5 | 400.1 | 397.8 | 508.9 | 5.028 | 3320.9 | 4612.5 | 27.7 | 176.744 | 500 | 0 |
| pope | C3_sdpa_r25 | 0.836 | 0.804 | 0.868 | 265.2 | 328.3 | 332.3 | 434.9 | 6.019 | 3320.9 | 4606.5 | 23.7 | 88.342 | 500 | 0 |
| pope | U2_sdpa_uniform_r50 | 0.852 | 0.822 | 0.882 | 315.0 | 391.2 | 392.0 | 502.5 | 5.102 | 3320.7 | 4594.5 | 29.2 | 176.744 | 500 | 0 |
| pope | U3_sdpa_uniform_r25 | 0.844 | 0.812 | 0.876 | 260.1 | 320.9 | 330.8 | 436.0 | 6.047 | 3320.7 | 4588.5 | 22.8 | 88.342 | 500 | 0 |
| pope | T0_sdpa_awq64_r100 | 0.848 | 0.818 | 0.878 | 295.9 | 363.9 | 370.9 | 477.1 | 5.393 | 3346.2 | 4644.5 | 25.9 | 353.444 | 500 | 0 |
| pope | T1_sdpa_awq64_r75 | 0.850 | 0.820 | 0.880 | 292.7 | 351.9 | 357.3 | 463.1 | 5.597 | 3337.1 | 4656.5 | 26.2 | 265.102 | 500 | 0 |
| pope | T2_sdpa_awq64_r50 | 0.850 | 0.820 | 0.878 | 260.2 | 342.0 | 324.7 | 451.2 | 6.159 | 3328.9 | 4656.5 | 23.8 | 176.744 | 500 | 0 |
| pope | T3_sdpa_awq64_r25 | 0.834 | 0.802 | 0.866 | 231.5 | 325.8 | 299.0 | 446.9 | 6.689 | 3320.9 | 4650.5 | 20.6 | 88.342 | 500 | 0 |
| pope | E0_eager_r100 | 0.848 | 0.818 | 0.878 | 533.2 | 709.8 | 606.3 | 814.4 | 3.299 | 3519.5 | 5062.5 | 43.7 | 353.444 | 500 | 0 |
| pope | E1_eager_r75 | 0.850 | 0.820 | 0.880 | 474.4 | 636.1 | 547.9 | 743.6 | 3.650 | 3519.5 | 5056.5 | 36.7 | 265.102 | 500 | 0 |
| pope | E2_eager_r50 | 0.850 | 0.820 | 0.878 | 403.0 | 560.5 | 474.4 | 668.1 | 4.216 | 3519.5 | 5056.5 | 32.2 | 176.744 | 500 | 0 |
| pope | E3_eager_r25 | 0.836 | 0.804 | 0.868 | 339.0 | 490.8 | 401.2 | 592.4 | 4.985 | 3519.5 | 5052.5 | 27.8 | 88.342 | 500 | 0 |
| textvqa | C0_sdpa_r100 | 0.775 | 0.739 | 0.808 | 957.2 | 1304.0 | 1245.1 | 1743.5 | 2.877 | 3506.7 | 4886.5 | 92.4 | 935.692 | 500 | 0 |
| textvqa | C1_sdpa_r75 | 0.772 | 0.736 | 0.807 | 1118.6 | 1211.0 | 1258.6 | 1728.2 | 2.665 | 3445.7 | 4852.5 | 96.2 | 701.702 | 500 | 0 |
| textvqa | C2_sdpa_r50 | 0.752 | 0.716 | 0.787 | 890.6 | 975.0 | 1032.1 | 1510.6 | 3.240 | 3445.7 | 4842.5 | 78.3 | 468.002 | 500 | 0 |
| textvqa | C3_sdpa_r25 | 0.646 | 0.604 | 0.685 | 686.5 | 780.6 | 831.5 | 1270.3 | 3.938 | 3445.7 | 4830.5 | 62.3 | 233.990 | 500 | 0 |
| textvqa | U2_sdpa_uniform_r50 | 0.733 | 0.696 | 0.769 | 852.4 | 934.6 | 992.6 | 1415.3 | 3.331 | 3437.4 | 4714.5 | 73.7 | 468.002 | 500 | 0 |
| textvqa | U3_sdpa_uniform_r25 | 0.578 | 0.537 | 0.618 | 654.5 | 736.8 | 793.0 | 1226.3 | 4.091 | 3437.4 | 4702.5 | 58.1 | 233.990 | 500 | 0 |
| textvqa | T0_sdpa_awq64_r100 | 0.775 | 0.739 | 0.808 | 843.4 | 934.4 | 975.8 | 1514.3 | 3.443 | 3506.7 | 4886.5 | 73.5 | 935.692 | 500 | 0 |
| textvqa | T1_sdpa_awq64_r75 | 0.772 | 0.735 | 0.806 | 765.1 | 861.1 | 899.0 | 1392.1 | 3.804 | 3445.7 | 4850.5 | 67.6 | 701.702 | 500 | 0 |
| textvqa | T2_sdpa_awq64_r50 | 0.750 | 0.715 | 0.786 | 654.8 | 744.7 | 789.0 | 1278.2 | 4.306 | 3445.7 | 4840.5 | 58.6 | 468.002 | 500 | 0 |
| textvqa | T3_sdpa_awq64_r25 | 0.646 | 0.605 | 0.685 | 601.7 | 691.7 | 729.5 | 1186.1 | 4.547 | 3445.7 | 4832.5 | 54.2 | 233.990 | 500 | 0 |
| textvqa | E0_eager_r100 | 0.774 | 0.739 | 0.807 | 1313.2 | 1715.0 | 1468.0 | 2185.7 | 2.252 | 5215.7 | 6794.5 | 102.4 | 935.692 | 500 | 0 |
| textvqa | E1_eager_r75 | 0.772 | 0.736 | 0.805 | 1344.7 | 1979.9 | 1481.1 | 2269.6 | 2.272 | 5215.7 | 6770.5 | 103.9 | 701.702 | 500 | 0 |
| textvqa | E2_eager_r50 | 0.750 | 0.715 | 0.786 | 1127.7 | 1764.3 | 1272.7 | 2078.3 | 2.615 | 5215.7 | 6762.5 | 86.7 | 468.002 | 500 | 0 |
| textvqa | E3_eager_r25 | 0.647 | 0.606 | 0.686 | 943.1 | 1573.3 | 1090.4 | 1861.3 | 2.914 | 5215.7 | 6750.5 | 72.9 | 233.990 | 500 | 0 |

## Table 3 - Relative to C0 (SDPA, 100% tokens); paired bootstrap 95% CIs

| config_name | dataset | n_paired | quality_retention_pct | quality_delta | quality_delta_ci_low | quality_delta_ci_high | latency_reduction_pct | latency_reduction_ci_low | latency_reduction_ci_high | ttft_reduction_pct | ttft_reduction_ci_low | ttft_reduction_ci_high | prefill_reduction_pct | prefill_reduction_ci_low | prefill_reduction_ci_high | vision_reduction_pct | vision_reduction_ci_low | vision_reduction_ci_high | vram_reduction_pct | vram_reduction_ci_low | vram_reduction_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | pope | 500 | 100.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| C1_sdpa_r75 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | 11.0 | 9.5 | 12.1 | 12.8 | 11.9 | 13.9 | 20.5 | 19.6 | 21.8 | -0.2 | -2.1 | 4.9 | 0.0 | 0.0 | 0.0 |
| C2_sdpa_r50 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | 24.9 | 23.4 | 26.4 | 30.0 | 28.8 | 30.8 | 44.2 | 43.4 | 44.9 | 0.7 | -1.8 | 2.4 | 0.0 | 0.0 | 0.0 |
| C3_sdpa_r25 | pope | 500 | 98.6 | -0.01 | -0.03 | 0.01 | 37.3 | 35.8 | 40.1 | 42.3 | 40.1 | 46.2 | 63.4 | 62.8 | 66.7 | 0.4 | -1.5 | 3.4 | 0.0 | 0.0 | 0.0 |
| U2_sdpa_uniform_r50 | pope | 500 | 100.5 | 0.00 | -0.02 | 0.02 | 26.0 | 24.5 | 26.9 | 31.4 | 30.4 | 32.0 | 43.9 | 43.1 | 44.9 | -0.2 | -1.7 | 2.4 | 0.0 | 0.0 | 0.0 |
| U3_sdpa_uniform_r25 | pope | 500 | 99.5 | -0.00 | -0.03 | 0.02 | 37.6 | 36.7 | 40.9 | 43.4 | 41.2 | 47.3 | 63.8 | 62.9 | 67.0 | -0.2 | -2.2 | 2.3 | 0.0 | 0.0 | 0.0 |
| T0_sdpa_awq64_r100 | pope | 500 | 100.0 | 0.00 | 0.00 | 0.00 | 30.0 | 29.0 | 31.7 | 35.6 | 34.0 | 36.1 | 49.8 | 49.3 | 51.7 | -0.1 | -2.6 | 2.5 | -0.7 | -0.7 | -0.7 |
| T1_sdpa_awq64_r75 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | 32.5 | 31.6 | 34.8 | 36.3 | 35.5 | 37.1 | 56.5 | 55.7 | 57.6 | -0.0 | -2.4 | 2.3 | -0.5 | -0.5 | -0.5 |
| T2_sdpa_awq64_r50 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | 38.7 | 37.6 | 40.9 | 43.4 | 42.3 | 44.4 | 68.0 | 61.2 | 69.0 | 0.6 | -3.0 | 2.6 | -0.2 | -0.2 | -0.2 |
| T3_sdpa_awq64_r25 | pope | 500 | 98.3 | -0.01 | -0.04 | 0.01 | 43.6 | 42.9 | 45.6 | 49.6 | 47.4 | 52.2 | 74.1 | 73.6 | 74.9 | -1.1 | -3.3 | 1.8 | 0.0 | 0.0 | 0.0 |
| E0_eager_r100 | pope | 500 | 100.0 | 0.00 | 0.00 | 0.00 | -14.5 | -16.8 | -13.0 | -16.1 | -21.4 | -15.0 | 2.9 | 2.0 | 4.1 | -69.8 | -73.9 | -63.5 | -6.0 | -6.0 | -6.0 |
| E1_eager_r75 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | -3.4 | -4.7 | -1.7 | -3.3 | -6.3 | -2.1 | 22.5 | 21.5 | 23.7 | -69.8 | -73.5 | -62.4 | -6.0 | -6.0 | -6.0 |
| E2_eager_r50 | pope | 500 | 100.2 | 0.00 | -0.01 | 0.01 | 10.4 | 7.0 | 12.0 | 12.3 | 7.6 | 13.4 | 45.6 | 44.8 | 46.6 | -72.2 | -74.8 | -63.0 | -6.0 | -6.0 | -6.0 |
| E3_eager_r25 | pope | 500 | 98.6 | -0.01 | -0.03 | 0.01 | 24.3 | 20.7 | 25.3 | 26.2 | 21.4 | 28.9 | 64.7 | 64.3 | 68.4 | -70.5 | -74.2 | -62.2 | -6.0 | -6.0 | -6.0 |
| C0_sdpa_r100 | textvqa | 500 | 100.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| C1_sdpa_r75 | textvqa | 500 | 99.7 | -0.00 | -0.02 | 0.01 | -1.1 | -5.4 | 1.4 | -16.9 | -21.3 | -5.0 | -26.7 | -50.1 | 10.8 | 0.2 | -0.3 | 0.4 | 1.7 | 1.7 | 1.8 |
| C2_sdpa_r50 | textvqa | 500 | 97.0 | -0.02 | -0.04 | -0.01 | 17.1 | 13.5 | 19.0 | 6.9 | 3.8 | 15.2 | 16.7 | 1.7 | 40.2 | 0.2 | -0.3 | 0.5 | 1.7 | 1.7 | 1.8 |
| C3_sdpa_r25 | textvqa | 500 | 83.3 | -0.13 | -0.16 | -0.10 | 33.2 | 30.4 | 34.6 | 28.3 | 25.8 | 34.6 | 54.7 | 46.8 | 67.2 | 0.3 | -0.3 | 0.4 | 1.7 | 1.7 | 1.8 |
| U2_sdpa_uniform_r50 | textvqa | 500 | 94.7 | -0.04 | -0.06 | -0.02 | 20.3 | 16.8 | 22.1 | 10.9 | 7.7 | 19.1 | 16.9 | 1.7 | 40.3 | 0.1 | -0.4 | 0.4 | 2.0 | 1.9 | 2.1 |
| U3_sdpa_uniform_r25 | textvqa | 500 | 74.6 | -0.20 | -0.24 | -0.16 | 36.3 | 33.5 | 37.7 | 31.6 | 29.3 | 37.8 | 54.7 | 46.8 | 67.2 | 0.2 | -0.3 | 0.5 | 2.0 | 1.9 | 2.1 |
| T0_sdpa_awq64_r100 | textvqa | 500 | 100.0 | 0.00 | 0.00 | 0.00 | 21.6 | 18.2 | 23.5 | 11.9 | 8.8 | 21.6 | 22.3 | 3.5 | 44.6 | 0.3 | -0.2 | 0.6 | 0.0 | 0.0 | 0.0 |
| T1_sdpa_awq64_r75 | textvqa | 500 | 99.6 | -0.00 | -0.02 | 0.01 | 27.8 | 24.8 | 29.8 | 20.1 | 17.2 | 28.5 | 39.3 | 28.8 | 61.3 | 0.2 | -0.2 | 0.5 | 1.7 | 1.7 | 1.8 |
| T2_sdpa_awq64_r50 | textvqa | 500 | 96.9 | -0.02 | -0.04 | -0.01 | 36.6 | 33.8 | 38.1 | 31.6 | 29.4 | 37.7 | 61.0 | 54.3 | 72.5 | 0.0 | -0.4 | 0.4 | 1.7 | 1.7 | 1.8 |
| T3_sdpa_awq64_r25 | textvqa | 500 | 83.4 | -0.13 | -0.16 | -0.10 | 41.4 | 38.7 | 42.6 | 37.1 | 35.1 | 42.7 | 71.4 | 66.5 | 79.4 | 0.1 | -0.4 | 0.5 | 1.7 | 1.7 | 1.8 |
| E0_eager_r100 | textvqa | 500 | 99.9 | -0.00 | -0.00 | 0.00 | -17.9 | -21.8 | -16.3 | -37.2 | -40.7 | -26.3 | 5.9 | 4.4 | 8.5 | -65.6 | -66.8 | -64.2 | -48.7 | -50.3 | -47.0 |
| E1_eager_r75 | textvqa | 500 | 99.6 | -0.00 | -0.02 | 0.01 | -19.0 | -24.0 | -15.8 | -40.5 | -45.0 | -28.2 | -17.7 | -39.1 | 17.0 | -65.9 | -67.0 | -64.5 | -48.7 | -50.3 | -47.0 |
| E2_eager_r50 | textvqa | 500 | 96.8 | -0.02 | -0.04 | -0.01 | -2.2 | -7.1 | 0.2 | -17.8 | -21.7 | -7.4 | 21.9 | 8.1 | 43.8 | -64.0 | -65.7 | -63.3 | -48.7 | -50.3 | -47.0 |
| E3_eager_r25 | textvqa | 500 | 83.5 | -0.13 | -0.16 | -0.10 | 12.4 | 8.0 | 14.9 | 1.5 | -1.8 | 10.6 | 57.3 | 49.9 | 69.0 | -64.8 | -66.8 | -63.7 | -48.7 | -50.3 | -47.0 |

## Table 4 - Pareto-efficient configurations

Noise-aware dominance: differences within 0.5 pp quality, 3% latency/energy or 1% memory (relative to C0) count as ties; `strict_front_q_lat_mem` is the tolerance-free front.

| dataset | config_name | quality | latency_p50_ms | ttft_p50_ms | peak_alloc_mb | energy_j | front_q_lat_mem | front_q_lat | front_q_mem | front_q_energy | strict_front_q_lat_mem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 0.848 | 529.7 | 459.4 | 3321.8 | 39.4 | False | False | True | False | False |
| pope | C1_sdpa_r75 | 0.850 | 471.4 | 400.4 | 3320.9 | 32.5 | False | False | True | False | False |
| pope | C2_sdpa_r50 | 0.850 | 397.8 | 321.5 | 3320.9 | 27.7 | False | False | True | False | False |
| pope | U2_sdpa_uniform_r50 | 0.852 | 392.0 | 315.0 | 3320.7 | 29.2 | False | False | True | False | True |
| pope | T0_sdpa_awq64_r100 | 0.848 | 370.9 | 295.9 | 3346.2 | 25.9 | False | False | True | False | False |
| pope | T1_sdpa_awq64_r75 | 0.850 | 357.3 | 292.7 | 3337.1 | 26.2 | False | False | True | False | False |
| pope | T2_sdpa_awq64_r50 | 0.850 | 324.7 | 260.2 | 3328.9 | 23.8 | True | True | True | True | True |
| pope | T3_sdpa_awq64_r25 | 0.834 | 299.0 | 231.5 | 3320.9 | 20.6 | True | True | False | True | True |
| textvqa | C1_sdpa_r75 | 0.772 | 1258.6 | 1118.6 | 3445.7 | 96.2 | False | False | True | False | True |
| textvqa | T1_sdpa_awq64_r75 | 0.772 | 899.0 | 765.1 | 3445.7 | 67.6 | True | True | True | True | True |
| textvqa | T2_sdpa_awq64_r50 | 0.750 | 789.0 | 654.8 | 3445.7 | 58.6 | True | True | False | True | True |
| textvqa | T3_sdpa_awq64_r25 | 0.646 | 729.5 | 601.7 | 3445.7 | 54.2 | True | True | False | True | True |

## VisionZip vs uniform subsampling at equal retention (paired)

| dataset | retention | n | quality_visionzip | quality_uniform | delta_vz_minus_uniform | delta_ci_low | delta_ci_high | compress_ms_visionzip_p50 | compress_ms_uniform_p50 | answers_identical_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| pope | 0.500 | 500 | 0.850 | 0.852 | -0.002 | -0.020 | 0.016 | 6.0 | 1.3 | 0.890 |
| pope | 0.250 | 500 | 0.836 | 0.844 | -0.008 | -0.034 | 0.020 | 6.0 | 1.3 | 0.754 |
| textvqa | 0.500 | 500 | 0.752 | 0.733 | 0.018 | -0.007 | 0.043 | 38.2 | 1.5 | 0.736 |
| textvqa | 0.250 | 500 | 0.646 | 0.578 | 0.068 | 0.025 | 0.111 | 38.0 | 1.5 | 0.468 |

## Stage-wise medians and shares

| dataset | config_name | preprocess_ms_p50 | vision_ms_p50 | compress_ms_p50 | prefill_ms_p50 | decode_ms_p50 | total_latency_ms_p50 | decode_ms_per_token_p50 | generated_tokens_p50 | prefill_seq_len_mean | frac_prefill_ge_1024 | preprocess_share | vision_share | compress_share | prefill_share | decode_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 8.2 | 129.1 | 1.2 | 320.7 | 64.4 | 529.7 | 64.4 | 2.00 | 392.7 | 0.00 | 0.02 | 0.25 | 0.00 | 0.61 | 0.12 |
| pope | C1_sdpa_r75 | 8.2 | 129.4 | 6.0 | 255.0 | 65.3 | 471.4 | 65.3 | 2.00 | 304.4 | 0.00 | 0.02 | 0.28 | 0.01 | 0.55 | 0.14 |
| pope | C2_sdpa_r50 | 8.2 | 128.2 | 6.0 | 179.0 | 64.8 | 397.8 | 64.8 | 2.00 | 216.0 | 0.00 | 0.02 | 0.33 | 0.02 | 0.46 | 0.17 |
| pope | C3_sdpa_r25 | 8.3 | 128.5 | 6.0 | 117.5 | 65.8 | 332.3 | 65.8 | 2.00 | 127.6 | 0.00 | 0.03 | 0.39 | 0.02 | 0.36 | 0.20 |
| pope | U2_sdpa_uniform_r50 | 8.1 | 129.3 | 1.3 | 179.9 | 66.3 | 392.0 | 66.3 | 2.00 | 216.0 | 0.00 | 0.02 | 0.34 | 0.00 | 0.47 | 0.17 |
| pope | U3_sdpa_uniform_r25 | 8.3 | 129.3 | 1.3 | 116.0 | 65.0 | 330.8 | 65.0 | 2.00 | 127.6 | 0.00 | 0.03 | 0.40 | 0.00 | 0.36 | 0.20 |
| pope | T0_sdpa_awq64_r100 | 8.3 | 129.2 | 1.2 | 161.1 | 64.6 | 370.9 | 64.6 | 2.00 | 392.7 | 0.00 | 0.02 | 0.35 | 0.00 | 0.44 | 0.18 |
| pope | T1_sdpa_awq64_r75 | 8.1 | 129.1 | 6.0 | 139.4 | 64.7 | 357.3 | 64.7 | 2.00 | 304.4 | 0.00 | 0.02 | 0.37 | 0.02 | 0.40 | 0.19 |
| pope | T2_sdpa_awq64_r50 | 8.1 | 128.2 | 5.9 | 102.5 | 64.8 | 324.7 | 64.8 | 2.00 | 216.0 | 0.00 | 0.03 | 0.41 | 0.02 | 0.33 | 0.21 |
| pope | T3_sdpa_awq64_r25 | 8.1 | 130.5 | 5.9 | 83.1 | 64.7 | 299.0 | 64.7 | 2.00 | 127.6 | 0.00 | 0.03 | 0.45 | 0.02 | 0.28 | 0.22 |
| pope | E0_eager_r100 | 8.3 | 219.2 | 1.2 | 311.5 | 64.0 | 606.3 | 64.0 | 2.00 | 392.7 | 0.00 | 0.01 | 0.36 | 0.00 | 0.52 | 0.11 |
| pope | E1_eager_r75 | 8.3 | 219.2 | 6.0 | 248.6 | 64.5 | 547.9 | 64.5 | 2.00 | 304.4 | 0.00 | 0.02 | 0.40 | 0.01 | 0.45 | 0.12 |
| pope | E2_eager_r50 | 8.3 | 222.2 | 6.0 | 174.6 | 64.6 | 474.4 | 64.6 | 2.00 | 216.0 | 0.00 | 0.02 | 0.47 | 0.01 | 0.37 | 0.14 |
| pope | E3_eager_r25 | 8.3 | 220.0 | 6.0 | 113.1 | 65.3 | 401.2 | 65.3 | 2.00 | 127.6 | 0.00 | 0.02 | 0.53 | 0.01 | 0.27 | 0.16 |
| textvqa | C0_sdpa_r100 | 19.6 | 394.9 | 1.4 | 518.3 | 146.0 | 1245.1 | 64.1 | 3.00 | 975.1 | 0.49 | 0.02 | 0.37 | 0.00 | 0.48 | 0.14 |
| textvqa | C1_sdpa_r75 | 19.8 | 394.2 | 37.8 | 656.7 | 142.5 | 1258.6 | 64.3 | 3.00 | 741.2 | 0.00 | 0.02 | 0.32 | 0.03 | 0.52 | 0.11 |
| textvqa | C2_sdpa_r50 | 20.0 | 394.0 | 38.2 | 431.8 | 144.3 | 1032.1 | 64.8 | 3.00 | 507.5 | 0.00 | 0.02 | 0.38 | 0.04 | 0.42 | 0.14 |
| textvqa | C3_sdpa_r25 | 20.0 | 393.7 | 38.0 | 234.7 | 141.5 | 831.5 | 65.0 | 3.00 | 273.4 | 0.00 | 0.02 | 0.48 | 0.05 | 0.28 | 0.17 |
| textvqa | U2_sdpa_uniform_r50 | 19.9 | 394.4 | 1.5 | 430.9 | 147.5 | 992.6 | 64.9 | 3.00 | 507.5 | 0.00 | 0.02 | 0.40 | 0.00 | 0.43 | 0.15 |
| textvqa | U3_sdpa_uniform_r25 | 19.6 | 394.2 | 1.5 | 234.7 | 140.3 | 793.0 | 65.3 | 3.00 | 273.4 | 0.00 | 0.02 | 0.50 | 0.00 | 0.30 | 0.18 |
| textvqa | T0_sdpa_awq64_r100 | 19.8 | 393.6 | 1.4 | 402.7 | 141.8 | 975.8 | 65.2 | 3.00 | 975.1 | 0.49 | 0.02 | 0.41 | 0.00 | 0.42 | 0.15 |
| textvqa | T1_sdpa_awq64_r75 | 19.8 | 394.0 | 37.5 | 314.8 | 143.5 | 899.0 | 64.9 | 3.00 | 741.2 | 0.00 | 0.02 | 0.43 | 0.04 | 0.35 | 0.16 |
| textvqa | T2_sdpa_awq64_r50 | 19.5 | 394.9 | 37.7 | 202.0 | 147.4 | 789.0 | 64.7 | 3.00 | 507.5 | 0.00 | 0.02 | 0.49 | 0.05 | 0.25 | 0.18 |
| textvqa | T3_sdpa_awq64_r25 | 19.9 | 394.5 | 37.8 | 148.4 | 145.8 | 729.5 | 64.5 | 3.00 | 273.4 | 0.00 | 0.03 | 0.53 | 0.05 | 0.20 | 0.20 |
| textvqa | E0_eager_r100 | 19.7 | 654.0 | 1.5 | 487.5 | 140.5 | 1468.0 | 64.3 | 3.00 | 975.1 | 0.49 | 0.02 | 0.50 | 0.00 | 0.37 | 0.11 |
| textvqa | E1_eager_r75 | 19.9 | 655.0 | 37.9 | 610.2 | 138.9 | 1481.1 | 64.4 | 3.00 | 741.2 | 0.00 | 0.01 | 0.45 | 0.03 | 0.42 | 0.10 |
| textvqa | E2_eager_r50 | 19.9 | 647.7 | 37.8 | 404.9 | 141.5 | 1272.7 | 64.9 | 3.00 | 507.5 | 0.00 | 0.02 | 0.52 | 0.03 | 0.32 | 0.11 |
| textvqa | E3_eager_r25 | 19.7 | 650.7 | 37.7 | 221.4 | 142.0 | 1090.4 | 65.2 | 3.00 | 273.4 | 0.00 | 0.02 | 0.61 | 0.04 | 0.21 | 0.13 |

## POPE detail (yes = positive class)

| config_name | pope_accuracy | pope_precision | pope_recall | pope_f1 | pope_yes_ratio | pope_f1_adversarial | pope_f1_popular |
|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | 0.848 | 0.901 | 0.787 | 0.840 | 0.444 | 0.835 | 0.850 |
| C1_sdpa_r75 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| C2_sdpa_r50 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| C3_sdpa_r25 | 0.836 | 0.902 | 0.760 | 0.825 | 0.428 | 0.804 | 0.866 |
| U2_sdpa_uniform_r50 | 0.852 | 0.898 | 0.799 | 0.846 | 0.452 | 0.839 | 0.859 |
| U3_sdpa_uniform_r25 | 0.844 | 0.907 | 0.772 | 0.834 | 0.432 | 0.830 | 0.843 |
| T0_sdpa_awq64_r100 | 0.848 | 0.901 | 0.787 | 0.840 | 0.444 | 0.835 | 0.850 |
| T1_sdpa_awq64_r75 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| T2_sdpa_awq64_r50 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| T3_sdpa_awq64_r25 | 0.834 | 0.898 | 0.760 | 0.823 | 0.430 | 0.804 | 0.861 |
| E0_eager_r100 | 0.848 | 0.901 | 0.787 | 0.840 | 0.444 | 0.835 | 0.850 |
| E1_eager_r75 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| E2_eager_r50 | 0.850 | 0.894 | 0.799 | 0.844 | 0.454 | 0.833 | 0.866 |
| E3_eager_r25 | 0.836 | 0.902 | 0.760 | 0.825 | 0.428 | 0.804 | 0.866 |

## Composition interaction (A = VisionZip r, B = SDPA vs eager, baseline = eager 100%)

| dataset | factor_B | baseline | retention | n | metric | R_A | R_B | R_AB | R_A*R_B | I | I_ci_low | I_ci_high | verdict | R_additive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | E2E latency | 0.904 | 0.874 | 0.778 | 0.789 | -0.012 | -0.022 | 0.009 | approximately independent | 0.777 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | TTFT | 0.890 | 0.862 | 0.751 | 0.767 | -0.016 | -0.027 | -0.001 | approximately independent | 0.751 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | prefill | 0.798 | 1.029 | 0.819 | 0.822 | -0.003 | -0.017 | 0.004 | approximately independent | 0.828 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | vision | 1.000 | 0.589 | 0.590 | 0.589 | 0.001 | -0.025 | 0.013 | approximately independent | 0.589 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | E2E latency | 0.782 | 0.874 | 0.656 | 0.684 | -0.027 | -0.046 | -0.017 | approximately independent | 0.656 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | TTFT | 0.756 | 0.862 | 0.603 | 0.651 | -0.048 | -0.058 | -0.033 | synergy (super-multiplicative gain) | 0.618 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | prefill | 0.560 | 1.029 | 0.575 | 0.577 | -0.002 | -0.011 | 0.005 | approximately independent | 0.590 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | vision | 1.014 | 0.589 | 0.585 | 0.597 | -0.012 | -0.026 | 0.014 | approximately independent | 0.603 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | E2E latency | 0.662 | 0.874 | 0.548 | 0.578 | -0.030 | -0.055 | -0.020 | approximately independent | 0.535 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | TTFT | 0.636 | 0.862 | 0.497 | 0.548 | -0.050 | -0.079 | -0.034 | synergy (super-multiplicative gain) | 0.497 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | prefill | 0.363 | 1.029 | 0.377 | 0.374 | 0.003 | -0.008 | 0.013 | approximately independent | 0.393 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | vision | 1.004 | 0.589 | 0.586 | 0.591 | -0.005 | -0.023 | 0.019 | approximately independent | 0.593 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | E2E latency | 0.890 | 0.700 | 0.675 | 0.623 | 0.051 | 0.034 | 0.072 | interference (sub-multiplicative gain) | 0.590 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | TTFT | 0.872 | 0.644 | 0.637 | 0.562 | 0.076 | 0.058 | 0.083 | interference (sub-multiplicative gain) | 0.516 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | prefill | 0.795 | 0.502 | 0.435 | 0.400 | 0.035 | 0.032 | 0.056 | interference (sub-multiplicative gain) | 0.298 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | vision | 1.002 | 1.001 | 1.000 | 1.003 | -0.003 | -0.030 | 0.050 | approximately independent | 1.003 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | peak alloc | 1.000 | 1.007 | 1.005 | 1.007 | -0.002 | -0.002 | -0.002 | approximately independent | 1.007 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | E2E latency | 0.751 | 0.700 | 0.613 | 0.526 | 0.087 | 0.070 | 0.112 | interference (sub-multiplicative gain) | 0.451 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | TTFT | 0.700 | 0.644 | 0.566 | 0.451 | 0.116 | 0.090 | 0.123 | interference (sub-multiplicative gain) | 0.344 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | prefill | 0.558 | 0.502 | 0.320 | 0.280 | 0.039 | 0.034 | 0.108 | interference (sub-multiplicative gain) | 0.061 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | vision | 0.993 | 1.001 | 0.994 | 0.994 | -0.000 | -0.036 | 0.048 | approximately independent | 0.994 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | peak alloc | 1.000 | 1.007 | 1.002 | 1.007 | -0.005 | -0.005 | -0.005 | approximately independent | 1.007 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | E2E latency | 0.627 | 0.700 | 0.564 | 0.439 | 0.125 | 0.111 | 0.136 | interference (sub-multiplicative gain) | 0.327 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | TTFT | 0.577 | 0.644 | 0.504 | 0.372 | 0.132 | 0.121 | 0.148 | interference (sub-multiplicative gain) | 0.222 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | prefill | 0.366 | 0.502 | 0.259 | 0.184 | 0.075 | 0.073 | 0.094 | interference (sub-multiplicative gain) | -0.131 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | vision | 0.996 | 1.001 | 1.011 | 0.996 | 0.015 | -0.023 | 0.056 | approximately independent | 0.996 |
| pope | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | peak alloc | 1.000 | 1.007 | 1.000 | 1.007 | -0.007 | -0.007 | -0.007 | approximately independent | 1.007 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | E2E latency | 1.009 | 0.848 | 0.857 | 0.856 | 0.002 | -0.013 | 0.029 | approximately independent | 0.857 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | TTFT | 1.024 | 0.729 | 0.852 | 0.746 | 0.105 | 0.029 | 0.129 | interference (sub-multiplicative gain) | 0.753 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | prefill | 1.252 | 1.063 | 1.347 | 1.331 | 0.016 | -0.020 | 0.043 | approximately independent | 1.315 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | vision | 1.002 | 0.604 | 0.603 | 0.605 | -0.002 | -0.006 | 0.003 | approximately independent | 0.605 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 500 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | E2E latency | 0.867 | 0.848 | 0.703 | 0.735 | -0.032 | -0.048 | -0.011 | approximately independent | 0.715 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | TTFT | 0.859 | 0.729 | 0.678 | 0.626 | 0.052 | -0.002 | 0.069 | approximately independent | 0.588 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | prefill | 0.831 | 1.063 | 0.886 | 0.883 | 0.003 | -0.021 | 0.020 | approximately independent | 0.894 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | vision | 0.990 | 0.604 | 0.602 | 0.598 | 0.004 | -0.001 | 0.008 | approximately independent | 0.594 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 500 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | E2E latency | 0.743 | 0.848 | 0.566 | 0.630 | -0.064 | -0.078 | -0.046 | synergy (super-multiplicative gain) | 0.591 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | TTFT | 0.718 | 0.729 | 0.523 | 0.523 | -0.001 | -0.045 | 0.014 | approximately independent | 0.447 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | prefill | 0.454 | 1.063 | 0.482 | 0.483 | -0.001 | -0.015 | 0.007 | approximately independent | 0.517 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | vision | 0.995 | 0.604 | 0.602 | 0.601 | 0.001 | -0.005 | 0.007 | approximately independent | 0.599 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 500 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | E2E latency | 1.011 | 0.784 | 0.722 | 0.792 | -0.070 | -0.109 | -0.050 | synergy (super-multiplicative gain) | 0.795 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | TTFT | 1.169 | 0.881 | 0.799 | 1.030 | -0.230 | -0.277 | -0.108 | synergy (super-multiplicative gain) | 1.050 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | prefill | 1.267 | 0.777 | 0.607 | 0.984 | -0.377 | -0.735 | -0.104 | synergy (super-multiplicative gain) | 1.044 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | vision | 0.998 | 0.997 | 0.998 | 0.995 | 0.003 | -0.004 | 0.005 | approximately independent | 0.995 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.750 | 500 | peak alloc | 0.983 | 1.000 | 0.983 | 0.983 | 0.000 | 0.000 | 0.000 | approximately independent | 0.983 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | E2E latency | 0.829 | 0.784 | 0.634 | 0.650 | -0.016 | -0.045 | 0.001 | approximately independent | 0.613 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | TTFT | 0.931 | 0.881 | 0.684 | 0.820 | -0.136 | -0.171 | -0.042 | synergy (super-multiplicative gain) | 0.812 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | prefill | 0.833 | 0.777 | 0.390 | 0.647 | -0.258 | -0.491 | -0.055 | synergy (super-multiplicative gain) | 0.610 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | vision | 0.998 | 0.997 | 1.000 | 0.994 | 0.005 | -0.002 | 0.008 | approximately independent | 0.994 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.500 | 500 | peak alloc | 0.983 | 1.000 | 0.983 | 0.983 | 0.000 | 0.000 | 0.000 | approximately independent | 0.983 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | E2E latency | 0.668 | 0.784 | 0.586 | 0.523 | 0.063 | 0.044 | 0.075 | interference (sub-multiplicative gain) | 0.452 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | TTFT | 0.717 | 0.881 | 0.629 | 0.632 | -0.003 | -0.027 | 0.061 | approximately independent | 0.598 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | prefill | 0.453 | 0.777 | 0.286 | 0.352 | -0.066 | -0.178 | 0.025 | approximately independent | 0.230 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | vision | 0.997 | 0.997 | 0.999 | 0.994 | 0.005 | -0.004 | 0.008 | approximately independent | 0.994 |
| textvqa | tuned AWQ dispatch (vs upstream) | C0_sdpa_r100 | 0.250 | 500 | peak alloc | 0.983 | 1.000 | 0.983 | 0.983 | 0.000 | 0.000 | 0.000 | approximately independent | 0.983 |

## Theoretical FLOPs vs measured time (SDPA + VisionZip)

| dataset | dispatch | config_name | retention | prefill_len | prefill_flops_ratio | prefill_time_ratio | ttft_flops_ratio | ttft_time_ratio | vit_share_of_flops_at_100 |
|---|---|---|---|---|---|---|---|---|---|
| pope | tuned(64) | T1_sdpa_awq64_r75 | 0.750 | 304.4 | 0.773 | 0.865 | 0.876 | 0.989 | 0.455 |
| pope | tuned(64) | T2_sdpa_awq64_r50 | 0.500 | 216.0 | 0.547 | 0.636 | 0.753 | 0.879 | 0.455 |
| pope | tuned(64) | T3_sdpa_awq64_r25 | 0.250 | 127.6 | 0.323 | 0.516 | 0.631 | 0.782 | 0.455 |
| pope | tuned(64) | T0_sdpa_awq64_r100 | 1.000 | 392.7 | 1.000 | 1.000 | 1.000 | 1.000 | 0.455 |
| pope | upstream | C1_sdpa_r75 | 0.750 | 304.4 | 0.773 | 0.795 | 0.876 | 0.872 | 0.455 |
| pope | upstream | C2_sdpa_r50 | 0.500 | 216.0 | 0.547 | 0.558 | 0.753 | 0.700 | 0.455 |
| pope | upstream | C3_sdpa_r25 | 0.250 | 127.6 | 0.323 | 0.366 | 0.631 | 0.577 | 0.455 |
| pope | upstream | C0_sdpa_r100 | 1.000 | 392.7 | 1.000 | 1.000 | 1.000 | 1.000 | 0.455 |
| textvqa | tuned(64) | T1_sdpa_awq64_r75 | 0.750 | 741.2 | 0.755 | 0.782 | 0.872 | 0.907 | 0.476 |
| textvqa | tuned(64) | T2_sdpa_awq64_r50 | 0.500 | 507.5 | 0.514 | 0.502 | 0.745 | 0.776 | 0.476 |
| textvqa | tuned(64) | T3_sdpa_awq64_r25 | 0.250 | 273.4 | 0.275 | 0.369 | 0.620 | 0.713 | 0.476 |
| textvqa | tuned(64) | T0_sdpa_awq64_r100 | 1.000 | 975.1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.476 |
| textvqa | upstream | C1_sdpa_r75 | 0.750 | 741.2 | 0.755 | 1.267 | 0.872 | 1.169 | 0.476 |
| textvqa | upstream | C2_sdpa_r50 | 0.500 | 507.5 | 0.514 | 0.833 | 0.745 | 0.931 | 0.476 |
| textvqa | upstream | C3_sdpa_r25 | 0.250 | 273.4 | 0.275 | 0.453 | 0.620 | 0.717 | 0.476 |
| textvqa | upstream | C0_sdpa_r100 | 1.000 | 975.1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.476 |

## Isolated per-config VRAM (10 largest images per dataset)

| config_name | dataset | k | peak_allocated_mb | peak_reserved_mb | context_overhead_mb | device_footprint_mb |
|---|---|---|---|---|---|---|
| C0_sdpa_r100 | textvqa | 10 | 3525.6 | 3786.0 | 1100.5 | 4886.5 |
| C1_sdpa_r75 | textvqa | 10 | 3455.3 | 3752.0 | 1100.5 | 4852.5 |
| C2_sdpa_r50 | textvqa | 10 | 3455.3 | 3742.0 | 1100.5 | 4842.5 |
| C3_sdpa_r25 | textvqa | 10 | 3455.3 | 3730.0 | 1100.5 | 4830.5 |
| E0_eager_r100 | textvqa | 10 | 5391.0 | 5694.0 | 1100.5 | 6794.5 |
| E1_eager_r75 | textvqa | 10 | 5391.0 | 5670.0 | 1100.5 | 6770.5 |
| E2_eager_r50 | textvqa | 10 | 5391.0 | 5662.0 | 1100.5 | 6762.5 |
| E3_eager_r25 | textvqa | 10 | 5391.0 | 5650.0 | 1100.5 | 6750.5 |
| T0_sdpa_awq64_r100 | textvqa | 10 | 3525.6 | 3786.0 | 1100.5 | 4886.5 |
| T1_sdpa_awq64_r75 | textvqa | 10 | 3455.3 | 3750.0 | 1100.5 | 4850.5 |
| T2_sdpa_awq64_r50 | textvqa | 10 | 3455.3 | 3740.0 | 1100.5 | 4840.5 |
| T3_sdpa_awq64_r25 | textvqa | 10 | 3455.3 | 3732.0 | 1100.5 | 4832.5 |
| U2_sdpa_uniform_r50 | textvqa | 10 | 3443.0 | 3614.0 | 1100.5 | 4714.5 |
| U3_sdpa_uniform_r25 | textvqa | 10 | 3443.0 | 3602.0 | 1100.5 | 4702.5 |
| C0_sdpa_r100 | pope | 10 | 3350.2 | 3500.0 | 1100.5 | 4600.5 |
| C1_sdpa_r75 | pope | 10 | 3347.4 | 3512.0 | 1100.5 | 4612.5 |
| C2_sdpa_r50 | pope | 10 | 3347.4 | 3512.0 | 1100.5 | 4612.5 |
| C3_sdpa_r25 | pope | 10 | 3347.4 | 3506.0 | 1100.5 | 4606.5 |
| E0_eager_r100 | pope | 10 | 3757.1 | 3962.0 | 1100.5 | 5062.5 |
| E1_eager_r75 | pope | 10 | 3757.1 | 3956.0 | 1100.5 | 5056.5 |
| E2_eager_r50 | pope | 10 | 3757.1 | 3956.0 | 1100.5 | 5056.5 |
| E3_eager_r25 | pope | 10 | 3757.1 | 3952.0 | 1100.5 | 5052.5 |
| T0_sdpa_awq64_r100 | pope | 10 | 3359.8 | 3544.0 | 1100.5 | 4644.5 |
| T1_sdpa_awq64_r75 | pope | 10 | 3352.3 | 3556.0 | 1100.5 | 4656.5 |
| T2_sdpa_awq64_r50 | pope | 10 | 3347.4 | 3556.0 | 1100.5 | 4656.5 |
| T3_sdpa_awq64_r25 | pope | 10 | 3347.4 | 3550.0 | 1100.5 | 4650.5 |
| U2_sdpa_uniform_r50 | pope | 10 | 3347.4 | 3494.0 | 1100.5 | 4594.5 |
| U3_sdpa_uniform_r25 | pope | 10 | 3347.4 | 3488.0 | 1100.5 | 4588.5 |

## Decode profile (32 forced new tokens, TextVQA, 40 samples)

| config_name | kv_len | prefill_ms_p50 | decode_ms_p50 | ms_per_token_p50 | ms_per_token_p95 | e2e_ms_p50 | decode_share | n |
|---|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | 992.6 | 578.9 | 1945.6 | 62.8 | 121.5 | 3048.9 | 0.6 | 40 |
| C3_sdpa_r25 | 278.1 | 218.7 | 1984.2 | 64.0 | 120.7 | 2629.9 | 0.8 | 40 |
| E0_eager_r100 | 992.6 | 541.9 | 1967.7 | 63.5 | 118.2 | 3242.6 | 0.6 | 40 |
| E3_eager_r25 | 278.1 | 208.4 | 1932.2 | 62.3 | 124.2 | 2828.8 | 0.7 | 40 |
| T0_sdpa_awq64_r100 | 992.6 | 411.0 | 1937.8 | 62.5 | 120.0 | 2743.3 | 0.7 | 40 |
| T3_sdpa_awq64_r25 | 278.1 | 148.8 | 1962.1 | 63.3 | 120.3 | 2533.3 | 0.8 | 40 |

## Run-to-run repeatability (40 samples x 3 repeats)

| dataset | config_name | repeats | ratio_vs_C0_by_repeat | ratio_spread_pp | per_query_cv_pct_p50 | e2e_p50_by_repeat | e2e_p50_spread_pct | ttft_p50_spread_pct | answers_identical_across_repeats |
|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 3 | 1.000 / 1.000 / 1.000 | 0.00 | 8.7 | 598 / 581 / 514 | 14.8 | 6.9 | 1.00 |
| pope | C2_sdpa_r50 | 3 | 0.807 / 0.800 / 0.755 | 5.23 | 11.8 | 482 / 465 / 388 | 21.2 | 14.8 | 1.00 |
| pope | C3_sdpa_r25 | 3 | 0.725 / 0.728 / 0.604 | 12.39 | 16.4 | 434 / 423 / 311 | 31.6 | 25.2 | 1.00 |
| pope | E0_eager_r100 | 3 | 1.237 / 1.250 / 1.132 | 11.84 | 13.7 | 739 / 727 / 582 | 23.0 | 18.2 | 1.00 |
| pope | T0_sdpa_awq64_r100 | 3 | 0.732 / 0.751 / 0.676 | 7.48 | 13.0 | 438 / 437 / 348 | 22.1 | 13.8 | 1.00 |
| pope | T2_sdpa_awq64_r50 | 3 | 0.739 / 0.742 / 0.623 | 11.85 | 18.1 | 442 / 431 / 321 | 30.5 | 18.7 | 1.00 |
| textvqa | C0_sdpa_r100 | 3 | 1.000 / 1.000 / 1.000 | 0.00 | 9.2 | 1392 / 1232 / 1311 | 12.2 | 3.2 | 1.00 |
| textvqa | C2_sdpa_r50 | 3 | 0.847 / 0.833 / 0.818 | 2.95 | 9.4 | 1180 / 1026 / 1072 | 14.0 | 3.0 | 1.00 |
| textvqa | C3_sdpa_r25 | 3 | 0.670 / 0.650 / 0.658 | 2.06 | 11.6 | 933 / 800 / 862 | 15.4 | 6.4 | 1.00 |
| textvqa | E0_eager_r100 | 3 | 1.247 / 1.167 / 1.181 | 8.01 | 16.4 | 1736 / 1438 / 1548 | 19.0 | 11.5 | 1.00 |
| textvqa | T0_sdpa_awq64_r100 | 3 | 0.841 / 0.794 / 0.772 | 6.85 | 10.7 | 1170 / 978 / 1012 | 18.3 | 5.5 | 1.00 |
| textvqa | T2_sdpa_awq64_r50 | 3 | 0.703 / 0.637 / 0.631 | 7.23 | 14.4 | 979 / 785 / 827 | 22.4 | 10.6 | 1.00 |

## AWQ INT4 linear-layer time vs rows M (36 layers, microbenchmark)

| M | fused | dequant | faster |
|---|---|---|---|
| 1 | 33.5 | 59.6 | Triton fused |
| 16 | 33.2 | 60.0 | Triton fused |
| 32 | 33.1 | 63.0 | Triton fused |
| 64 | 59.9 | 58.3 | dequant+cuBLAS |
| 128 | 90.1 | 63.5 | dequant+cuBLAS |
| 192 | 132.8 | 98.6 | dequant+cuBLAS |
| 256 | 172.3 | 87.8 | dequant+cuBLAS |
| 384 | 245.8 | 118.3 | dequant+cuBLAS |
| 512 | 327.1 | 140.3 | dequant+cuBLAS |
| 768 | 482.8 | 187.5 | dequant+cuBLAS |
| 1023 | 646.4 | 245.1 | dequant+cuBLAS |
| 1024 | 652.7 | 246.3 | dequant+cuBLAS |
| 1536 | 991.2 | 363.9 | dequant+cuBLAS |
| 2048 | 1308.8 | 440.7 | dequant+cuBLAS |

