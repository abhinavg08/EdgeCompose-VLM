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

`peak_allocated_mb_p50` = per-query live-tensor peak (interleaved run); `device_footprint_mb` = isolated per-config peak reserved + CUDA context/other processes on the 10 largest images.

| dataset | config_name | quality | quality_ci_low | quality_ci_high | ttft_ms_p50 | ttft_ms_p95 | total_latency_ms_p50 | total_latency_ms_p95 | tokens_per_second_p50 | peak_allocated_mb_p50 | device_footprint_mb | energy_j_median | visual_tokens_after_mean | n_ok | n_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 0.870 | 0.825 | 0.915 | 496.8 | 586.0 | 580.7 | 711.1 | 3.444 | 3321.8 | 4600.5 | 40.9 | 349.005 | 200 | 0 |
| pope | C1_sdpa_r75 | 0.885 | 0.840 | 0.925 | 426.7 | 515.5 | 510.5 | 634.7 | 3.917 | 3320.9 | 4612.5 | 33.9 | 261.780 | 200 | 0 |
| pope | C2_sdpa_r50 | 0.880 | 0.835 | 0.920 | 352.0 | 427.9 | 443.1 | 547.9 | 4.513 | 3320.9 | 4612.5 | 28.9 | 174.515 | 200 | 0 |
| pope | C3_sdpa_r25 | 0.865 | 0.815 | 0.910 | 290.2 | 360.6 | 393.3 | 486.9 | 5.085 | 3320.9 | 4606.5 | 25.1 | 87.225 | 200 | 0 |
| pope | U2_sdpa_uniform_r50 | 0.890 | 0.845 | 0.930 | 343.3 | 410.7 | 435.9 | 542.9 | 4.589 | 3320.7 | 4594.5 | 30.1 | 174.515 | 200 | 0 |
| pope | U3_sdpa_uniform_r25 | 0.885 | 0.840 | 0.930 | 284.2 | 340.8 | 382.7 | 473.9 | 5.226 | 3320.7 | 4588.5 | 24.3 | 87.225 | 200 | 0 |
| pope | E0_eager_r100 | 0.870 | 0.825 | 0.915 | 598.6 | 759.4 | 669.8 | 884.1 | 2.986 | 3519.5 | 5062.5 | 44.7 | 349.005 | 200 | 0 |
| pope | E1_eager_r75 | 0.880 | 0.835 | 0.920 | 528.7 | 676.1 | 614.7 | 799.3 | 3.254 | 3519.5 | 5056.5 | 38.3 | 261.780 | 200 | 0 |
| pope | E2_eager_r50 | 0.875 | 0.830 | 0.920 | 460.7 | 610.7 | 549.8 | 743.6 | 3.637 | 3519.5 | 5056.5 | 33.8 | 174.515 | 200 | 0 |
| pope | E3_eager_r25 | 0.865 | 0.815 | 0.910 | 399.3 | 545.1 | 510.4 | 665.5 | 3.918 | 3519.5 | 5052.5 | 28.9 | 87.225 | 200 | 0 |
| textvqa | C0_sdpa_r100 | 0.780 | 0.726 | 0.834 | 1034.5 | 1352.6 | 1296.2 | 2279.0 | 2.537 | 3506.6 | 4886.5 | 94.4 | 937.195 | 200 | 0 |
| textvqa | C1_sdpa_r75 | 0.773 | 0.718 | 0.827 | 1134.1 | 1312.4 | 1291.8 | 2072.5 | 2.605 | 3445.7 | 4852.5 | 96.2 | 702.830 | 200 | 0 |
| textvqa | C2_sdpa_r50 | 0.738 | 0.679 | 0.796 | 904.0 | 1062.6 | 1069.8 | 1818.9 | 2.989 | 3445.7 | 4842.5 | 78.2 | 468.740 | 200 | 0 |
| textvqa | C3_sdpa_r25 | 0.619 | 0.554 | 0.685 | 697.8 | 827.7 | 880.6 | 1490.6 | 3.633 | 3445.7 | 4830.5 | 62.4 | 234.365 | 200 | 0 |
| textvqa | U2_sdpa_uniform_r50 | 0.752 | 0.697 | 0.808 | 863.2 | 1026.7 | 1026.8 | 1690.1 | 3.135 | 3437.4 | 4714.5 | 73.6 | 468.740 | 200 | 0 |
| textvqa | U3_sdpa_uniform_r25 | 0.633 | 0.570 | 0.698 | 661.6 | 813.4 | 819.5 | 1416.2 | 3.868 | 3437.4 | 4702.5 | 59.9 | 234.365 | 200 | 0 |
| textvqa | E0_eager_r100 | 0.780 | 0.726 | 0.834 | 1344.5 | 1736.5 | 1508.0 | 2479.9 | 2.073 | 5215.7 | 6794.5 | 102.9 | 937.195 | 200 | 0 |
| textvqa | E1_eager_r75 | 0.765 | 0.711 | 0.821 | 1371.3 | 1842.2 | 1564.8 | 2433.2 | 2.211 | 5215.7 | 6770.5 | 104.1 | 702.830 | 200 | 0 |
| textvqa | E2_eager_r50 | 0.738 | 0.679 | 0.796 | 1154.6 | 1626.4 | 1340.5 | 2167.2 | 2.470 | 5215.7 | 6762.5 | 89.3 | 468.740 | 200 | 0 |
| textvqa | E3_eager_r25 | 0.624 | 0.559 | 0.690 | 952.9 | 1338.9 | 1162.3 | 1912.6 | 2.818 | 5215.7 | 6750.5 | 74.6 | 234.365 | 200 | 0 |

## Table 3 - Relative to C0 (SDPA, 100% tokens); paired bootstrap 95% CIs

| config_name | dataset | n_paired | quality_retention_pct | quality_delta | quality_delta_ci_low | quality_delta_ci_high | latency_reduction_pct | latency_reduction_ci_low | latency_reduction_ci_high | ttft_reduction_pct | ttft_reduction_ci_low | ttft_reduction_ci_high | prefill_reduction_pct | prefill_reduction_ci_low | prefill_reduction_ci_high | vision_reduction_pct | vision_reduction_ci_low | vision_reduction_ci_high | vram_reduction_pct | vram_reduction_ci_low | vram_reduction_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | pope | 200 | 100.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| C1_sdpa_r75 | pope | 200 | 101.7 | 0.02 | 0.00 | 0.04 | 12.1 | 10.2 | 13.3 | 14.1 | 10.6 | 15.5 | 21.6 | 19.8 | 23.7 | -1.0 | -2.2 | 0.4 | 0.0 | 0.0 | 0.0 |
| C2_sdpa_r50 | pope | 200 | 101.1 | 0.01 | -0.01 | 0.03 | 23.7 | 21.5 | 25.9 | 29.1 | 26.0 | 30.4 | 45.1 | 43.7 | 46.6 | -0.7 | -2.0 | 1.1 | 0.0 | 0.0 | 0.0 |
| C3_sdpa_r25 | pope | 200 | 99.4 | -0.01 | -0.04 | 0.03 | 32.3 | 30.1 | 36.5 | 41.6 | 38.5 | 42.9 | 62.6 | 61.6 | 63.7 | 0.2 | -1.3 | 1.8 | 0.0 | 0.0 | 0.0 |
| U2_sdpa_uniform_r50 | pope | 200 | 102.3 | 0.02 | 0.00 | 0.05 | 24.9 | 23.0 | 26.7 | 30.9 | 27.9 | 31.9 | 44.3 | 42.7 | 46.2 | -0.5 | -1.5 | 0.9 | 0.0 | 0.0 | 0.0 |
| U3_sdpa_uniform_r25 | pope | 200 | 101.7 | 0.02 | -0.03 | 0.06 | 34.1 | 31.6 | 37.6 | 42.8 | 39.9 | 43.9 | 62.6 | 61.6 | 63.8 | 0.6 | -1.1 | 2.2 | 0.0 | 0.0 | 0.0 |
| E0_eager_r100 | pope | 200 | 100.0 | 0.00 | 0.00 | 0.00 | -15.4 | -21.4 | -14.0 | -20.5 | -26.4 | -17.8 | 5.9 | 4.0 | 8.1 | -78.9 | -93.6 | -70.3 | -6.0 | -6.0 | -6.0 |
| E1_eager_r75 | pope | 200 | 101.1 | 0.01 | -0.01 | 0.03 | -5.9 | -12.2 | -3.0 | -6.4 | -11.9 | -4.2 | 23.5 | 21.9 | 25.7 | -74.8 | -92.8 | -69.0 | -6.0 | -6.0 | -6.0 |
| E2_eager_r50 | pope | 200 | 100.6 | 0.01 | -0.02 | 0.03 | 5.3 | -1.8 | 8.6 | 7.3 | 2.1 | 9.5 | 46.4 | 45.1 | 47.6 | -76.5 | -95.0 | -71.2 | -6.0 | -6.0 | -6.0 |
| E3_eager_r25 | pope | 200 | 99.4 | -0.01 | -0.04 | 0.03 | 12.1 | 5.7 | 16.4 | 19.6 | 12.6 | 22.6 | 64.3 | 63.5 | 65.5 | -76.0 | -94.1 | -71.2 | -6.0 | -6.0 | -6.0 |
| C0_sdpa_r100 | textvqa | 200 | 100.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| C1_sdpa_r75 | textvqa | 200 | 99.2 | -0.01 | -0.03 | 0.02 | 0.3 | -5.5 | 3.7 | -9.6 | -19.8 | -0.2 | -5.7 | -47.8 | 15.4 | 0.1 | -0.6 | 0.5 | 1.7 | 1.7 | 1.8 |
| C2_sdpa_r50 | textvqa | 200 | 94.7 | -0.04 | -0.07 | -0.01 | 17.5 | 13.0 | 21.1 | 12.6 | 5.2 | 19.4 | 30.2 | 2.5 | 43.7 | -0.0 | -0.8 | 0.4 | 1.7 | 1.7 | 1.8 |
| C3_sdpa_r25 | textvqa | 200 | 79.4 | -0.16 | -0.21 | -0.11 | 32.1 | 28.6 | 36.0 | 32.5 | 26.6 | 38.1 | 62.1 | 47.3 | 69.3 | -0.3 | -1.0 | 0.4 | 1.7 | 1.7 | 1.8 |
| U2_sdpa_uniform_r50 | textvqa | 200 | 96.5 | -0.03 | -0.06 | 0.01 | 20.8 | 15.4 | 23.4 | 16.6 | 9.1 | 23.3 | 30.6 | 3.1 | 43.7 | 0.3 | -0.3 | 0.8 | 2.0 | 1.7 | 2.1 |
| U3_sdpa_uniform_r25 | textvqa | 200 | 81.3 | -0.15 | -0.21 | -0.09 | 36.8 | 32.9 | 39.5 | 36.0 | 30.3 | 41.1 | 62.1 | 47.4 | 69.3 | 0.5 | -0.4 | 0.9 | 2.0 | 1.7 | 2.1 |
| E0_eager_r100 | textvqa | 200 | 100.0 | 0.00 | 0.00 | 0.00 | -16.3 | -20.7 | -13.5 | -30.0 | -40.0 | -20.6 | 1.6 | -3.5 | 8.6 | -67.4 | -70.5 | -65.0 | -48.7 | -50.6 | -43.6 |
| E1_eager_r75 | textvqa | 200 | 98.2 | -0.01 | -0.04 | 0.01 | -20.7 | -27.9 | -14.8 | -32.6 | -44.0 | -21.7 | 1.4 | -38.3 | 21.6 | -68.6 | -72.7 | -66.2 | -48.7 | -50.6 | -43.6 |
| E2_eager_r50 | textvqa | 200 | 94.7 | -0.04 | -0.07 | -0.01 | -3.4 | -10.3 | 1.8 | -11.6 | -20.8 | -2.1 | 34.7 | 8.4 | 47.1 | -66.9 | -70.1 | -64.9 | -48.7 | -50.6 | -43.6 |
| E3_eager_r25 | textvqa | 200 | 80.1 | -0.16 | -0.21 | -0.11 | 10.3 | 4.8 | 17.0 | 7.9 | -0.3 | 15.3 | 64.2 | 50.2 | 70.8 | -65.8 | -68.6 | -64.0 | -48.7 | -50.6 | -43.6 |

## Table 4 - Pareto-efficient configurations

Noise-aware dominance: differences within 0.5 pp quality, 3% latency/energy or 1% memory (relative to C0) count as ties; `strict_front_q_lat_mem` is the tolerance-free front.

| dataset | config_name | quality | latency_p50_ms | ttft_p50_ms | peak_alloc_mb | energy_j | front_q_lat_mem | front_q_lat | front_q_mem | front_q_energy | strict_front_q_lat_mem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C1_sdpa_r75 | 0.885 | 510.5 | 426.7 | 3320.9 | 33.9 | False | False | True | False | False |
| pope | U2_sdpa_uniform_r50 | 0.890 | 435.9 | 343.3 | 3320.7 | 30.1 | False | False | True | False | True |
| pope | U3_sdpa_uniform_r25 | 0.885 | 382.7 | 284.2 | 3320.7 | 24.3 | True | True | True | True | True |
| textvqa | C0_sdpa_r100 | 0.780 | 1296.2 | 1034.5 | 3506.6 | 94.4 | True | True | True | True | True |
| textvqa | C1_sdpa_r75 | 0.773 | 1291.8 | 1134.1 | 3445.7 | 96.2 | True | False | True | False | True |
| textvqa | U2_sdpa_uniform_r50 | 0.752 | 1026.8 | 863.2 | 3437.4 | 73.6 | True | True | False | True | True |
| textvqa | U3_sdpa_uniform_r25 | 0.633 | 819.5 | 661.6 | 3437.4 | 59.9 | True | True | False | True | True |

## VisionZip vs uniform subsampling at equal retention (paired)

| dataset | retention | n | quality_visionzip | quality_uniform | delta_vz_minus_uniform | delta_ci_low | delta_ci_high | compress_ms_visionzip_p50 | compress_ms_uniform_p50 | answers_identical_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| pope | 0.500 | 200 | 0.880 | 0.890 | -0.010 | -0.040 | 0.015 | 9.0 | 2.2 | 0.875 |
| pope | 0.250 | 200 | 0.865 | 0.885 | -0.020 | -0.060 | 0.025 | 9.1 | 2.2 | 0.755 |
| textvqa | 0.500 | 200 | 0.738 | 0.752 | -0.014 | -0.046 | 0.018 | 38.8 | 1.7 | 0.740 |
| textvqa | 0.250 | 200 | 0.619 | 0.633 | -0.014 | -0.082 | 0.051 | 39.3 | 1.6 | 0.475 |

## Stage-wise medians and shares

| dataset | config_name | preprocess_ms_p50 | vision_ms_p50 | compress_ms_p50 | prefill_ms_p50 | decode_ms_p50 | total_latency_ms_p50 | decode_ms_per_token_p50 | generated_tokens_p50 | prefill_seq_len_mean | frac_prefill_ge_1024 | preprocess_share | vision_share | compress_share | prefill_share | decode_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | C0_sdpa_r100 | 10.5 | 145.5 | 2.0 | 328.1 | 117.0 | 580.7 | 117.0 | 2.00 | 388.3 | 0.00 | 0.02 | 0.24 | 0.00 | 0.54 | 0.19 |
| pope | C1_sdpa_r75 | 10.4 | 147.0 | 9.0 | 257.1 | 116.8 | 510.5 | 116.8 | 2.00 | 301.1 | 0.00 | 0.02 | 0.27 | 0.02 | 0.48 | 0.22 |
| pope | C2_sdpa_r50 | 11.0 | 146.5 | 9.0 | 180.2 | 117.4 | 443.1 | 117.4 | 2.00 | 213.8 | 0.00 | 0.02 | 0.32 | 0.02 | 0.39 | 0.25 |
| pope | C3_sdpa_r25 | 11.8 | 145.2 | 9.1 | 122.8 | 117.2 | 393.3 | 117.2 | 2.00 | 126.5 | 0.00 | 0.03 | 0.36 | 0.02 | 0.30 | 0.29 |
| pope | U2_sdpa_uniform_r50 | 10.9 | 146.3 | 2.2 | 182.7 | 117.6 | 435.9 | 117.6 | 2.00 | 213.8 | 0.00 | 0.02 | 0.32 | 0.00 | 0.40 | 0.26 |
| pope | U3_sdpa_uniform_r25 | 9.7 | 144.6 | 2.2 | 122.6 | 116.6 | 382.7 | 116.6 | 2.00 | 126.5 | 0.00 | 0.02 | 0.37 | 0.01 | 0.31 | 0.29 |
| pope | E0_eager_r100 | 9.7 | 260.3 | 2.0 | 308.7 | 117.8 | 669.8 | 117.8 | 2.00 | 388.3 | 0.00 | 0.01 | 0.37 | 0.00 | 0.44 | 0.17 |
| pope | E1_eager_r75 | 10.4 | 254.4 | 9.1 | 251.0 | 117.6 | 614.7 | 117.6 | 2.00 | 301.1 | 0.00 | 0.02 | 0.40 | 0.01 | 0.39 | 0.18 |
| pope | E2_eager_r50 | 11.0 | 256.9 | 9.0 | 175.9 | 117.4 | 549.8 | 117.4 | 2.00 | 213.8 | 0.00 | 0.02 | 0.45 | 0.02 | 0.31 | 0.21 |
| pope | E3_eager_r25 | 10.8 | 256.1 | 9.1 | 117.1 | 117.4 | 510.4 | 117.4 | 2.00 | 126.5 | 0.00 | 0.02 | 0.50 | 0.02 | 0.23 | 0.23 |
| textvqa | C0_sdpa_r100 | 20.3 | 397.7 | 1.6 | 623.9 | 183.7 | 1296.2 | 66.6 | 3.00 | 976.6 | 0.48 | 0.02 | 0.32 | 0.00 | 0.51 | 0.15 |
| textvqa | C1_sdpa_r75 | 20.4 | 397.3 | 39.2 | 659.2 | 185.9 | 1291.8 | 65.9 | 3.00 | 742.2 | 0.00 | 0.02 | 0.31 | 0.03 | 0.51 | 0.14 |
| textvqa | C2_sdpa_r50 | 20.1 | 397.8 | 38.8 | 435.6 | 185.8 | 1069.8 | 66.6 | 3.00 | 508.1 | 0.00 | 0.02 | 0.37 | 0.04 | 0.40 | 0.17 |
| textvqa | C3_sdpa_r25 | 19.9 | 398.8 | 39.3 | 236.3 | 182.3 | 880.6 | 65.9 | 3.00 | 273.7 | 0.00 | 0.02 | 0.45 | 0.04 | 0.27 | 0.21 |
| textvqa | U2_sdpa_uniform_r50 | 20.0 | 396.4 | 1.7 | 433.1 | 186.7 | 1026.8 | 66.4 | 3.00 | 508.1 | 0.00 | 0.02 | 0.38 | 0.00 | 0.42 | 0.18 |
| textvqa | U3_sdpa_uniform_r25 | 19.9 | 395.8 | 1.6 | 236.1 | 174.6 | 819.5 | 66.0 | 3.00 | 273.7 | 0.00 | 0.02 | 0.48 | 0.00 | 0.29 | 0.21 |
| textvqa | E0_eager_r100 | 19.9 | 665.6 | 1.7 | 613.7 | 180.8 | 1508.0 | 66.0 | 3.00 | 976.6 | 0.48 | 0.01 | 0.45 | 0.00 | 0.41 | 0.12 |
| textvqa | E1_eager_r75 | 20.1 | 670.7 | 39.5 | 615.4 | 176.0 | 1564.8 | 66.6 | 3.00 | 742.2 | 0.00 | 0.01 | 0.44 | 0.03 | 0.40 | 0.12 |
| textvqa | E2_eager_r50 | 19.9 | 663.6 | 39.3 | 407.4 | 181.0 | 1340.5 | 66.7 | 3.00 | 508.1 | 0.00 | 0.02 | 0.51 | 0.03 | 0.31 | 0.14 |
| textvqa | E3_eager_r25 | 19.9 | 659.4 | 39.2 | 223.6 | 181.0 | 1162.3 | 65.8 | 3.00 | 273.7 | 0.00 | 0.02 | 0.59 | 0.03 | 0.20 | 0.16 |

## POPE detail (yes = positive class)

| config_name | pope_accuracy | pope_precision | pope_recall | pope_f1 | pope_yes_ratio | pope_f1_adversarial | pope_f1_popular | pope_f1_random |
|---|---|---|---|---|---|---|---|---|
| C0_sdpa_r100 | 0.870 | 0.912 | 0.822 | 0.865 | 0.455 | 0.872 | 0.862 | 0.857 |
| C1_sdpa_r75 | 0.885 | 0.915 | 0.851 | 0.882 | 0.470 | 0.872 | 0.900 | 0.877 |
| C2_sdpa_r50 | 0.880 | 0.905 | 0.851 | 0.878 | 0.475 | 0.872 | 0.900 | 0.862 |
| C3_sdpa_r25 | 0.865 | 0.930 | 0.792 | 0.856 | 0.430 | 0.857 | 0.893 | 0.815 |
| U2_sdpa_uniform_r50 | 0.890 | 0.925 | 0.851 | 0.887 | 0.465 | 0.897 | 0.900 | 0.857 |
| U3_sdpa_uniform_r25 | 0.885 | 0.953 | 0.812 | 0.877 | 0.430 | 0.877 | 0.897 | 0.857 |
| E0_eager_r100 | 0.870 | 0.912 | 0.822 | 0.865 | 0.455 | 0.872 | 0.862 | 0.857 |
| E1_eager_r75 | 0.880 | 0.914 | 0.842 | 0.876 | 0.465 | 0.872 | 0.900 | 0.857 |
| E2_eager_r50 | 0.875 | 0.904 | 0.842 | 0.872 | 0.470 | 0.872 | 0.900 | 0.842 |
| E3_eager_r25 | 0.865 | 0.930 | 0.792 | 0.856 | 0.430 | 0.857 | 0.893 | 0.815 |

## Composition interaction (A = VisionZip r, B = SDPA vs eager, baseline = eager 100%)

| dataset | factor_B | baseline | retention | n | metric | R_A | R_B | R_AB | R_A*R_B | I | I_ci_low | I_ci_high | verdict | R_additive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | E2E latency | 0.918 | 0.867 | 0.762 | 0.796 | -0.033 | -0.049 | -0.005 | approximately independent | 0.785 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | TTFT | 0.883 | 0.830 | 0.713 | 0.733 | -0.020 | -0.044 | 0.010 | approximately independent | 0.713 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | prefill | 0.813 | 1.063 | 0.833 | 0.864 | -0.031 | -0.054 | -0.005 | approximately independent | 0.876 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | vision | 0.978 | 0.559 | 0.565 | 0.546 | 0.018 | -0.021 | 0.033 | approximately independent | 0.537 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | E2E latency | 0.821 | 0.867 | 0.662 | 0.712 | -0.050 | -0.079 | -0.030 | synergy (super-multiplicative gain) | 0.688 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | TTFT | 0.770 | 0.830 | 0.588 | 0.639 | -0.051 | -0.076 | -0.021 | synergy (super-multiplicative gain) | 0.600 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | prefill | 0.570 | 1.063 | 0.584 | 0.605 | -0.022 | -0.044 | -0.007 | approximately independent | 0.632 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | vision | 0.987 | 0.559 | 0.563 | 0.552 | 0.011 | -0.025 | 0.030 | approximately independent | 0.546 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | E2E latency | 0.762 | 0.867 | 0.587 | 0.661 | -0.073 | -0.102 | -0.051 | synergy (super-multiplicative gain) | 0.629 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | TTFT | 0.667 | 0.830 | 0.485 | 0.554 | -0.069 | -0.110 | -0.039 | synergy (super-multiplicative gain) | 0.497 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | prefill | 0.379 | 1.063 | 0.398 | 0.403 | -0.005 | -0.014 | 0.005 | approximately independent | 0.442 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | vision | 0.984 | 0.559 | 0.558 | 0.550 | 0.008 | -0.030 | 0.022 | approximately independent | 0.543 |
| pope | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | peak alloc | 1.000 | 0.944 | 0.944 | 0.944 | -0.000 | -0.000 | -0.000 | approximately independent | 0.944 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | E2E latency | 1.038 | 0.860 | 0.857 | 0.892 | -0.035 | -0.062 | 0.005 | approximately independent | 0.897 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | TTFT | 1.020 | 0.769 | 0.844 | 0.785 | 0.059 | -0.008 | 0.122 | approximately independent | 0.789 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | prefill | 1.003 | 1.017 | 1.074 | 1.019 | 0.055 | -0.031 | 0.128 | approximately independent | 1.019 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | vision | 1.008 | 0.598 | 0.597 | 0.602 | -0.005 | -0.015 | 0.005 | approximately independent | 0.605 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.750 | 200 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | E2E latency | 0.889 | 0.860 | 0.709 | 0.764 | -0.055 | -0.085 | -0.025 | synergy (super-multiplicative gain) | 0.748 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | TTFT | 0.859 | 0.769 | 0.672 | 0.661 | 0.012 | -0.038 | 0.061 | approximately independent | 0.628 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | prefill | 0.664 | 1.017 | 0.710 | 0.675 | 0.035 | -0.022 | 0.083 | approximately independent | 0.680 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | vision | 0.997 | 0.598 | 0.598 | 0.596 | 0.002 | -0.009 | 0.012 | approximately independent | 0.595 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.500 | 200 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | E2E latency | 0.771 | 0.860 | 0.584 | 0.662 | -0.079 | -0.106 | -0.046 | synergy (super-multiplicative gain) | 0.630 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | TTFT | 0.709 | 0.769 | 0.519 | 0.545 | -0.026 | -0.070 | 0.014 | approximately independent | 0.478 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | prefill | 0.364 | 1.017 | 0.385 | 0.370 | 0.015 | -0.017 | 0.040 | approximately independent | 0.381 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | vision | 0.991 | 0.598 | 0.599 | 0.592 | 0.007 | -0.003 | 0.017 | approximately independent | 0.588 |
| textvqa | SDPA (vs eager) | E0_eager_r100 | 0.250 | 200 | peak alloc | 1.000 | 0.672 | 0.661 | 0.672 | -0.012 | -0.012 | -0.012 | approximately independent | 0.672 |

## Theoretical FLOPs vs measured time (SDPA + VisionZip)

| dataset | dispatch | config_name | retention | prefill_len | prefill_flops_ratio | prefill_time_ratio | ttft_flops_ratio | ttft_time_ratio | vit_share_of_flops_at_100 |
|---|---|---|---|---|---|---|---|---|---|
| pope | upstream | C1_sdpa_r75 | 0.750 | 301.1 | 0.774 | 0.784 | 0.876 | 0.859 | 0.454 |
| pope | upstream | C2_sdpa_r50 | 0.500 | 213.8 | 0.548 | 0.549 | 0.753 | 0.709 | 0.454 |
| pope | upstream | C3_sdpa_r25 | 0.250 | 126.5 | 0.324 | 0.374 | 0.631 | 0.584 | 0.454 |
| pope | upstream | C0_sdpa_r100 | 1.000 | 388.3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.454 |
| textvqa | upstream | C1_sdpa_r75 | 0.750 | 742.2 | 0.755 | 1.057 | 0.872 | 1.096 | 0.476 |
| textvqa | upstream | C2_sdpa_r50 | 0.500 | 508.1 | 0.514 | 0.698 | 0.745 | 0.874 | 0.476 |
| textvqa | upstream | C3_sdpa_r25 | 0.250 | 273.7 | 0.275 | 0.379 | 0.620 | 0.675 | 0.476 |
| textvqa | upstream | C0_sdpa_r100 | 1.000 | 976.6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.476 |

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

