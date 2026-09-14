# EdgeInspect results (dev)

## Macro over categories (mean over reference seeds)

| k | retention | n_seeds | f1 | f1_std | precision | recall | f1_max | auroc | f1_structural | f1_logical | auroc_structural | auroc_logical | pred_anomalous_ratio | ttft_ms_p50 | total_latency_ms_p50 | vram_mb | visual_tokens_after | feasible_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 1 | 0.781 | — | 0.658 | 0.963 | 0.800 | 0.624 | 0.644 | 0.655 | 0.745 | 0.502 | 0.975 | 689.7 | 946.2 | 3442.0 | 502.000 | True |
| 1 | 1.000 | 1 | 0.768 | — | 0.652 | 0.938 | 0.825 | 0.670 | 0.644 | 0.632 | 0.787 | 0.552 | 0.958 | 905.1 | 1157.8 | 3561.1 | 1005.000 | True |
| 4 | 0.500 | 1 | 0.781 | — | 0.658 | 0.963 | 0.800 | 0.515 | 0.644 | 0.655 | 0.660 | 0.370 | 0.975 | 1791.1 | 2041.9 | 3714.0 | 1255.000 | True |
| 4 | 1.000 | 1 | 0.775 | — | 0.655 | 0.950 | 0.817 | 0.688 | 0.644 | 0.644 | 0.807 | 0.568 | 0.967 | 2812.0 | 3052.4 | 4576.8 | 2512.500 | True |

## Pareto-efficient (feasible) configurations on f1

| k | retention | f1 | total_latency_ms_p50 | vram_mb | visual_tokens_after | front_q_lat | front_q_vram | front_q_tokens | front_q_lat_vram |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.781 | 946.2 | 3442.0 | 502.000 | True | True | True | True |

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

