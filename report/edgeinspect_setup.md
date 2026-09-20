## 4. Experimental setup

**Hardware.** NVIDIA RTX 4060 Laptop GPU (8188 MB), 15.7 GB RAM, Windows 11. About 1.1 GB of
device memory is taken by the CUDA context and other processes (display, browser), so the
usable budget for the model process is ≈7 GB. On Windows the NVIDIA driver's default
*CUDA sysmem fallback* policy spills to shared system RAM instead of raising OOM; exceeding
VRAM therefore shows up as footprint > device memory and a latency cliff, which we record
as `over_vram`.

**Model stack (from EdgeCompose).** Qwen2.5-VL-3B-Instruct-AWQ (LLM INT4, float16), vision
tower in bfloat16 (see the EdgeCompose erratum), PyTorch SDPA, AWQ dispatch threshold 64
(EdgeCompose's Pareto-optimal setting), VisionZip applied per image. Per-image budget
128-512 visual tokens (`max_pixels = 512·28²`): MVTec LOCO images (≈1600x1100 px) become
≈493 visual tokens each, so a k-shot prompt carries ≈493·(k+1) visual tokens (986 at k=1,
4437 at k=8). The budget is half of EdgeCompose's single-image budget so that 8-shot prompts
remain runnable on the GPU; it is fixed for all configurations.

**Dataset.** MVTec LOCO AD (Bergmann et al., IJCV 2022; CC BY-NC-SA 4.0), five categories,
original splits and labels: train/good (1,772) for references, validation/good (305) for
warmup and calibration, test (1,568: 575 good, 551 logical, 442 structural) for queries.

| Category | train good | val good | test good | test logical | test structural |
|---|---|---|---|---|---|
| breakfast_box | 351 | 62 | 102 | 83 | 90 |
| juice_bottle | 335 | 54 | 94 | 142 | 94 |
| pushpins | 372 | 69 | 138 | 91 | 81 |
| screw_bag | 360 | 60 | 122 | 137 | 82 |
| splicing_connectors | 360 | 60 | 119 | 108 | 85 |

**Queries.** Per category a fixed stratified subset of the test split (seed 1234): 20 good,
10 logical, 10 structural (40 queries; 200 over the five categories). The development grid
used 60 label-balanced queries for pushpins and splicing_connectors.

**References.** k ∈ {1, 2, 4, 8} normal images from train/good, sampled with a fixed seed
per category; sets are nested (the k=1 image is the first of the k=2 set, ...) and saved as
manifests (`data/manifests/edgeinspect/<category>_k<k>_seed<s>.json`). The same references
are used for every token-retention setting. Seed 0 covers the full grid; seeds 1 and 2
repeat k ∈ {1, 4} x r ∈ {100, 50}% to measure reference-sampling variability.

**Grid.** k ∈ {1, 2, 4, 8} x r ∈ {100, 75, 50, 25}% x 5 categories (80 conditions), all
configurations run back-to-back per query in rotated order (as in EdgeCompose).

**Calibration (normal-only).** For each (category, k, r), 10 validation/good images (disjoint
from warmup, references and queries) are scored; the decision threshold is the 90th
percentile of their log-likelihood ratio (≈10% false-positive target). No anomalous image
is used to set any threshold or prompt.

**Metrics.** Positive = anomalous. Score = log P("ANOMALOUS") − log P("NORMAL") from exact
teacher-forced answer likelihoods. Reported: AUROC (threshold-free), F1 / precision /
recall / balanced accuracy of the calibrated decision (F1_cal), F1 of the raw greedy answer,
F1-max, and per anomaly type (normal + structural, normal + logical). Note that with 50%
anomalous queries the trivial "always anomalous" classifier has F1 = 0.667; F1 and F1-max
must be read against that baseline (AUROC 0.5 = chance). Systems: TTFT, E2E latency,
prefill/vision/decode, peak allocated memory per query, isolated device footprint vs k.
Macro averages are over categories; seed variability is reported as mean ± std.
