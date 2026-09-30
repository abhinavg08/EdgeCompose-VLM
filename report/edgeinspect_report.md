# EdgeInspect-VLM: Few-Shot Industrial Visual Inspection with Efficient Vision-Language Models on Memory-Constrained GPUs

*Studying reference scaling, visual-token compression, and deployment trade-offs on an RTX 4060 8 GB GPU*

## Abstract

Industrial inspection lines have abundant images of known-good products but few labelled
defects, which makes few-shot, reference-based inspection attractive. Vision-language models
(VLMs) can compare a query with reference images without task-specific training, but every
additional reference image adds hundreds of visual tokens, so reference count is limited by
GPU memory and latency. We study this trade-off on an RTX 4060 Laptop GPU (8 GB) with
Qwen2.5-VL-3B-Instruct-AWQ, reusing the efficiency stack characterized in EdgeCompose-VLM
(INT4 AWQ language model, bf16 vision tower, SDPA, tuned AWQ kernel dispatch, per-image
VisionZip token compression). On MVTec LOCO AD (five categories, structural and logical
anomalies) we evaluate k ∈ {1, 2, 4, 8} normal references x {100, 75, 50, 25}% visual-token
retention on 200 test queries per configuration, score anomalies with exact answer likelihoods
and calibrate the operating threshold on normal validation images only. Measured on this
hardware, full-resolution prompts stay VRAM-resident only up to k = 4 references; compression
raises the limit to k = 8 / 12 / 16 at 75 / 50 / 25% retention. More references improve anomaly
discrimination: going from 1 to 8 references raises mean AUROC by +0.08 to +0.13 at every
retention level (paired bootstrap CIs exclude zero), while compression at a fixed reference
count has no detectable effect for k ≤ 4. The best VRAM-resident configuration, 8 references at
75% retention, reaches a mean AUROC of 0.697 [0.624, 0.769] versus 0.605 for the best
uncompressed configuration that fits (k = 4; paired difference +0.093 [+0.026, +0.170]). A second,
independent reference seed reproduces the direction of these effects (with a smaller, non-significant
reference gain at 75%) and gives 0.702 for k = 8 at 75%.
Absolute performance remains modest — pushpins stays near chance, and the model's own
NORMAL/ANOMALOUS answers are almost always "ANOMALOUS" — so the system is a research
prototype, not a deployable inspector. The result that transfers is systemic: on an 8 GB GPU,
token compression is what buys the reference context that improves few-shot inspection.

## 1. Introduction

Industrial visual inspection rarely has many labelled defects: defects are rare, varied and
expensive to collect, while images of *known-good* products are abundant. Few-shot,
reference-based inspection uses exactly that asymmetry: show a model a handful of normal
products and ask whether a new one deviates.

Modern VLMs can compare images and reason about object counts, arrangement and damage in
natural language without task-specific training, which in principle covers both *structural*
defects (scratches, contamination, deformation) and *logical* defects (a missing, extra or
misplaced component, a wrong quantity). They are also expensive: every image becomes hundreds
of visual tokens, so a prompt with k reference images and one query carries (k+1) times the
visual-token load, increasing vision-encoder time, prefill latency and memory. On an 8 GB
consumer GPU this cost directly limits how many references can be shown.

EdgeCompose-VLM, the companion study, measured how quantization, token compression, attention
kernels and quantized-kernel dispatch interact for single-image VQA on the same GPU.
EdgeInspect-VLM asks whether those systems effects translate into application capability:
**can an efficient VLM use multiple normal reference images to perform useful industrial
anomaly inspection within an 8 GB memory budget, and does visual-token compression let it use
more of them?** We do not train any model and do not propose a new detector; we measure what an
off-the-shelf quantized VLM can do under a fixed hardware budget and what limits it.

## 2. System

**EdgeCompose-VLM (reused).** An instrumented runner around the unmodified Hugging Face
Qwen2.5-VL modules with CUDA-synchronized stage timing (preprocess, vision encoder, fusion +
compression, LLM prefill, decode), per-query peak-memory accounting and NVML power sampling.
Its measured Pareto-optimal single-image setting is used unchanged:
* **AWQ INT4** language model (official `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`, float16 activations);
* **bf16 vision tower** (EdgeCompose found that float16 overflows in the last ViT block for ~3%
  of images, producing garbage output; the unquantized vision tower runs in its native bf16);
* **PyTorch SDPA** attention (memory-efficient kernel on this build);
* **tuned AWQ dispatch** (dequantize + cuBLAS for ≥ 64 rows instead of AutoAWQ's 1024, from
  EdgeCompose's kernel microbenchmark);
* **VisionZip** visual-token compression, ported from the official Qwen2.5-VL implementation.

**EdgeInspect pipeline.**
```
k references R1..Rk (train/good) ─┐
                                   ├─> prompt: "You are performing industrial visual inspection of a <product>..."
query Iq (test)  ──────────────────┘    "Reference 1 (known-good):" <img> ... "Query image (the product to inspect):" <img>
                                        "...decide whether the query product is NORMAL or ANOMALOUS ... Answer with exactly one word"
      → ViT per image (bf16) → VisionZip per image (retention r, same r for all k+1 images)
      → LLM prefill (AWQ INT4, SDPA) → Stage A: greedy one-word answer (timed: TTFT, latency)
      → exact log P("NORMAL"), log P("ANOMALOUS") by teacher forcing on the prompt's KV cache
        (timed separately, not in latency)  →  anomaly score s = log P(ANOMALOUS) − log P(NORMAL)
      → decision: s > τ_(category,k,r), τ calibrated on normal validation images only
      → [if flagged] Stage B: JSON {status, anomaly_type, issue, explanation} (not in latency)
```
* The runner was generalized from single-image to interleaved multi-image prompts
  (`QwenVLRunner.run_images`); single-image EdgeCompose outputs were re-verified unchanged.
* **Per-image VisionZip.** Attention statistics are computed *within* each image (the ViT's
  full-attention blocks do not attend across images) and each image keeps the fraction r of its
  own tokens; surviving tokens keep their original M-RoPE positions.
* **Answer-likelihood scoring.** The model's greedy answer was "ANOMALOUS" for 3,150 of the 3,200
  test queries of the final grid (98.4%), normal ones included, so the text answer carries almost
  no information. The exact likelihoods of the two allowed answers still rank queries, so their
  log-ratio is the anomaly score. It is a ranking score, not a calibrated probability.
* **Prompt.** Only the product name is given per category (e.g. "box of pushpins"); no
  category-specific rules (such as the breakfast-box composition) are provided — the references
  are the only source of "normal". The prompt was fixed before any evaluation and never tuned.

## 3. Dataset and protocol

**Dataset.** MVTec LOCO AD (Bergmann et al., IJCV 2022; CC BY-NC-SA 4.0 — non-commercial use),
original splits and labels:

| Category | train good | val good | test good | test logical | test structural |
|---|---|---|---|---|---|
| breakfast_box | 351 | 62 | 102 | 83 | 90 |
| juice_bottle | 335 | 54 | 94 | 142 | 94 |
| pushpins | 372 | 69 | 138 | 91 | 81 |
| screw_bag | 360 | 60 | 122 | 137 | 82 |
| splicing_connectors | 360 | 60 | 119 | 108 | 85 |
| **total** | 1,778 | 305 | 575 | 561 | 432 |

*Structural* anomalies are local physical defects (scratches, contamination, deformation,
broken parts). *Logical* anomalies violate a global constraint while each part looks normal: a
missing or extra component, a wrong count, a wrong combination or arrangement (e.g. two
pushpins in one compartment, a missing tangerine, a cable connecting mismatched terminals).
Labels: normal → 0; structural or logical anomaly → 1; per-type metrics use normals + that type.

**Queries.** Per category a fixed stratified subset of the test split (seed 1234): 20 good,
10 logical, 10 structural = 40 queries, 200 over the five categories (50% anomalous). A
development grid (pushpins and splicing_connectors, 60 queries, k ∈ {1,4}, r ∈ {100,50}%) was
run first to validate the pipeline.

**References.** k ∈ {1, 2, 4, 8} images from train/good, sampled with a fixed seed per category
and saved as manifests (`data/manifests/edgeinspect/<category>_k<k>_seed<s>.json`). Sets are
nested (the k = 1 image is the first of the k = 2 set, ...), so changing k changes only how many
references are shown. The same references are used for every retention level. Queries (test)
and references (train) never overlap.

**Grid.** k ∈ {1,2,4,8} x retention ∈ {100, 75, 50, 25}% x 5 categories with reference seed 0
(80 conditions, 4,000 rows including calibration). A variability run with reference seed 1
repeats k ∈ {1, 8} x retention ∈ {100, 75, 50}% (low vs high reference count, uncompressed vs
moderate vs strong compression) for all categories.

**Normal-only threshold calibration.** For each (category, k, retention), 10 validation/good
images — disjoint from warmup, references and queries — are scored; τ is the 90th percentile of
their scores (≈10% false-positive target). No anomalous image is used to set any threshold or
prompt. The resulting precision / recall / F1 / accuracy describe a *deployment operating
point*, not an F1-optimal threshold.

**Metrics.** Positive = anomalous. Primary: **AUROC** of the anomaly score (threshold-free;
0.5 = chance), macro-averaged over the five categories, with 95% stratified bootstrap CIs
(queries resampled within category and label) and paired bootstrap CIs for differences between
configurations (all configurations score the same queries). Also AUPRC (chance = 0.5), the
calibrated operating point, per anomaly type, and the raw greedy answer (reported to document
its bias: with 50% anomalies, "always anomalous" already has F1 = 0.667).

## 4. Systems methodology

**Hardware.** NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MB, 115 W; 15.7 GB system RAM;
Windows 11 (WDDM); PyTorch 2.7.1+cu118, Transformers 4.57.1, AutoAWQ 0.2.9 + triton-windows.
Per-image budget 128-512 visual tokens (`max_pixels = 512·28²`): MVTec LOCO images become ≈500
visual tokens each, so a k-shot prompt carries ≈500·(k+1) visual tokens (1,001 at k = 1, 4,505
at k = 8). This is half of EdgeCompose's single-image budget and is fixed for all runs.

**Latency.** Every query is timed with CUDA-synchronized stage boundaries; TTFT = everything up to
the first generated token; end-to-end latency includes the 5-6-token answer. The likelihood-
scoring pass (median 73 ms) is excluded. All 16 configurations run back-to-back for each query
in rotated order after two warmup images per configuration; medians over the 200 test queries
are reported.

**Memory.** Per query: peak allocated and peak reserved memory (statistics reset before each
query). The runner **releases the caching allocator's pool before every query** (outside the
timed region). We added this after observing that in one long-running process serving all
configurations, blocks cached by the largest prompts grew the process to **10.7 GB reserved —
more than the card** — which slowed *other* configurations (k = 8 at 100%: 73 s/query instead
of ≈6.5 s). With per-query release, a configuration's peak reserved memory is attributable to
it, as in a per-request deployment. Device footprint = peak reserved + the measured CUDA context
and other processes on the GPU (1,100 MB on this machine, which also runs a desktop session and
a browser).

**Residency classes.**
* **VRAM-resident:** footprint ≤ 8,188 MB.
* **Spill/degraded:** footprint > 8,188 MB. On Windows, the NVIDIA driver's default "CUDA sysmem
  fallback" policy pages the excess to shared system RAM instead of raising CUDA OOM, so queries
  complete but slow down sharply. These configurations are reported but treated as *not viable*:
  excluded from Pareto fronts and, by default, from the deployment optimizer.
* **OOM/failure:** a CUDA OOM exception (none occurred with SDPA in this study).

**Isolated memory-scaling probe.** A separate process runs 3 anomalous pushpins queries per
(k, retention) with an emptied cache for k ∈ {1, 2, 4, 8, 12, 16, 24}, stopping a retention's
scan after a > 60 s query or two consecutive non-resident k. A one-off eager-attention check repeats k ∈ {4, 8} x retention ∈ {100, 50}% with the first 2 of the same 3 queries, to test whether the unoptimized attention path changes the feasible reference capacity.

## 5. Results

Final grid: 4,000 rows (3,200 test + 800 calibration), 80/80 conditions x 50 rows, 0 failures,
0 OOM, 0 unparseable answers, 0 degenerate outputs; references verified against manifests.
Tables: `results/edgeinspect/aggregate/final_tables.md`; raw rows archived read-only with
SHA-256 checksums in `results/edgeinspect/raw/archive_final_grid_seed0_2026-10-01/`.

### Table 1 — Main results (seed 0; mean over 5 categories; 200 test queries per configuration)

| k | retention | AUROC [95% CI] | AUPRC | F1_cal | Prec_cal | Rec_cal | Spec_cal | TTFT (ms) | E2E (ms) | VRAM footprint (MB) | Residency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100% | 0.594 [0.511, 0.670] | 0.638 | 0.357 | 0.578 | 0.320 | 0.780 | 965 | 1,353 | 4,945 | resident |
| 1 | 75% | 0.614 [0.532, 0.689] | 0.638 | 0.369 | 0.667 | 0.330 | 0.810 | 863 | 1,257 | 4,917 | resident |
| 1 | 50% | 0.578 [0.494, 0.652] | 0.620 | 0.289 | 0.718 | 0.230 | 0.860 | 746 | 1,137 | 4,797 | resident |
| 1 | 25% | 0.580 [0.500, 0.658] | 0.642 | 0.348 | 0.696 | 0.270 | 0.850 | 666 | 1,055 | 4,787 | resident |
| 2 | 100% | 0.636 [0.554, 0.714] | 0.673 | 0.418 | 0.755 | 0.350 | 0.860 | 1,584 | 1,973 | 5,327 | resident |
| 2 | 75% | 0.643 [0.565, 0.714] | 0.688 | 0.436 | 0.705 | 0.390 | 0.850 | 1,334 | 1,718 | 5,165 | resident |
| 2 | 50% | 0.636 [0.558, 0.709] | 0.661 | 0.300 | 0.521 | 0.270 | 0.840 | 1,135 | 1,527 | 5,055 | resident |
| 2 | 25% | 0.620 [0.541, 0.692] | 0.669 | 0.351 | 0.655 | 0.290 | 0.860 | 969 | 1,355 | 4,921 | resident |
| 4 | 100% | 0.605 [0.522, 0.686] | 0.644 | 0.323 | 0.499 | 0.260 | 0.860 | 2,938 | 3,326 | 6,213 | resident |
| 4 | 75% | 0.613 [0.534, 0.691] | 0.653 | 0.390 | 0.578 | 0.320 | 0.900 | 2,451 | 2,836 | 5,725 | resident |
| 4 | 50% | 0.606 [0.524, 0.680] | 0.642 | 0.353 | 0.600 | 0.280 | 0.900 | 1,941 | 2,334 | 5,495 | resident |
| 4 | 25% | 0.618 [0.540, 0.691] | 0.674 | 0.329 | 0.810 | 0.230 | 0.920 | 1,619 | 2,011 | 5,201 | resident |
| 8 | 100% | 0.723 [0.647, 0.793] | 0.777 | 0.586 | 0.773 | 0.480 | 0.870 | 7,865 | 8,250 | **8,911** | **spill** |
| 8 | **75%** | **0.697 [0.624, 0.769]** | 0.756 | **0.583** | 0.863 | 0.470 | 0.920 | 5,035 | 5,420 | 7,565 | resident |
| 8 | 50% | 0.656 [0.581, 0.727] | 0.721 | 0.511 | 0.786 | 0.400 | 0.860 | 3,812 | 4,191 | 6,487 | resident |
| 8 | 25% | 0.674 [0.601, 0.743] | 0.721 | 0.487 | 0.730 | 0.410 | 0.820 | 2,900 | 3,296 | 5,743 | resident |

`*_cal` = operating point from the normal-only threshold. The raw greedy answer predicted
"ANOMALOUS" for 94-100% of queries in every configuration (F1 0.66-0.68 ≈ the trivial 0.667).

### Table 2 — Paired contrasts of mean AUROC (same 200 queries; 95% paired stratified bootstrap)

| Contrast | ΔAUROC | 95% CI |
|---|---|---|
| k = 1 → 8 at 100% | +0.129 | [+0.047, +0.216] |
| k = 1 → 8 at 75% | +0.083 | [+0.006, +0.163] |
| k = 1 → 8 at 50% | +0.078 | [+0.002, +0.153] |
| k = 1 → 8 at 25% | +0.095 | [+0.014, +0.171] |
| k = 4 → 8 at 75% | +0.084 | [+0.023, +0.152] |
| compression at k ≤ 4 (all 9 contrasts, 100% → 75/50/25%) | −0.016 … +0.020 | every CI contains 0 |
| compression at k = 8: 100% → 75% | −0.026 | [−0.070, +0.018] |
| compression at k = 8: 100% → 50% | −0.067 | [−0.132, −0.002] |
| compression at k = 8: 100% → 25% | −0.049 | [−0.123, +0.024] |
| **k = 4 at 100% (best full-token resident) → k = 8 at 75% (resident)** | **+0.093** | **[+0.026, +0.170]** |
| k = 4 at 100% → k = 8 at 50% | +0.051 | [−0.024, +0.127] |
| k = 4 at 100% → k = 8 at 25% | +0.070 | [−0.017, +0.154] |

### Table 3 — Maximum VRAM-resident reference count (isolated probe, pushpins, SDPA)

| Retention | k tested | Max VRAM-resident k | Footprint at max k | Latency at max k | First non-resident k: footprint, latency |
|---|---|---|---|---|---|
| 100% | 1, 2, 4, 8, 12 | **4** | 6,131 MB | 3.0 s | k = 8: 8,635 MB, 6.4 s; k = 12: 12,275 MB, **104 s** |
| 75% | 1 … 16 | **8** | 7,393 MB | 4.9 s | k = 12: 9,701 MB, 21 s; k = 16: 12,643 MB, 111 s |
| 50% | 1 … 24 | **12** | 7,633 MB | 5.6 s | k = 16: 9,443 MB, 16 s; k = 24: 13,561 MB, 139 s |
| 25% | 1 … 24 | **16** | 7,385 MB | 5.5 s | k = 24: 9,167 MB, 16 s |

**Eager attention vs SDPA at representative points** (isolated, pushpins):

| k | retention | SDPA footprint / latency | eager footprint / latency | residency (SDPA / eager) |
|---|---|---|---|---|
| 4 | 100% | 6,131 MB / 2.95 s | 6,310 MB / 4.75 s | resident / resident |
| 8 | 100% | 8,635 MB / 6.4 s | 9,020 MB / 11.4 s | spill / spill |
| 4 | 50% | 5,469 MB / 1.96 s | 5,940 MB / 3.80 s | resident / resident |
| 8 | 50% | 6,385 MB / 3.66 s | 6,740 MB / 6.64 s | resident / resident |

Eager attention adds 0.18-0.47 GB and 61-94% latency but does not change residency at the probed
points; it was not probed at k = 12 / 50% (SDPA: 7.63 GB, so the extra ≈0.4 GB would plausibly push
it over the limit — not measured).

### Table 4 — Structural vs logical (mean AUROC over categories, normals + that type; 50 anomalies of each type)

| k | retention | AUROC structural [95% CI] | AUROC logical [95% CI] |
|---|---|---|---|
| 1 | 100% | 0.583 [0.491, 0.673] | 0.606 [0.511, 0.702] |
| 2 | 75% | 0.645 [0.544, 0.739] | 0.642 [0.543, 0.740] |
| 4 | 100% | 0.603 [0.500, 0.702] | 0.606 [0.513, 0.699] |
| 8 | 100% (spill) | 0.725 [0.614, 0.818] | 0.721 [0.622, 0.816] |
| 8 | 75% | 0.697 [0.592, 0.797] | 0.697 [0.597, 0.789] |
| 8 | 50% | 0.646 [0.544, 0.750] | 0.666 [0.577, 0.760] |
| 8 | 25% | 0.666 [0.570, 0.751] | 0.683 [0.592, 0.773] |

### Table 5 — Reference-seed variability (seed 0 vs seed 1, mean AUROC)

| k | retention | seed 0 | seed 1 | 2-seed mean | Δ (seed 1 − seed 0) | per-category mean |Δ| (max) |
|---|---|---|---|---|---|---|
| 1 | 100% | 0.594 | 0.636 | 0.615 | +0.042 | 0.099 (0.167) |
| 1 | 75% | 0.614 | 0.660 | 0.638 | +0.046 | 0.110 (0.235) |
| 1 | 50% | 0.578 | 0.576 | 0.577 | −0.002 | 0.082 (0.160) |
| 8 | 100% (spill) | 0.723 | 0.699 | 0.711 | −0.024 | 0.059 (0.185) |
| 8 | 75% | 0.697 | 0.702 | 0.699 | +0.004 | 0.052 (0.120) |
| 8 | 50% | 0.656 | 0.658 | 0.657 | +0.001 | 0.049 (0.080) |

Seed 1 (1,500 rows, 0 failures) paired contrasts, same 200 queries: k = 1 → 8 at 100% +0.063
[+0.002, +0.127], at 75% +0.041 [−0.020, +0.110], at 50% +0.082 [+0.012, +0.156]; compression
at k = 8, 100% → 75%: +0.002 [−0.034, +0.038]; 100% → 50%: −0.042 [−0.094, +0.010]; at k = 1,
100% → 50%: −0.061 [−0.122, −0.005]. With only two seeds, the "standard deviation" of a mean is
|Δ|/√2 (0.001-0.033) and is reported only as an indication.

### Figures (`plots/edgeinspect/`)
* **Figure 1** `fig1_auroc_vs_references.png` — AUROC vs k, one curve per retention (seed-0 CIs; seed-1 hollow markers).
* **Figure 2** `fig2_auroc_vs_vram.png` — AUROC vs device VRAM footprint, 8 GB limit, spill marked.
* **Figure 3** `fig3_reference_capacity.png` — maximum VRAM-resident k per retention.
* **Figure 4** `fig4_structural_vs_logical.png` — structural vs logical AUROC vs retention, per k.
* **Figure 5** `fig5_category_heatmap.png` — per-category AUROC for nine selected configurations.
* **Figure 6** `fig6_auroc_vs_latency.png` — AUROC vs median latency, Pareto front among resident configs.
* `examples.png` — qualitative examples (Section 5.1).

### 5.1 Qualitative examples and demo

Five examples selected by fixed rules from the seed-0 grid (decision = normal-only calibrated
threshold; explanations generated by Stage B and **not verified** against ground-truth masks).
Figure: `plots/edgeinspect/examples.png`; data: `results/edgeinspect/examples.json`.

| Story | Query (truth) | Config | Score vs threshold → decision | Latency / peak alloc | Stage-B explanation |
|---|---|---|---|---|---|
| Easy normal | splicing_connectors good/021 (normal) | k=4, 75% | −0.06 vs +4.88 → NORMAL ✓ | 2.66 s / 4.1 GB | — |
| Clear structural | breakfast_box structural/026 | k=4, 75% | +4.01 vs +0.26 → ANOMALOUS ✓ | 2.58 s / 4.1 GB | "logical — chocolate in the tray" (region plausible, type wrong) |
| Clear logical | juice_bottle logical/113 | k=4, 75% | +4.97 vs +4.40 → ANOMALOUS ✓ | 2.61 s / 4.1 GB | "logical — extra component (a banana) on the label" (partly right) |
| More references help | screw_bag logical/041 | k=1, 100% → k=8, 75% | +1.16 vs +1.37 NORMAL ✗ → +1.80 vs +0.99 ANOMALOUS ✓ | 4.85 s / 5.4 GB (k=8) | "logical — extra nut" |
| Compression hurts | pushpins structural/034 | k=4, 100% → 25% | +3.69 vs +3.11 ANOMALOUS ✓ → +2.76 vs +2.80 NORMAL ✗ | 2.19 s / 3.7 GB (25%) | — |

The compression failure is marginal (0.04 below the threshold); the explanations localize plausible
evidence but confuse anomaly types, so they are illustrations, not a localization result.

**Demo** (`scripts/inspect_demo.py`; real run saved in `results/edgeinspect/demo_output.txt`;
query chosen by a fixed rule — the first logical anomaly in the breakfast_box final manifest —
with the best VRAM-resident configuration):

```text
EdgeInspect-VLM
Model:            Qwen2.5-VL-3B-Instruct-AWQ (INT4 LLM, bf16 vision tower, SDPA, tuned AWQ dispatch)
Hardware:         NVIDIA GeForce RTX 4060 Laptop GPU (8188 MB)
Category:         breakfast_box
Reference count:  8  (train/good: 271, 248, 267, 219, 145, 149, 056, 028; seed 0)
Token retention:  75%  (VisionZip per image; visual tokens 4500 -> 3375)
Prediction:       ANOMALOUS
Anomaly score:    +0.436  = log P(ANOMALOUS) - log P(NORMAL); normal-only calibrated threshold +0.319
Raw model answer: 'ANOMALOUS'
Anomaly type:     logical
Issue:            extra nuts
Explanation:      the right side of the tray has extra nuts that are not present in the other images
                  (explanation pass 19872 ms, not included below)
Performance (classification pass, measured):
  TTFT:           5274 ms  (vision 2129 ms, prefill 2731 ms)
  Latency:        5752 ms
  Peak VRAM:      5475 MB allocated / 6348 MB reserved by the process
```
The ground truth is a logical anomaly; the decision margin is small (+0.12), and the explanation is
model-generated and unverified.

## 6. Analysis

Each answer separates **measured (M)**, **interpretation (I)** and, where relevant, **speculation (S)**.

### RQ1 — Do more references improve anomaly discrimination?
**M.** Yes at the high end: k = 8 beats k = 1 by +0.08 to +0.13 mean AUROC at every retention
level (all paired CIs exclude zero), and beats k = 4 by +0.05 to +0.12. The curve is not
monotonic: k = 2 (0.62-0.64) is slightly above k = 4 (0.61-0.62), within noise. Per category the
gain is concentrated in breakfast_box (0.69 → 0.86-0.87), splicing_connectors (0.57-0.67 at
k ≤ 4 → 0.77-0.85 at k = 8) and juice_bottle (≈0.50 at k = 1 → 0.66-0.72 at k = 8); screw_bag is
flat (0.53-0.70) and **pushpins stays near chance at every k (0.40-0.62)**. **Reference-seed check (M):** with a second, independent reference set the direction holds (k = 1 → 8: +0.063, +0.041 and +0.082 at 100/75/50%; two of three paired CIs exclude 0) but the gain is smaller at 75%. Reference choice matters more with one reference (per-category |Δ| up to 0.235 between seeds) than with eight (≤ 0.185, mean 0.05-0.06), and the mean AUROC of k = 8 / 75% is stable (0.697 vs 0.702).
**I.** The model does use additional references, but only a large reference set changes its
relative preference reliably. Pushpins anomalies (a missing or extra pin in one of 15
compartments, a bent pin) are small relative to the ≈500-token image budget; plausibly they are
not resolved at this resolution (not tested).
**Is 8-shot always better than 4-shot?** In mean AUROC, yes in all four retention levels
(significant in three). Per category, no: screw_bag and pushpins do not improve from k = 4 to 8.

### RQ2 — How much compression is tolerable?
**M.** At a fixed reference count ≤ 4, none of the nine compression contrasts (100 → 75/50/25%)
changes mean AUROC detectably (−0.016 to +0.020; all CIs contain 0). At k = 8, 75% retention is
statistically indistinguishable from 100% (−0.026 [−0.070, +0.018]); 50% loses 0.067
[−0.132, −0.002]; 25% loses 0.049 (n.s.). The most compression-sensitive categories at k = 8
(100% → 25%) are breakfast_box (0.855 → 0.723) and pushpins (0.615 → 0.500); juice_bottle and
splicing_connectors do not degrade (40 queries per category — per-category differences of this
size are within noise).
**I.** For this task and model, ≈75% retention is essentially free and 25-50% costs at most a
few AUROC points. The quality limit comes from the model's discrimination, not from token
pruning.

### RQ3 — Does compression expand the feasible reference capacity?
**M.** Yes. Maximum VRAM-resident k on the 8 GB card: **4 at 100%, 8 at 75%, 12 at 50%, 16 at 25%**
(isolated probe, Table 3, Figure 3). In the grid, k = 8 at 100% has an 8,911 MB footprint and
spills; k = 8 at 75% needs 7,565 MB and stays resident. Beyond the limit, latency degrades by up to
two orders of magnitude (k = 12 at 100%: 104 s per query vs 5.6 s for k = 12 at 50%).
**Capacity translates into quality:** the best resident configuration without compression is
k = 4 at 100% (AUROC 0.605); with 75% retention k = 8 becomes resident and reaches 0.697 — a
paired improvement of +0.093 [+0.026, +0.170]. At 50% and 25% the improvement over k = 4 / 100%
is +0.05 to +0.07 but not significant.
**I.** Memory, not model size, is the binding constraint for multi-image inspection on this
card: the quantized weights occupy 3.3 GB; the remainder is consumed by per-image vision-encoder
activations and the LLM prefill over thousands of visual tokens. VisionZip shrinks the latter.

### RQ4 — Are structural and logical anomalies differently sensitive?
**M.** No difference is detectable. Structural and logical AUROCs track each other within
±0.03 in every configuration (e.g. k = 8 / 75%: 0.697 vs 0.697), both improve with k, and both
show the same mild decline under strong compression at k = 8. With 50 anomalies of each type,
CIs are ±0.10.
**I.** We expected logical anomalies to depend more on references and structural ones more on
resolution; the data do not show it at this sample size. Logical anomalies are not harder for
this model on average (breakfast_box logical AUROC reaches 0.91 at k = 8), but categories differ.

### RQ5 — Which configuration gives the best quality/resource trade-off?
**M.** Among VRAM-resident configurations the AUROC-latency Pareto front is k = 1 / 25%
(0.580, 1.06 s), k = 1 / 75% (0.614, 1.26 s), k = 2 / 50% (0.636, 1.53 s), **k = 8 / 25% (0.674,
3.30 s)** and **k = 8 / 75% (0.697, 5.42 s)**. Under a 7 GB footprint budget (≈ the usable budget
on this machine) the best AUROC is k = 8 / 25% (0.674, 5,743 MB, 3.3 s); k = 8 / 75% needs 7.6 GB. The lowest-latency configuration (k = 1 / 25%, 1.06 s) has an
AUROC of 0.58 — barely above chance — so the fastest setting is **not** acceptable for inspection.
The measured-data optimizer encodes these trade-offs (`results/edgeinspect/optimizer_examples.txt`): with ≤ 7 GB, ≤ 6 s and AUROC ≥ 0.65 it recommends k = 8 / 25% (0.674 [0.601, 0.743], 3.3 s, 5.7 GB); requiring ≥ 4 references and minimizing latency gives k = 4 / 25% (0.618, 2.0 s); a 2 s budget gives k = 2 / 75% (0.643, 1.7 s); and the example target "AUROC ≥ 0.80 within 7 GB and 2 s" is **infeasible** — no measured configuration reaches 0.80, and the optimizer reports every rejection reason rather than a recommendation.
**I.** The practical sweet spot on an RTX 4060 8 GB is "as many references as fit, with moderate
compression": k = 8 at 75% if ≈7.6 GB and ≈5.4 s are acceptable, else k = 8 at 25% (5.7 GB, 3.3 s).

### RQ6 — Does the best generic EdgeCompose configuration remain optimal?
**M.** Partly. The stack itself transfers (SDPA, tuned dispatch, bf16 vision tower; no failures
in 4,000 queries), and 75% retention is again the best trade-off: in EdgeCompose, tuned dispatch
+ VisionZip 75% kept 99.6% of TextVQA accuracy at −27.8% latency; in EdgeInspect, 75% retention
costs no measurable AUROC at a fixed k. But the *reason* to compress changes: for single-image VQA
it buys latency and energy, while memory hardly moves (< 2% peak memory); for multi-image
inspection it buys **reference capacity**, which is what improves quality. The best inspection
configuration therefore spends the savings on more references rather than on speed — at k = 8 /
75% a query takes 5.4 s, four times the k = 1 latency.
**I.** A configuration that is Pareto-optimal for generic single-image benchmarks is the right
*per-image* setting, but application-level optimization must decide what to spend the savings on.

## 7. Systems findings

1. **Memory residency, not model size, decides feasibility.** Quantized weights take 3.3 GB; a
   k = 8, 100% prompt pushes the footprint to 8.9 GB. The limit moves from k = 4 to k = 16 with
   compression (Figure 3).
2. **Silent spill instead of OOM.** On Windows, exceeding VRAM does not raise an error; the driver
   pages to system RAM. The first non-resident step costs little (k = 8 at 100%: 8.25 s vs 5.4 s
   at 75%), the next ones are catastrophic (21-139 s per query). "It ran" is not evidence that
   it fits; footprint and latency must be checked.
3. **Allocator cache growth.** A long-running process serving several prompt sizes accumulated
   10.7 GB of cached blocks, silently pushing itself into the spill regime and slowing unrelated
   configurations by up to ~11x. Releasing the cache per request fixed it; production servers
   need an explicit memory policy (per-request release, a reserved-memory cap, or batching by size).
4. **The vision encoder becomes the bottleneck and the memory floor.** VisionZip prunes after the
   ViT, so vision time grows linearly with the number of images (0.42 s at k = 1, 1.11 s at k = 4,
   2.00 s at k = 8) regardless of retention; at k = 8 / 25% it is 61% of latency. Peak allocated
   memory at 25% retention still grows from 3.4 GB (k = 1) to 4.1 GB (k = 8): the capacity limit at
   low retention is eventually set by the uncompressed vision encoder.
5. **Kernel dispatch.** All prompts here exceed 64 tokens, so every prefill uses the tuned
   dequantize + cuBLAS path. (Interpretation: with AutoAWQ's default 1024 threshold, compressed
   prompts below 1024 tokens — e.g. k = 1 at ≤ 75% — would fall onto the slower fused kernel, as in
   EdgeCompose; not re-measured here.)
6. **Decode is negligible** for this task: the one-word answer takes ≈385 ms (5-6 tokens at ≈64
   ms/token) in every configuration; the exact-likelihood score adds a median 73 ms.
7. **Attention kernel matters for latency more than for capacity here.** Eager attention adds 0.18-0.47
   GB and 61-94% latency at the probed points but did not change residency there (in EdgeCompose's
   single-image setting it added 1.7 GB). With ≈500 visual tokens per image, the per-image ViT attention
   matrices are small; the dominant memory term is the LLM prefill over all images, which compression
   reduces and the attention kernel does not.

## 8. Limitations

* **One VLM family and size** (Qwen2.5-VL-3B-AWQ); no comparison with specialized anomaly
  detectors (PatchCore, WinCLIP), which typically reach much higher AUROC on MVTec LOCO structural
  anomalies. This study is about how a general VLM uses reference context under memory limits,
  not about state-of-the-art detection.
* **One consumer GPU** (RTX 4060 Laptop, Windows); the residency limits include ≈1.1 GB used by the
  CUDA context and other applications on this machine and would shift on a headless or desktop GPU.
* **Limited seeds and samples.** One full reference seed plus one partial seed; 40 test queries per
  category (200 per configuration); per-category results are noisy (±0.15 AUROC).
* **No training, fixed prompt.** Results depend on the prompt wording and the per-image budget
  (≤ 512 visual tokens); neither was tuned (by design, to avoid test leakage).
* **Answer bias and calibration.** The generated answer is almost always "ANOMALOUS"; the score is
  a likelihood ratio, not a calibrated probability; the threshold uses only 10 normal images per
  (category, k, retention), so realized specificity varied from 0.78 to 0.92 around the 0.90 target.
* **No localization evaluation**; Stage-B explanations are qualitative only and can be wrong.
* **No factory deployment**; MVTec LOCO images are clean and aligned.
* **Dataset licence**: CC BY-NC-SA 4.0 — the full dataset is not bundled. Selected images in the qualitative montage retain this license; see [image attribution](../plots/edgeinspect/examples.LICENSE.md).

## 9. Conclusion

On an 8 GB RTX 4060, the number of known-good references a VLM can see is limited by memory
residency, and that limit is set by visual tokens, not by model weights. Visual-token compression
moves it from 4 references at full resolution to 8, 12 or 16, and the extra references pay off:
the best VRAM-resident configuration (8 references, 75% of tokens) improves mean AUROC by +0.09
over the best uncompressed configuration that fits, while compression at a fixed reference count
costs little or nothing measurable. Absolute detection performance of this 3B general VLM remains
modest and uneven across categories, and its generated answers are unusable without likelihood
scoring and normal-only calibration, so EdgeInspect is a research prototype rather than a
production inspector. The transferable finding is systemic: inference optimizations are not only
benchmark speedups — under a fixed hardware budget they determine how much few-shot visual
context an application can use.
