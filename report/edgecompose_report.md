# EdgeCompose-VLM: Understanding Compositional Efficiency for Vision-Language Models on Resource-Constrained GPUs

*A systematic hardware-aware study and deployment framework for composing VLM inference
optimizations under strict memory constraints.*

## Abstract

Small, weight-quantized vision-language models (VLMs) fit on consumer GPUs, and many
inference optimizations report large speedups in isolation, but deployments compose them.
We study whether independently effective optimizations add up on genuinely constrained
hardware: Qwen2.5-VL-3B-Instruct with INT4 AWQ weights on an RTX 4060 Laptop GPU (8 GB),
batch size 1. We build an instrumented, reproducible runner that times preprocessing, vision
encoding, token compression, LLM prefill and decode separately, and evaluate 14
configurations — VisionZip visual-token retention (100/75/50/25%) crossed with PyTorch SDPA
vs eager attention and with AutoAWQ kernel dispatch (upstream vs re-tuned) — on 500 TextVQA
and 500 POPE questions each (14,000 interleaved, CUDA-synchronized queries). Visual-token
compression is nearly free in quality down to 75% (TextVQA 99.7%, POPE 100.2% of baseline)
but costs 16.7% of TextVQA accuracy at 25%, while POPE is unaffected. Composition is
stage-dependent: SDPA (vision encoder) and VisionZip (prefill) act on disjoint stages and
their savings add; VisionZip and the AWQ kernel path interact strongly — AutoAWQ's fixed
dispatch threshold made 75% retention 17% *slower* in TTFT on TextVQA, while re-tuning the
threshold to the measured crossover turned the same pruning into a 28% end-to-end latency
reduction at 99.6% quality (27% less GPU energy). After pruning, the vision encoder becomes
the dominant stage. Pareto-efficient configurations are exclusively tuned-dispatch +
VisionZip. We release the framework, raw per-query logs and a constraint-based deployment
optimizer.



## 1. Introduction

Small vision-language models (VLMs) with a few billion parameters, combined with 4-bit
weight quantization, now fit on commodity 8 GB GPUs. A large body of work proposes further
inference optimizations: visual-token pruning or merging (FastV, VisionZip, SparseVLM, ...),
fused or memory-efficient attention kernels (FlashAttention, PyTorch SDPA), and low-bit
weight quantization (AWQ, GPTQ). These techniques are almost always evaluated *in
isolation*, often on data-center GPUs and at batch sizes where the reported bottleneck is
the one the technique targets.

A deployment on a constrained device composes them. Whether independently measured gains
add up, overlap, or interfere is not obvious: each technique acts on a different stage of
the pipeline (vision encoder, LLM prefill, autoregressive decode), and the composed system
may be limited by a stage none of them touches, or by kernel-selection logic in the
quantization library that reacts to the changed tensor shapes.

**Research gap.** Composition of established VLM efficiency techniques on a single
memory-constrained consumer GPU, measured per inference stage with explicit quality and
statistical controls, is under-reported.

**This work.** EdgeCompose-VLM is a reproducible evaluation and deployment framework. On an
RTX 4060 Laptop GPU (8 GB) it composes AWQ INT4 weights (Qwen2.5-VL-3B-Instruct-AWQ),
VisionZip visual-token compression (100/75/50/25% retention) and optimized attention
(PyTorch SDPA vs eager), measures quality on TextVQA and POPE together with stage-wise
latency, TTFT, VRAM and energy, quantifies composition interactions, derives
Pareto-efficient deployment configurations and exposes them through a constraint-based
selector. We do **not** propose a new quantization, token-compression or attention
method; AWQ, VisionZip, SDPA and FlashAttention are prior work.

## 2. Background

**VLM inference.** Qwen2.5-VL encodes an image with a ViT (≈0.67 B parameters by our count
from the config: 32 blocks, width 1280, 14x14 patches; 28 blocks use 112-pixel windowed
attention and 4 — blocks 7/15/23/31 — use full attention). A
PatchMerger fuses 2x2 patches into one *visual token* for the language model. The LLM
(Qwen2.5-3B, 36 layers, GQA 16/2 heads) consumes text and visual tokens with 3-D
multimodal RoPE (M-RoPE). At batch 1 the pipeline is: CPU preprocessing (resize, patchify)
→ vision encoder → token fusion → **prefill** (one forward pass over the whole prompt,
producing the first token) → **decode** (one forward pass per generated token with a KV
cache).

**Visual tokens.** The number of visual tokens scales with image area (here 256-1024 per
image), and for short-answer VQA they dominate the prompt: roughly 900-1000 of ~1000 prefill
tokens on TextVQA.

**Weight quantization.** AWQ (activation-aware weight quantization) stores LLM weights in
4 bits with per-group (128) scales and zero points; activations stay FP16 (W4A16). At run
time each linear layer either runs a fused W4A16 GEMM or dequantizes the weight to FP16 and
calls a dense GEMM. In AutoAWQ this choice is a hard-coded threshold on the number of
activation rows M = batch x sequence length (dequantize if M ≥ 1024).

**Token compression.** VisionZip selects *dominant* visual tokens that receive the most
attention in the vision encoder and merges the remaining ones into a few *contextual*
tokens by key similarity. It is training-free and runs once per image, after the vision
encoder, so it shortens the LLM prompt but does not reduce vision-encoder work.

**Optimized attention.** "Eager" attention materializes the S x S score matrix;
PyTorch SDPA dispatches to fused kernels (FlashAttention, memory-efficient/CUTLASS, or math)
that avoid it. FlashAttention-2 is a separate CUDA package.

**TTFT / prefill / decode.** Time-to-first-token (TTFT) = preprocessing + vision encoder +
fusion/compression + prefill; end-to-end latency adds decode. Prefill is compute-bound
(large GEMMs), decode at batch 1 is memory/launch-bound (weight reads per token).

## 3. Methodology

### 3.1 Hardware and software
NVIDIA GeForce RTX 4060 Laptop GPU (Ada, sm_89, 24 SMs, 8188 MB, 115 W TGP, driver 560.94),
15.7 GB RAM, Windows 11 (WDDM). Python 3.9.24, PyTorch 2.7.1+cu118, Transformers 4.57.1,
AutoAWQ 0.2.9 with triton-windows 3.3.1 kernels. On this PyTorch build SDPA's FlashAttention
kernel is not compiled in, so **SDPA resolves to the memory-efficient kernel**; the
flash-attn package has no binary for this platform and no CUDA toolkit was available to
build it (Section 6). Exact environment: `requirements-lock.txt`, `results/system_info.json`.

### 3.2 Model
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (official checkpoint; LLM INT4, group 128; vision tower
FP16), fully GPU-resident (3251 MB after load). Image budget 256-1024 visual tokens
(`min_pixels = 256·28²`, `max_pixels = 1024·28²`). Batch size 1, single image.

### 3.3 Instrumented inference
Instead of `generate()`, a runner drives the unmodified HF sub-modules (vision tower,
language model, LM head) with its own greedy loop so every stage can be timed. We verified
it reproduces `generate()` token-for-token (pure greedy); the checkpoint's generation config
sets `repetition_penalty = 1.05`, which HF applies even with `do_sample=False`; it is disabled
for every configuration. Stages and their boundaries (each boundary CUDA-synchronized):

| Stage | Contents |
|---|---|
| preprocess | chat template, image resize/patchify (CPU), host→device copy |
| vision | ViT + PatchMerger |
| compress | text embedding, image-token fusion, M-RoPE positions, token compression (incl. VisionZip attention statistics) |
| prefill | LLM forward over the full prompt + LM head + argmax → first token |
| decode | greedy loop with KV cache until EOS or 32 new tokens |

### 3.4 Visual-token compression
VisionZip for Qwen2.5-VL was ported from the official implementation
(`dvlab-research/VisionZip/Qwen2_5_VL`, a fork of the transformers-4.49 modeling file that
does not run on 4.57) by isolating its selection logic: softmax attention of the last
(full-attention) vision block, averaged over heads and summed over queries, pooled over
2x2 merge groups; top-k *dominant* tokens; 5% of tokens as *contextual* tokens formed by
cosine-similarity assignment on keys (`target + mean(assigned)`), exactly the reference
budget split (`contextual = max(int(0.05 N), 1)`, `dominant = int(rN) − contextual`).
Surviving tokens keep their original M-RoPE positions; decode positions continue after the
unchanged maximum position. A deterministic **uniform raster subsampling** control (same
token budget, no attention) isolates the value of VisionZip's selection. Unit tests
verify budgets, that dominant tokens are unmodified, and the window→raster reordering.

### 3.5 Configurations
Main factorial (all INT4 AWQ): visual-token retention {100, 75, 50, 25}% x attention
{SDPA, eager}; a third arm with **tuned AWQ kernel dispatch** (dequantize + cuBLAS for
M ≥ 64 instead of ≥ 1024, chosen from a kernel microbenchmark, Section 4) under SDPA; and
uniform-subsampling controls at 50/25% (SDPA). 14 configurations; FlashAttention-2 configs
(C4-C7) are defined but could not run.

### 3.6 Datasets and metrics
**TextVQA** validation (5,000 questions; lmms-lab parquet release), standard VQA accuracy
with the official EvalAI answer normalization (min(#matches/3, 1) averaged over the 10
leave-one-out subsets); no OCR tokens. **POPE** test (9,000 questions over 500 COCO
images; random/popular/adversarial), accuracy, precision, recall, F1 and yes-ratio with the
original POPE answer parsing. Prompt for both: `<question>\nAnswer the question using a
single word or phrase.` Subsets are prefixes of a seeded permutation (seed 1234; POPE
stratified by setting): development n = 200, final n = 500 per dataset; the 200 are contained
in the 500. Five samples outside the evaluation subset are used for warmup.

### 3.7 Measurement protocol
One model load per sweep; attention implementation and AWQ dispatch are switched in place
(verified on both vision and text sub-configs). For every sample all configurations run
back-to-back in a **rotated order**, so clock/thermal drift and background load affect all
configurations alike. Before each query: CUDA synchronize and reset peak-memory stats.
Per query we log answer, score, stage times, TTFT, E2E latency, generated tokens, tokens/s,
peak allocated/reserved memory, and NVML power/energy; failures are logged, never dropped
(none occurred). Statistics: medians (p50), p95, mean and std; 95% bootstrap CIs for
quality; paired bootstrap CIs (same sample IDs) for deltas and ratios vs the baseline.
Supplementary passes: (i) isolated per-config VRAM (empty cache + reset, 10 largest images),
(ii) decode profiling with 32 forced tokens, (iii) three full repeats on 40 samples for
run-to-run variability, (iv) an AWQ kernel microbenchmark.

**Energy.** Mean NVML power (10 ms polling) x query wall time. NVML's cumulative energy
counter was also recorded but implied average power above the 115 W board limit for
sub-second windows on this laptop, so it is not used.

### 3.8 Composition interaction
For a cost C and baseline 0 (the arm *without* factor B at 100% tokens):
R_A = C_A/C_0, R_B = C_B/C_0, R_AB = C_AB/C_0 and **I(A,B) = R_AB − R_A·R_B**, where A is
VisionZip at retention r and B is either SDPA (vs eager; baseline E0) or tuned AWQ dispatch
(vs upstream; baseline C0). I < 0: composition beats multiplicative independence; I ≈ 0:
multiplicative; I > 0: interference. Because techniques acting on *disjoint* stages
compose additively in absolute time (R_AB ≈ R_A + R_B − 1, which implies I ≈ −(1−R_A)(1−R_B)
< 0), we also report the additive prediction. I is computed on medians with a 95% paired
bootstrap CI (2,000 resamples); we call an effect only when the CI excludes 0.

### 3.9 Pareto analysis and deployment selector
Per dataset, a configuration is Pareto-efficient if no other configuration is at least as
good on quality, median latency and peak allocated memory and strictly better on one. To
avoid declaring noise "dominance", differences within 0.5 pp quality, 3% latency/energy or
1% memory (relative to C0) are treated as ties (a strict front is also reported).
`scripts/optimize.py` filters the measured table by user constraints (VRAM, latency,
TTFT, absolute or baseline-relative quality, per dataset) and ranks feasible configurations
by the chosen objective (quality, latency, memory, energy). No model is trained.


## 4. Results

All numbers: 500 questions per dataset, 14 configurations interleaved per sample (14,000
queries, 0 failures), medians unless stated. Quality CIs are 95% bootstrap; deltas and
reductions are paired-bootstrap vs C0. Full tables: `results/aggregate/tables_500.md`.

### Table 1 — System configuration
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MB, sm_89, driver 560.94 |
| System RAM / OS | 15.7 GB / Windows 11 (WDDM) |
| CUDA (torch build) / PyTorch / Transformers | 11.8 / 2.7.1+cu118 / 4.57.1 |
| AWQ kernels | AutoAWQ 0.2.9 + triton-windows 3.3.1 (Triton GEMM / dequantize) |
| Model / quantization | Qwen2.5-VL-3B-Instruct-AWQ; LLM INT4 (w4, g128), vision FP16; 3251 MB resident |
| SDPA kernels available | memory-efficient, math (no flash on this build); flash-attn not installable |

### Table 2 — Main results (selected; all 14 configs in `tables_500.md`)
| Dataset | Config | Quality [95% CI] | TTFT | E2E | tok/s | Peak alloc | Device footprint | Energy |
|---|---|---|---|---|---|---|---|---|
| TextVQA | **C0** SDPA, 100% | 0.775 [0.739, 0.808] | 957 ms | 1245 ms | 2.9 | 3507 MB | 4887 MB | 92.4 J |
| TextVQA | C1 SDPA, VZ 75% | 0.772 [0.736, 0.807] | 1119 ms | 1259 ms | 2.7 | 3446 MB | 4853 MB | 96.2 J |
| TextVQA | C2 SDPA, VZ 50% | 0.752 [0.716, 0.787] | 891 ms | 1032 ms | 3.2 | 3446 MB | 4843 MB | 78.3 J |
| TextVQA | C3 SDPA, VZ 25% | 0.646 [0.604, 0.685] | 687 ms | 832 ms | 3.9 | 3446 MB | 4831 MB | 62.3 J |
| TextVQA | T0 tuned, 100% | 0.775 [0.739, 0.808] | 843 ms | 976 ms | 3.4 | 3507 MB | 4887 MB | 73.5 J |
| TextVQA | **T1 tuned, VZ 75%** | 0.772 [0.735, 0.806] | 765 ms | 899 ms | 3.8 | 3446 MB | 4851 MB | 67.6 J |
| TextVQA | T2 tuned, VZ 50% | 0.750 [0.715, 0.786] | 655 ms | 789 ms | 4.3 | 3446 MB | 4841 MB | 58.6 J |
| TextVQA | T3 tuned, VZ 25% | 0.646 [0.605, 0.685] | 602 ms | 730 ms | 4.5 | 3446 MB | 4833 MB | 54.2 J |
| TextVQA | E0 eager, 100% | 0.774 [0.739, 0.807] | 1313 ms | 1468 ms | 2.3 | 5216 MB | 6795 MB | 102.4 J |
| TextVQA | U3 uniform 25% | 0.578 [0.537, 0.618] | 655 ms | 793 ms | 4.1 | 3437 MB | 4703 MB | 58.1 J |
| POPE | **C0** SDPA, 100% | 0.848 [0.818, 0.878] | 459 ms | 530 ms | 3.8 | 3322 MB | 4601 MB | 39.4 J |
| POPE | C2 SDPA, VZ 50% | 0.850 [0.820, 0.878] | 322 ms | 398 ms | 5.0 | 3321 MB | 4613 MB | 27.7 J |
| POPE | C3 SDPA, VZ 25% | 0.836 [0.804, 0.868] | 265 ms | 332 ms | 6.0 | 3321 MB | 4607 MB | 23.7 J |
| POPE | T0 tuned, 100% | 0.848 [0.818, 0.878] | 296 ms | 371 ms | 5.4 | 3346 MB | 4645 MB | 25.9 J |
| POPE | **T2 tuned, VZ 50%** | 0.850 [0.820, 0.878] | 260 ms | 325 ms | 6.2 | 3329 MB | 4657 MB | 23.8 J |
| POPE | T3 tuned, VZ 25% | 0.834 [0.802, 0.866] | 232 ms | 299 ms | 6.7 | 3321 MB | 4651 MB | 20.6 J |
| POPE | E0 eager, 100% | 0.848 [0.818, 0.878] | 533 ms | 606 ms | 3.3 | 3520 MB | 5063 MB | 43.7 J |

Device footprint = isolated per-config peak reserved on the 10 largest images + 1.1 GB CUDA
context and other GPU processes (a browser was running).

### Table 3 — Relative to C0 (paired bootstrap 95% CI)
| Dataset | Config | Quality retention | Quality Δ (pp) | E2E latency ↓ | TTFT ↓ | Prefill ↓ | Vision ↓ | Peak alloc ↓ |
|---|---|---|---|---|---|---|---|---|
| TextVQA | C1 VZ 75% | 99.7% | −0.2 [−1.6, +1.1] | −1.1% [−5.4, 1.4] | **−16.9%** [−21.3, −5.0] | −26.7% [−50.1, 10.8] | 0.2% | 1.7% |
| TextVQA | C2 VZ 50% | 97.0% | −2.3 [−4.1, −0.6] | 17.1% [13.5, 19.0] | 6.9% [3.8, 15.2] | 16.7% [1.7, 40.2] | 0.2% | 1.7% |
| TextVQA | C3 VZ 25% | 83.3% | −12.9 [−15.9, −9.9] | 33.2% [30.4, 34.6] | 28.3% [25.8, 34.6] | 54.7% [46.8, 67.2] | 0.3% | 1.7% |
| TextVQA | T0 tuned 100% | 100.0% | 0.0 | 21.6% [18.2, 23.5] | 11.9% [8.8, 21.6] | 22.3% [3.5, 44.6] | 0.3% | 0.0% |
| TextVQA | T1 tuned + VZ 75% | 99.6% | −0.3 [−1.6, +1.0] | **27.8%** [24.8, 29.8] | 20.1% [17.2, 28.5] | 39.3% [28.8, 61.3] | 0.2% | 1.7% |
| TextVQA | T2 tuned + VZ 50% | 96.9% | −2.4 [−4.2, −0.7] | 36.6% [33.8, 38.1] | 31.6% [29.4, 37.7] | 61.0% [54.3, 72.5] | 0.0% | 1.7% |
| TextVQA | T3 tuned + VZ 25% | 83.4% | −12.8 [−15.9, −9.9] | 41.4% [38.7, 42.6] | 37.1% [35.1, 42.7] | 71.4% [66.5, 79.4] | 0.1% | 1.7% |
| TextVQA | E0 eager 100% | 99.9% | −0.1 [−0.2, 0.0] | −17.9% [−21.8, −16.3] | −37.2% | 5.9% | **−65.6%** | **−48.7%** |
| POPE | C1 VZ 75% | 100.2% | +0.2 [−0.8, +1.0] | 11.0% [9.5, 12.1] | 12.8% | 20.5% | −0.2% | 0.0% |
| POPE | C2 VZ 50% | 100.2% | +0.2 [−1.0, +1.4] | 24.9% [23.4, 26.4] | 30.0% | 44.2% | 0.7% | 0.0% |
| POPE | C3 VZ 25% | 98.6% | −1.2 [−3.4, +0.8] | 37.3% [35.8, 40.1] | 42.3% | 63.4% | 0.4% | 0.0% |
| POPE | T0 tuned 100% | 100.0% | 0.0 | 30.0% [29.0, 31.7] | 35.6% | 49.8% | −0.1% | −0.7% |
| POPE | T2 tuned + VZ 50% | 100.2% | +0.2 [−1.0, +1.4] | **38.7%** [37.6, 40.9] | 43.4% | 68.0% | 0.6% | −0.2% |
| POPE | T3 tuned + VZ 25% | 98.3% | −1.4 [−3.6, +0.8] | 43.6% [42.9, 45.6] | 49.6% | 74.1% | −1.1% | 0.0% |
| POPE | E0 eager 100% | 100.0% | 0.0 | −14.5% [−16.8, −13.0] | −16.1% | 2.9% | −69.8% | −6.0% |

(Positive = reduction/improvement vs C0; quality Δ in percentage points.)

### Table 4 — Pareto-efficient configurations (quality, median latency, peak allocated memory)
Noise-aware dominance (ties within 0.5 pp quality, 3% latency, 1% memory):

| Dataset | Config | Quality | E2E (ms) | TTFT (ms) | Peak alloc (MB) | Energy (J) |
|---|---|---|---|---|---|---|
| TextVQA | T1 tuned + VZ 75% | 0.772 | 899 | 765 | 3446 | 67.6 |
| TextVQA | T2 tuned + VZ 50% | 0.750 | 789 | 655 | 3446 | 58.6 |
| TextVQA | T3 tuned + VZ 25% | 0.646 | 730 | 602 | 3446 | 54.2 |
| POPE | T2 tuned + VZ 50% | 0.850 | 325 | 260 | 3329 | 23.8 |
| POPE | T3 tuned + VZ 25% | 0.834 | 299 | 232 | 3321 | 20.6 |

The baseline C0, every eager configuration and every upstream-dispatch configuration are
dominated. (Without tolerance, C1 and U2 also appear on fronts through sub-noise differences.)

### Stage-wise latency (medians, ms; share of the stacked total)
| Dataset | Config | preprocess | vision | compress | prefill | decode | E2E |
|---|---|---|---|---|---|---|---|
| TextVQA | C0 | 20 (2%) | 395 (37%) | 1 (0%) | 518 (48%) | 146 (14%) | 1245 |
| TextVQA | C3 VZ 25% | 20 (2%) | 394 (48%) | 38 (5%) | 235 (28%) | 142 (17%) | 832 |
| TextVQA | T3 tuned + VZ 25% | 20 (3%) | 395 (53%) | 38 (5%) | 148 (20%) | 146 (20%) | 730 |
| TextVQA | E0 eager | 20 (2%) | 654 (50%) | 2 (0%) | 488 (37%) | 141 (11%) | 1468 |
| POPE | C0 | 8 (2%) | 129 (25%) | 1 (0%) | 321 (61%) | 64 (12%) | 530 |
| POPE | T3 tuned + VZ 25% | 8 (3%) | 131 (45%) | 6 (2%) | 83 (28%) | 65 (22%) | 299 |

Decode profile (32 forced tokens, 40 TextVQA samples): **62.5-64.0 ms/token for every
configuration**, independent of KV length (278 vs 993 tokens) and attention backend.

### AWQ kernel microbenchmark (sum over the 36-layer LLM's INT4 linears)
| rows M | 1 | 32 | 64 | 128 | 256 | 512 | 1023 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| Triton split-K fused GEMM (ms) | 33.5 | 33.1 | 59.9 | 90.1 | 172.3 | 327.1 | 646.4 | 652.7 | 1308.8 |
| Triton dequantize + cuBLAS (ms) | 59.6 | 63.0 | 58.3 | 63.5 | 87.8 | 140.3 | 245.1 | 246.3 | 440.7 |

AutoAWQ uses the fused kernel for M < 1024 — 2.6x slower than the alternative at M ≈ 1000.

### Erratum — fp16 overflow in the vision tower (found after the final sweep)
**M.** In the reported sweeps the vision tower ran in float16 (the LLM must be float16 for the
AWQ kernels). For 16 of the 500 TextVQA images (3.2%; none on POPE) the residual stream of
the last ViT block (block 31) exceeds the float16 range → Inf → NaN, and the model emits
`!!!!` in every configuration (only 12 and 9 of them under uniform subsampling, which drops
some of the offending tokens). A paired re-run of 100 samples (the 16 + 84 random) with the
vision tower in **bfloat16** — the checkpoint's native dtype; the ViT is not quantized —
removes all 16 failures (C0 accuracy on them 0.0 → 0.806), leaves healthy samples
essentially unchanged (0.790 → 0.798, 96% identical answers) and changes neither vision
time (395 → 385 ms) nor peak memory (±3 MB) materially (`results/aggregate/vision_dtype_erratum.csv`).
The code default is now a bfloat16 vision tower (`--vision-dtype float16` reproduces the
reported sweep).

**Corrected TextVQA quality** (final sweep, the 484 samples on which no configuration
overflowed, paired; `results/aggregate/textvqa_quality_clean484.csv`):

| Config | Reported (n=500) | Corrected (n=484) | Kept vs C0 | Δ vs C0 (pp) [95% CI] |
|---|---|---|---|---|
| C0 SDPA 100% | 0.775 | 0.800 | 100% | — |
| C1 VZ 75% | 0.772 | 0.798 | 99.7% | −0.2 [−1.6, +1.1] |
| C2 VZ 50% | 0.752 | 0.776 | 97.0% | −2.4 [−4.2, −0.5] |
| C3 VZ 25% | 0.646 | 0.667 | 83.3% | −13.3 [−16.4, −10.3] |
| T1 tuned VZ 75% | 0.772 | 0.797 | 99.6% | −0.3 [−1.6, +1.0] |
| T2 tuned VZ 50% | 0.750 | 0.775 | 96.9% | −2.5 [−4.4, −0.7] |
| U2 uniform 50% | 0.733 | 0.751 | 93.9% | −4.9 [−7.1, −2.7] |
| U3 uniform 25% | 0.578 | 0.586 | 73.2% | −21.5 [−25.4, −17.6] |

Absolute TextVQA accuracy rises by ~2.5 pp for every configuration and all relative
conclusions stand. The one comparison the bug biased is VisionZip vs uniform (uniform
failed on fewer samples): on clean samples VisionZip leads by **+2.5 pp [+0.1, +5.2] at 50%**
(previously reported as not significant) and **+8.1 pp [+3.7, +12.3] at 25%**. Latency,
memory and energy results are unaffected (the NaN samples take the same compute path).

### Figures
Fig. 1 quality vs latency (`plots/fig1_quality_vs_latency_500.png`), Fig. 2 quality vs memory,
Fig. 3 stage-wise latency, Fig. 4 retention vs quality, Fig. 5 retention vs TTFT, Fig. 6
quality vs energy, Fig. 7 interaction heatmap, Fig. 8 theory vs measured, Fig. 9 prefill vs
length, Fig. 10 AWQ kernel crossover.

## 5. Analysis

Each answer separates **measured fact (M)**, **plausible systems explanation (E)** and,
where relevant, **speculation (S)**.

### RQ1 — Visual-token reduction on an already-quantized VLM
**M.** On TextVQA, VisionZip keeps 99.7% of baseline accuracy at 75% of visual tokens
(Δ −0.2 pp, CI [−1.6, +1.1]), 97.0% at 50% (−2.3 pp, significant) and 83.3% at 25% (−12.9 pp).
On POPE, accuracy is unchanged within noise at every ratio (98.6-100.2%, all CIs include 0).
TextVQA is therefore disproportionately hurt by aggressive reduction. VisionZip's attention-
based selection pays off on TextVQA: on the overflow-free samples (erratum) it beats uniform
subsampling by +2.5 pp [+0.1, +5.2] at 50% and +8.1 pp [+3.7, +12.3] at 25%; on POPE the
difference is not significant, and on the 200-sample development subset it was not visible
at all — a reminder that small dev sets can hide real effects.
**E.** Reading text needs small, high-resolution regions; once those tokens are dropped the
answer cannot be recovered, whereas POPE's object-presence questions tolerate coarse
evidence. Qwen2.5-VL's PatchMerger already fuses 2x2 patches, so each remaining token is
information-dense (consistent with the VisionZip authors' note that gains on Qwen2.5-VL are
smaller than on LLaVA).
**POPE hallucination (M).** Precision stays ≈0.90 at every ratio; at 25% recall drops
(0.787 → 0.760) and the yes-ratio falls (0.444 → 0.428): the compressed model answers "no"
slightly more often, i.e. it becomes more conservative, not more hallucination-prone. These
shifts are within noise at n=500.

### RQ2 — Are token compression and optimized attention additive?
**M.** SDPA (vs eager) and VisionZip act on *different stages*: SDPA cuts vision-encoder
time by 40% (R_B = 0.60 TextVQA, 0.59 POPE) and leaves prefill essentially unchanged (eager
prefill is even 3-6% faster at these lengths); VisionZip leaves the vision encoder unchanged
and cuts prefill. Stage-level interactions are ≈0 (|I| ≤ 0.016 on prefill and vision). At
the end-to-end level the composition is **additive in absolute time**: e.g. TextVQA 25%:
R_AB = 0.566 vs additive prediction 0.591 and multiplicative 0.630; POPE 50%: R_AB = 0.656 =
additive 0.656. In the multiplicative framing this shows up as small *negative* I
(−0.03 to −0.06, "super-multiplicative"), which is the expected signature of savings on
disjoint stages rather than true synergy. The one exception is TextVQA at 75%, where the
TTFT interaction is +0.105 because the compressed prompt crosses the AWQ dispatch threshold
(RQ5).
**E.** The eager vision encoder materializes 4096x4096 attention matrices in its four
full-attention blocks; SDPA's fused kernel avoids that traffic, which is also why eager
needs +1.7 GB of memory on TextVQA.

### RQ3 — Which stage becomes the bottleneck?
**M.** Baseline TextVQA: prefill 48%, vision 37%, decode 14%. After VisionZip 25%: vision
48%, prefill 28%; with tuned dispatch as well: vision 53%, prefill 20%, decode 20%. On
POPE the vision share rises from 25% to 45%. Under eager attention the vision encoder is
already 50% at 100% tokens and 61% at 25%. Decode costs ~64 ms per token regardless of KV
length, configuration or backend.
**E.** VisionZip prunes *after* the vision encoder, so it can only shrink the LLM part;
once prefill is small, the ViT (≈0.67 B FP16 parameters over 4 patches per visual token)
and the fixed per-token decode (every INT4 weight is read and every layer's kernels are
launched once per generated token at batch 1)
dominate. Further TTFT gains therefore require vision-side reduction (smaller input
resolution or in-ViT pruning), not more LLM-token pruning.

### RQ4 — Pareto frontier on the RTX 4060 8 GB
**M.** Efficient configurations are exclusively *tuned-dispatch + VisionZip*: T1/T2/T3 on
TextVQA and T2/T3 on POPE (Table 4). The unmodified baseline is dominated by T1 on TextVQA
(same quality within 0.3 pp, 28% lower latency) and by T0-T2 on POPE. Memory is not a
discriminating axis for single-image VQA: all SDPA configurations sit within 2% of each
other (3.32-3.51 GB allocated), and eager is dominated everywhere.

### RQ5 — Do individual gains predict composed gains?
**M.** Not in general. Three regimes were measured:
1. *Disjoint stages* (SDPA x VisionZip): composed gain ≈ sum of individual absolute savings.
2. *Same stage, same kernel regime* (tuned dispatch x VisionZip on POPE, all prompts < 1024
   tokens): **interference**, I = +0.05 to +0.13 on E2E latency (CIs exclude 0). Predicted
   E2E ratio at 50%: 0.700 x 0.751 = 0.526; measured 0.613.
3. *One technique changes the other's kernel regime* (tuned dispatch x VisionZip on
   TextVQA): **synergy**, I = −0.070 (E2E), −0.230 (TTFT), −0.377 (prefill) at 75%.
   Measured alone under upstream AutoAWQ, VisionZip-75% makes TTFT **17% worse**; composed
   with tuned dispatch the same pruning gives a 28% end-to-end reduction.
**E.** AutoAWQ switches kernels at M = 1024 rows. Uncompressed TextVQA prompts (975 tokens on
average; 49% ≥ 1024) often run on the fast dequantize + cuBLAS path; pruning to ~740 tokens
moves every prompt onto the fused Triton GEMM, which is 2.6x slower per row at this size.
With the threshold at the measured crossover (M = 64), prefill follows the FLOP count again
(T2 prefill ratio 0.502 vs FLOP ratio 0.514). On POPE the tuned path pays a roughly constant dequantization cost per
forward pass (the microbenchmark's dequantize + cuBLAS total is ≈58-64 ms for M ≤ 128), so at short prompts prefill stops scaling with tokens
(T3: measured 0.516 vs FLOP 0.323) — the source of the interference.

### Additional questions
1. **Does token compression reduce latency on the RTX 4060?** Yes at 50% and 25% (17-44%
   E2E), but **not at 75% on TextVQA** with upstream AutoAWQ (−1.1% E2E, +17% TTFT).
2. **Are benefits concentrated in TTFT/prefill?** Yes: vision is unchanged (±1%), decode per
   token is unchanged; all savings come from prefill (up to 74%).
3. **Does aggressive reduction disproportionately hurt TextVQA?** Yes: −12.9 pp at 25% vs
   −1.2 pp (n.s.) on POPE.
4. **Hallucination on POPE?** No increase detected; slightly more "no" answers at 25%.
5. **Does the attention backend change the usefulness of compression?** Barely in relative
   terms (compression acts on prefill, SDPA on vision), but SDPA removes 1.7 GB of memory;
   the *quantization kernel* backend changes compression's usefulness far more (RQ5).
6. **Does theoretical compute reduction match measured speedup?** Only within one kernel
   regime: POPE upstream prefill 0.558 vs FLOPs 0.547 at 50%; TextVQA upstream 1.267 vs 0.755
   at 75%. TTFT savings are capped by the vision encoder (≈46-48% of TTFT FLOPs).
7. **Does the 8 GB constraint change the optimal configuration?** Not for single-image VQA at
   ≤1024 visual tokens: the optimum (T1/T2) uses 3.45 GB allocated / 4.85 GB device. The
   constraint does rule out complacency with eager attention (6.8 GB device footprint on
   TextVQA, i.e. ~1.4 GB headroom) and becomes binding for multi-image prompts (EdgeInspect).

**Energy (M).** GPU energy per query follows latency: TextVQA 92.4 J (C0) → 67.6 J (T1,
−27%); POPE 39.4 J → 23.8 J (T2, −40%).

**Variability (M).** Over three repeated 40-sample sessions, answers were 100% identical;
absolute median latencies moved 12-32% between sessions and config-to-config ratios by
1-5 pp in two sessions and up to 12 pp in one (POPE, smallest configurations). Within-
session interleaving makes the comparisons above valid; exact millisecond values are
machine-state dependent.



## 6. Limitations

1. **One GPU, one OS, one software stack.** All numbers come from a single RTX 4060 *Laptop*
   GPU under Windows/WDDM with PyTorch 2.7.1+cu118. Laptop GPUs change clocks with
   temperature and power source; a desktop 4060, Linux, or a CUDA 12 PyTorch build (which
   ships SDPA's FlashAttention kernel) can shift absolute latencies and some ratios.
2. **FlashAttention-2 was not evaluated.** No flash-attn binary exists for Python 3.9 /
   torch 2.7 / CUDA 11.8 on Windows and no CUDA toolkit was available to build it. "Optimized
   attention" therefore means PyTorch SDPA (memory-efficient kernel) *vs eager*; FA2 configs
   C4-C7 are defined but unrun. The eager baseline is a deliberately unoptimized reference,
   not a competitive alternative.
3. **Run-to-run variability.** Across three repeated sessions, absolute median latencies moved
   by 12-32% and some config-to-config latency ratios by up to ~12 pp (POPE, smallest
   configurations), while answers were identical. All main comparisons are within one
   interleaved session (valid for relative claims); absolute milliseconds and exact ratios
   should be read as indicative for this machine state.
4. **Sample sizes.** Final results use 500 questions per dataset (prefix of a seeded
   permutation), not the full splits (5,000 TextVQA / 9,000 POPE). Quality CIs are about
   ±3-4 pp; small quality differences (≤ 2 pp) are generally not significant.
5. **Short answers.** The prompts request a single word or phrase, so decode is 2-4 tokens.
   Conclusions about bottlenecks apply to short-answer VQA; long-form generation is
   decode-dominated (see the 32-token decode profile) and would change the stage mix.
6. **One model family and size.** Only Qwen2.5-VL-3B-AWQ was studied. Its PatchMerger
   already compresses 4 patches into 1 token, which the VisionZip authors note reduces
   VisionZip's benefit for Qwen2.5-VL; results may differ for LLaVA-style models with
   576+ unmerged tokens.
7. **VisionZip port.** VisionZip was re-implemented from the official Qwen2.5-VL code rather
   than run as-is (the reference is a fork of a transformers version that does not run here).
   The selection logic, budget split and merging rule follow the reference; positional
   handling of compressed sequences is our own (original M-RoPE positions retained).
8. **AWQ kernel path.** AWQ runs through AutoAWQ's Triton kernels (triton-windows), not the
   CUDA extension kernels (no Windows wheel for this torch). The dispatch finding is specific
   to this kernel pair; other W4A16 back-ends (Marlin, ExLlama, vLLM) have different
   crossovers.
9. **Energy.** GPU-only, NVML-sampled power x time at 10 ms polling; NVML power is itself
   driver-averaged. CPU/system energy is not included. The NVML energy counter was
   rejected as implausible on this device.
10. **Memory.** Peak *allocated* memory is per query; the device footprint (isolated pass) adds
   the CUDA context *and other processes on the GPU* (~1.1 GB here, including a browser),
   so footprints are machine-state dependent.
11. **fp16 vision tower in the reported sweeps.** 3.2% of TextVQA images overflowed in the last
   ViT block and failed in every configuration; corrected numbers on overflow-free samples are
   given in the erratum (Section 4). A full re-run with the bf16 vision tower was not done.
12. **Host environment.** Experiments ran in a pre-existing conda environment (at the user's
   request) with small, documented additions and two workarounds (AutoAWQ symbol alias,
   MKL threading layer); `requirements-lock.txt` records the exact package set.

## Future research extensions

- **FlashAttention-2 / SDPA-flash** on Linux with a CUDA 12 build to complete the attention axis.
- **Vision-side compression** (token pruning *inside* the ViT, lower `max_pixels`, or
  resolution-adaptive encoding) — our stage analysis shows the vision encoder becomes the
  dominant stage once LLM tokens are pruned.
- **Other W4A16 kernels** (Marlin, ExLlamaV2, GPTQ, bitsandbytes NF4) and INT8/FP8 paths, and
  a principled, shape-aware dispatch policy rather than a single threshold.
- **Long-form generation and multi-turn chat**, where decode and KV-cache size dominate.
- **More benchmarks** (MMBench, AI2D, DocVQA, ChartQA) — document/chart tasks are expected to be
  even more sensitive to token pruning than TextVQA.
- **Second architecture** (e.g. SmolVLM2-2.2B, InternVL, LLaVA-OneVision) to test whether the
  interaction patterns generalize.
- **Other runtimes as external references** (llama.cpp/Ollama GGUF, vLLM, TensorRT-LLM) —
  they cannot host the stage-level instrumentation or token pruning used here, but give
  end-to-end reference points.
- **Jetson Orin / other edge devices** and power-capped operation (energy-optimal configs).
- **Adaptive per-query configuration** (e.g. choose retention from image text density),
  using the optimizer's measured table as the policy space.


## 7. Conclusion

On an 8 GB laptop RTX 4060, composing established efficiency techniques for Qwen2.5-VL-3B
is **not** a matter of multiplying individually measured speedups. SDPA and VisionZip save
time on disjoint stages and add up; VisionZip and the AWQ kernel path interact strongly —
a fixed dispatch threshold in the quantization library made moderate token pruning
counter-productive, and re-tuning it both recovered and amplified the pruning benefit.
After LLM-token pruning, the vision encoder and fixed per-token decode become the
bottleneck. The measured Pareto front consists only of tuned-dispatch + VisionZip
configurations: T1 (75% tokens) keeps 99.6% of TextVQA accuracy at 28% lower latency and 27%
less energy than the baseline; T2 (50%) keeps POPE accuracy unchanged at 39% lower latency.
Practical guidance: profile per stage, re-derive kernel dispatch thresholds for the target
GPU, prune moderately (≤50%) for text-heavy tasks, and look to the vision encoder next.

We present EdgeCompose-VLM, a reproducible evaluation and deployment framework for studying
how low-bit quantization, visual-token compression and optimized attention interact in
resource-constrained VLM inference. Using an RTX 4060 8 GB GPU we characterize quality-
latency-memory trade-offs, identify stage-specific bottleneck shifts, quantify composition
effects (additive, interfering and synergistic), and derive Pareto-optimal deployment
configurations exposed through a measured-data deployment optimizer.
