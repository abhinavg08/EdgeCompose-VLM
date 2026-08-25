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
