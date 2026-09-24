# Interview preparation — EdgeCompose-VLM / EdgeInspect-VLM

Every number below is measured in this repository (RTX 4060 Laptop 8 GB, batch 1).
EdgeCompose: 14 configs x 500 TextVQA + 500 POPE (14,000 queries). EdgeInspect: MVTec LOCO AD,
5 categories x k ∈ {1,2,4,8} x retention ∈ {100,75,50,25}% x 40 test (+10 calibration) queries.

## 30-second explanation
"I studied what happens when you stack standard efficiency tricks for a 3-billion-parameter
vision-language model on an 8 GB laptop GPU: 4-bit AWQ weights, visual-token pruning with
VisionZip, and fused attention. Profiling every stage showed the gains don't simply multiply —
token pruning collided with the quantization library's kernel selection and made the model
slower until I re-tuned that dispatch threshold; then it was 28% faster at 99.6% of the
accuracy. I then used the same stack for few-shot industrial inspection on MVTec LOCO, where
compression is what lets more known-good reference images fit in 8 GB: from 4 references at
full resolution to 16 at 25% of the visual tokens. With 8 references at 75% of the tokens, mean AUROC rose from 0.605 to 0.697 compared with the best uncompressed setting that fits — a real gain, although absolute detection quality stays modest."

## 2-minute explanation
"The question was whether VLM inference optimizations that each look good in isolation still
help when composed on constrained hardware. I used Qwen2.5-VL-3B with the official AWQ INT4
weights on an RTX 4060 laptop GPU and wrote my own generation loop around the Hugging Face
modules so I could time preprocessing, the vision encoder, token compression, prefill and
decode separately — verified token-for-token against `generate()`.

I swept VisionZip token retention against SDPA-vs-eager attention, interleaving all
configurations per sample so thermal drift couldn't bias one of them, on 1,000 TextVQA/POPE
questions. Two things stood out. First, SDPA and VisionZip act on different stages — SDPA speeds
up the vision encoder by about 40%, VisionZip shortens prefill — so their savings add. Second,
keeping 75% of tokens made TextVQA *slower*: AutoAWQ switches GEMM kernels at 1024 tokens, and
pruning pushed prompts onto a Triton kernel that my microbenchmark showed is 2.6x slower at that
size. Moving the threshold to the measured crossover gave 21.6% speedup at identical outputs,
and with VisionZip-75% a 27.8% latency and 27% energy reduction at 99.6% accuracy. After
pruning, the vision encoder is the bottleneck.

I also found and fixed an fp16 overflow in the vision tower that silently broke 3.2% of TextVQA
answers — running the unquantized vision tower in bf16 fixed all of them.

For the application, I built few-shot anomaly inspection on MVTec LOCO: k known-good references
plus a query, scored by the exact likelihoods of 'NORMAL' vs 'ANOMALOUS'. On 8 GB, full-resolution
prompts stop being VRAM-resident beyond 4 references; with 75/50/25% of tokens the limit rises to
8/12/16. More references help — going from 1 to 8 raises mean AUROC by 0.08-0.13 at every compression level — and compression costs nothing measurable at a fixed reference count up to 4. So 8 references at 75% tokens, which stays in VRAM, beats the best uncompressed setting that fits (4 references) by 0.09 AUROC: 0.697 vs 0.605. Absolute quality is modest: pushpins stays near chance, and the model's own answers are almost always 'anomalous', which is why I score likelihoods and calibrate the threshold on normal images only."

## 5-minute technical walkthrough
1. **Setup (M).** RTX 4060 Laptop, 8 GB, Windows, PyTorch 2.7 (CUDA 11.8), Transformers 4.57, AutoAWQ
   0.2.9 with Triton kernels. Qwen2.5-VL-3B-Instruct-AWQ: 3.25 GB resident. SDPA on this build is
   the memory-efficient kernel (no FlashAttention compiled in; flash-attn not installable).
2. **Instrumentation.** Own greedy loop over HF sub-modules; stages = preprocess, vision, fusion +
   compression, prefill, decode; CUDA-synchronized boundaries; per-query peak memory; NVML power.
   Verified identical to `generate()` (after disabling the checkpoint's repetition_penalty 1.05).
3. **Compression.** VisionZip ported from the official Qwen2.5-VL code: last ViT block attention
   picks dominant tokens, 5% contextual tokens merged by key similarity; survivors keep their
   M-RoPE positions. Uniform subsampling as a control.
4. **Protocol.** 14 configs interleaved per sample in rotated order; 500 TextVQA + 500 POPE; paired
   bootstrap CIs; interaction I = R_AB − R_A·R_B; noise-aware Pareto; repeat sessions.
5. **Findings.** (a) Quality: 99.7/97.0/83.3% of TextVQA accuracy at 75/50/25% tokens; POPE flat.
   (b) Composition: SDPA (vision) and VisionZip (prefill) add; VisionZip and AWQ dispatch interact
   (synergy on TextVQA, interference on POPE). (c) Kernel dispatch: fused Triton GEMM wins only for
   ≤32 rows; dequant+cuBLAS 2.6x faster at ~1000 rows; threshold 64 → T1 = −27.8% latency, 99.6%
   quality, −27% energy. (d) Bottleneck: vision share 37% → 53%; decode ~64 ms/token regardless.
   (e) Erratum: fp16 ViT overflow on 3.2% of images; bf16 vision tower fixes it.
6. **EdgeInspect.** k references (train/good, nested) + query; per-image VisionZip; anomaly score =
   log P(ANOMALOUS) − log P(NORMAL) by teacher forcing; threshold from 10 normal validation images
   (90th percentile); AUROC as the main metric. Memory: max VRAM-resident k = 4/8/12/16 at
   100/75/50/25% tokens; beyond that Windows pages to system RAM (k=12 at 100%: 104 s/query).
   ⟨EI:5min-results⟩
7. **Limits.** One GPU/OS, one model, 500/200 queries, 1-2 reference seeds, prompt dependence, no
   localization, no factory data.

## Questions and answers

**1. Why Qwen2.5-VL-3B?** It is a current, strong small VLM with an official AWQ checkpoint (no
own calibration needed), native dynamic resolution and multi-image prompts, first-class
Transformers support, and it fits an 8 GB GPU with room for activations (3.25 GB resident).
Measured TextVQA accuracy 0.800 (overflow-free samples) and POPE accuracy 0.848.

**2. Why AWQ?** Weight-only 4-bit quantization with activation-aware scaling preserves quality
well at 3B scale and keeps activations in 16-bit; the official pre-quantized checkpoint makes the
baseline reproducible. It cuts the resident model to 3.25 GB (the vision tower stays 16-bit),
which is what makes an 8 GB card viable at all. It does not reduce activation/KV memory or
prefill cost, which is why token compression still matters.

**3. Why VisionZip?** Training-free, representative of post-encoder visual-token pruning, and
it has an official Qwen2.5-VL implementation to port faithfully. It keeps the most-attended tokens
plus merged contextual tokens. Against uniform subsampling it wins on TextVQA (+2.5 pp at 50%,
+8.1 pp at 25%, paired, overflow-free samples); on POPE the difference is not significant.

**4. Why SDPA?** It ships with PyTorch (no compilation), and on this Windows build it dispatches
to the memory-efficient fused kernel. Compared with eager attention it cuts vision-encoder time by
~40% and peak memory by 1.7 GB on TextVQA, because eager materializes 4096x4096 attention matrices
in the ViT's full-attention blocks. FlashAttention-2 had no wheel for this platform.

**5. Why did pruning initially slow the model?** At 100% tokens about half of TextVQA prompts
have ≥1024 tokens and run AutoAWQ's dequantize + cuBLAS path. Pruning to 75% (~740 tokens) moved
every prompt onto AutoAWQ's fused Triton split-K GEMM, which is 2.6x slower at that size on this
GPU: prefill +27%, TTFT +17%, end-to-end +1% instead of a gain.

**6. What caused the 1024-token behavior?** A hard-coded heuristic in AutoAWQ's W4A16 linear:
if batch x sequence ≥ 1024 → dequantize to FP16 and use cuBLAS, else use the fused kernel. Figure 9
shows the cliff in per-query prefill time exactly at 1024 prompt tokens.

**7. Why did you introduce tuned dispatch?** Because the stage timing pointed to it and a kernel
microbenchmark over the LLM's exact layer shapes showed the fused kernel only wins up to 32 rows
(decode). Setting the threshold to 64 keeps the fused kernel for decode and uses cuBLAS for every
prefill; it changes kernel selection, not the model, and was chosen from the microbenchmark, not
from benchmark accuracy. Result: −21.6% TextVQA latency at identical outputs (T0), −27.8% with
VisionZip-75%.

**8. Why bf16 only for the vision tower?** The LLM must stay FP16 because the AWQ kernels compute
in FP16. The vision tower is not quantized, its native checkpoint dtype is bf16, bf16 has FP32's
exponent range, and Ada GPUs run it at full speed — so it removes overflow with no measured cost
(vision 395 → 385 ms, memory ±3 MB).

**9. Why did FP16 overflow?** For some images the residual stream of the last ViT block exceeds
FP16's maximum (65,504) → Inf → NaN after the merger → NaN logits, and argmax falls to token 0,
which decodes as "!". Forward hooks on every block located the first non-finite value in block 31.
16 of 500 TextVQA images failed in every configuration; with bf16: 0.

**10. Why AUROC for anomaly detection?** It measures ranking of anomalous above normal images
independent of any threshold or class balance, and it is the standard image-level metric on MVTec.
With 50% anomalous queries, the trivial "always anomalous" rule already gets F1 = 0.667, so F1 or
F1-max alone can look good while the model discriminates nothing.

**11. Why not trust the generated NORMAL/ANOMALOUS strings?** Because they were degenerate: on the
first 40-query check the greedy answer was ANOMALOUS for 100% of queries, normal ones included.
The exact likelihoods of the two answers still ranked queries (AUROC 0.71 in that check), so the
score is the log-likelihood ratio, not the text, and not a model-stated confidence.

**12. How was the threshold calibrated?** For each (category, k, retention): 10 normal images from
the validation split — never used as references or queries — are scored, and the threshold is the
90th percentile of their scores (≈10% false-positive target). No anomalous image is used. It is a
deployment operating point, not an F1-optimal threshold; with 10 images it is coarse.

**13. Why MVTec LOCO?** It is the industrial benchmark with both structural and logical anomalies,
original splits, and only normal training images — the realistic few-shot, normal-reference
setting. 1,568 test images over five categories (breakfast box, juice bottle, pushpins, screw bag,
splicing connectors).

**14. Structural vs logical anomaly?** Structural = local physical defects (scratch, contamination,
deformation). Logical = every part looks fine but a global constraint is violated (missing or
extra component, wrong count or arrangement). In our measurements the two types behaved alike: 0.697 vs 0.697 mean AUROC at k = 8 / 75%, and both gained similarly from more references and lost similarly under strong compression. With 50 anomalies of each type the CIs are about ±0.10, so small differences are not resolvable.

**15. Why multiple reference images?** Normal products vary (positions, lighting, fill levels);
several references show the model which variation is acceptable and what the correct composition
is. Measured: going from 1 to 8 references raises mean AUROC by +0.08 to +0.13 at every retention level (paired CIs exclude 0). The gain is not monotonic (k = 4 is not better than k = 2) and not universal: breakfast box, splicing connectors and juice bottle improve, screw bag and pushpins do not.

**16. Why does token compression help the industrial application?** A k-shot prompt carries
≈493·(k+1) visual tokens; the LLM prefill and its memory scale with them. On 8 GB the largest
VRAM-resident k is 4 at full resolution but 8/12/16 at 75/50/25% retention. And the extra references pay off: 8 references at 75% (VRAM-resident, 7.6 GB, 5.4 s/query) reach 0.697 mean AUROC versus 0.605 for the best uncompressed configuration that fits (k = 4) — a paired difference of +0.092 [+0.026, +0.170] — while compression at a fixed k ≤ 4 has no detectable quality cost.

**17. Why is system-memory spill important?** On Windows the NVIDIA driver pages to system RAM
instead of raising CUDA OOM, so an over-budget configuration "works" but becomes dramatically
slower (k=12 references at 100% tokens: 104 s/query vs 5.6 s VRAM-resident at 50%, ≈19x). Feasibility must be judged by footprint and
latency, not by the absence of an exception; we also found the caching allocator itself could
grow the process past VRAM (10.7 GB reserved) until we released it per query.

**18. What is the actual contribution?** Not a new compression or quantization method. It is a
reproducible, instrumented study of how established techniques compose on real constrained
hardware: stage-wise evidence that attention and token pruning add while token pruning and
quantized-kernel dispatch interact; the dispatch fix; an fp16 failure mode and its fix; a
measured-data deployment optimizer; and an application showing that these systems choices
translate into more usable few-shot context for industrial inspection on 8 GB.

**19. What would you do with a 24 GB GPU?** Measure how far AUROC keeps improving with k at full
resolution (beyond the 8-GB limit of k=4), compare against an unquantized 16-bit model and a 7B
model, run FlashAttention-2 on Linux, and re-derive the dispatch crossover for that GPU —
the point of the study is that these optima are hardware-specific.

**20. What would you research next at MERL?** (a) Vision-side compression — pruning or adaptive
resolution inside the ViT, now the dominant stage and the memory limit at high k; (b) automatic,
shape-aware kernel dispatch for quantized models; (c) better-calibrated VLM anomaly scores and
reference selection for inspection; (d) deployment on embedded GPUs with power caps.
