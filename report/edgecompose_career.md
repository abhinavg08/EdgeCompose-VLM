# EdgeCompose-VLM — CV, application and interview material

All numbers below are measured in this repository (`results/aggregate/tables_500.md`,
500 TextVQA + 500 POPE questions, RTX 4060 Laptop 8 GB, batch 1).

## Concise project description (CV)

**EdgeCompose-VLM — Efficient Multimodal Foundation Model Inference.** A reproducible
systems study and deployment framework for composing INT4 AWQ quantization, VisionZip
visual-token compression and optimized attention for Qwen2.5-VL-3B on an 8 GB consumer GPU,
with stage-wise profiling, composition-interaction analysis, Pareto analysis and a
measured-data deployment optimizer.

## Two CV bullets (measured)

**EdgeCompose-VLM — Efficient Multimodal Foundation Model Inference**

• Built an instrumented INT4 (AWQ) Qwen2.5-VL-3B inference and profiling stack on an RTX 4060
  8 GB GPU, composing VisionZip visual-token compression, SDPA attention and AWQ kernel-dispatch
  settings across 14 configurations and 14,000 CUDA-synchronized TextVQA/POPE queries with
  stage-wise (vision/prefill/decode) latency, VRAM and energy profiling.

• Found that AutoAWQ's fixed kernel-dispatch threshold made 75% token retention 17% *slower* in
  time-to-first-token; re-tuning it to the microbenchmarked crossover and composing it with
  VisionZip reduced end-to-end latency by 27.8% (TextVQA) and 38.7% (POPE) while retaining
  99.6% and 100.2% of baseline benchmark quality, and cut GPU energy per query by 27%.

## 100-150 word summary (internship applications)

EdgeCompose-VLM asks whether independently effective VLM inference optimizations still help
when stacked on constrained hardware. I built an instrumented runner for the 4-bit AWQ
Qwen2.5-VL-3B model on an 8 GB RTX 4060 laptop GPU that times vision encoding, prefill and
decode separately, then evaluated 14 combinations of VisionZip token compression, SDPA vs
eager attention and AWQ kernel dispatch on 1,000 TextVQA/POPE questions (14,000 queries).
The composition turned out to be stage-dependent: attention and token pruning act on
different stages and add up, but token pruning interacted with the quantization library's
kernel selection — a hard-coded threshold made moderate pruning slower. Re-tuning it gave a
28% latency reduction at 99.6% of baseline accuracy. The project includes Pareto analysis,
an interaction metric with bootstrap CIs, and a constraint-based deployment optimizer.

## Technical explanation for an interview (~2 minutes)

"I wanted to know whether VLM efficiency techniques compose on a real 8 GB GPU. I used
Qwen2.5-VL-3B with official AWQ INT4 weights and wrote my own greedy loop around the Hugging
Face modules so I could time preprocessing, the vision encoder, token compression, prefill
and decode separately — I verified it reproduces `generate()` token for token. I ported
VisionZip's Qwen2.5-VL selection logic, and swept 100/75/50/25% visual tokens against SDPA
versus eager attention, running all configurations back-to-back per sample so GPU clock
drift couldn't bias one of them.

The first surprise was that keeping 75% of tokens made TextVQA *slower*. Stage timing showed
prefill got slower, and plotting prefill time against prompt length showed a cliff exactly at
1024 tokens: AutoAWQ uses a fused Triton GEMM below 1024 rows and dequantize-plus-cuBLAS
above. A microbenchmark showed the fused kernel is 2.6x slower at ~1000 rows on this GPU and
only wins below ~64 rows. Moving the threshold to 64 made the baseline 22% faster at identical
outputs, and made token pruning pay off again: tuned dispatch plus 75% tokens is 28% faster
at 99.6% accuracy.

To quantify composition I used I = R_AB − R_A·R_B with paired bootstrap CIs. SDPA and
VisionZip act on different stages, so their savings add; dispatch and pruning both act on
prefill and interfere on POPE but are synergistic on TextVQA, because one changes the other's
kernel regime. After pruning, the vision encoder is the bottleneck — the next lever is
vision-side reduction, not more LLM-token pruning."

(30-second version: "I stacked 4-bit quantization, visual-token pruning and fused attention
for a 3B VLM on an 8 GB laptop GPU and profiled every stage. The gains weren't multiplicative:
attention and pruning add up because they hit different stages, but pruning collided with the
quantization library's kernel selection and made things slower. Fixing that dispatch threshold
gave 28% lower latency at 99.6% accuracy, and showed the vision encoder is the next bottleneck.")

## Limitations (short list)

1. Single GPU (RTX 4060 Laptop, Windows/WDDM, CUDA 11.8 PyTorch build); absolute latencies drift
   12-32% between sessions.
2. FlashAttention-2 could not be evaluated; SDPA used its memory-efficient kernel.
3. 500 questions per dataset; quality CIs ±3-4 pp; short-answer VQA only (2-4 decode tokens).
4. One model family; VisionZip re-implemented from the official Qwen2.5-VL code.
5. AWQ via AutoAWQ's Triton kernels; the dispatch finding is specific to that kernel pair.
6. GPU-only energy from sampled NVML power.

## Future research extensions

1. FlashAttention-2 / SDPA-flash on Linux + CUDA 12 builds.
2. Vision-side compression (in-ViT pruning, adaptive resolution) — now the dominant stage.
3. Other W4A16 kernels (Marlin, ExLlamaV2) and a shape-aware dispatch policy.
4. Long-form generation, where decode and KV cache dominate.
5. Document/chart benchmarks (DocVQA, ChartQA) and a second VLM architecture.
6. Jetson/edge devices, power-capped operation and per-query adaptive configuration.
