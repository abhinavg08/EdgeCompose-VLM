# MERL application package — EdgeCompose-VLM / EdgeInspect-VLM

All numbers are measured in this repository (RTX 4060 Laptop GPU, 8 GB; Qwen2.5-VL-3B-Instruct-AWQ;
batch 1). Sources: `results/aggregate/tables_500.md`, `results/aggregate/textvqa_quality_clean484.csv`,
`results/edgeinspect/aggregate/final_tables.md`.

## A. CV project entry

**EdgeCompose-VLM / EdgeInspect-VLM — Efficient Multimodal Foundation Models on Constrained Hardware**

• Built an instrumented INT4 (AWQ) inference and profiling stack for Qwen2.5-VL-3B on an 8 GB RTX 4060,
  composing visual-token compression, SDPA attention and quantized-kernel dispatch with stage-wise
  latency, VRAM and energy measurement across 14,000 benchmarked VQA queries.

• Traced a counter-intuitive slowdown (25% visual-token pruning made time-to-first-token 17% worse) to the
  quantization library's fixed GEMM dispatch threshold; re-tuning it from a kernel microbenchmark cut
  end-to-end latency by 27.8% at 99.6% of baseline TextVQA accuracy and reduced GPU energy per query by 27%.

• Applied the stack to few-shot industrial anomaly inspection on MVTec LOCO AD: token compression raised
  the number of reference images that stay in 8 GB VRAM from 4 to 8-16, and 8 references at 75% token
  retention improved mean AUROC from 0.605 to 0.697 over the best uncompressed configuration that fits.

## B. 120-word application summary

I study how to make multimodal foundation models usable under hard hardware limits. In EdgeCompose-VLM I
built an instrumented inference stack for a 4-bit Qwen2.5-VL-3B model on an 8 GB RTX 4060 and measured how
visual-token compression, attention kernels, quantized-kernel dispatch and numerical precision interact
across 14,000 profiled queries. The gains did not simply multiply: token pruning collided with the
quantization library's kernel selection until I re-tuned it, which gave a 28% latency reduction at 99.6%
of baseline accuracy. In EdgeInspect-VLM I applied the stack to few-shot industrial anomaly inspection
on MVTec LOCO: compression let two to four times more known-good reference images fit in memory, and
the larger reference set measurably improved anomaly discrimination.

*(120 words)*

## C. One-page research summary

See [one_page_summary.md](one_page_summary.md) (PDF-ready single page).

## D. GitHub presentation

**Repository tagline:** Hardware-aware study of efficient VLM inference on an 8 GB GPU — and what the
saved memory buys for few-shot industrial inspection.

**Two-sentence description:** EdgeCompose-VLM measures how 4-bit AWQ quantization, VisionZip token
compression, attention kernels and quantized-kernel dispatch interact for Qwen2.5-VL-3B on an RTX 4060
8 GB, with stage-wise profiling, bootstrap statistics, Pareto analysis and a measured-data deployment
optimizer. EdgeInspect-VLM applies the stack to few-shot anomaly inspection on MVTec LOCO AD and shows
that visual-token compression raises the number of VRAM-resident reference images from 4 to 8-16, which
improves anomaly discrimination.

**Topics:** `vision-language-models` `efficient-inference` `quantization` `awq` `token-pruning`
`visionzip` `qwen2-5-vl` `edge-ai` `gpu-profiling` `anomaly-detection` `mvtec-loco` `few-shot`
`industrial-inspection` `pytorch` `pareto-optimization`

**Suggested README hero section:**

> # EdgeCompose-VLM + EdgeInspect-VLM
> **Efficient multimodal foundation models on an 8 GB GPU — and what the savings buy.**
> 4-bit Qwen2.5-VL-3B on an RTX 4060 Laptop: token pruning collided with the quantization library's
> kernel dispatch (+17% TTFT) until re-tuned (−27.8% latency at 99.6% accuracy); for few-shot industrial
> inspection, compression lifts the VRAM-resident reference count from 4 to 8-16 and raises AUROC
> from 0.605 to 0.697.
> `14,000 profiled VQA queries · 5,500 measured inspection queries · 44 unit tests · reproducible from one env file`

## E. Interview preparation

See [interview_prep.md](interview_prep.md).

## Positioning notes
* Contribution = systematic, hardware-aware composition study + deployment framework + application
  evidence. Not a new quantization, compression or attention method.
* EdgeInspect is a research prototype (mean AUROC ≤ 0.72, one category near chance), not production-ready.
* Negative/engineering results are part of the story: kernel-dispatch slowdown, fp16 overflow,
  allocator-cache growth past VRAM, silent system-memory spill.
