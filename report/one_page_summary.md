# EdgeCompose-VLM / EdgeInspect-VLM — one-page research summary

**Efficient multimodal foundation models on constrained hardware** · Qwen2.5-VL-3B-AWQ on an
RTX 4060 Laptop GPU (8 GB) · all numbers measured in this repository

### Problem
Small VLMs fit on consumer GPUs once quantized, and many inference optimizations report
speedups in isolation. Deployments stack them. Do their gains compose on real 8 GB hardware —
and does saved compute/memory translate into better application capability?

### Method
* Instrumented inference runner (preprocess / vision / compression / prefill / decode, CUDA-
  synchronized, per-query VRAM and energy), verified token-for-token against HF `generate()`.
* 14 configurations — VisionZip visual-token retention 100/75/50/25% x SDPA vs eager attention x
  AutoAWQ kernel dispatch (upstream vs tuned) — on 500 TextVQA + 500 POPE questions (14,000
  interleaved queries); paired bootstrap CIs, interaction metric I = R_AB − R_A·R_B, Pareto analysis.
* EdgeInspect: few-shot inspection on MVTec LOCO AD with k ∈ {1,2,4,8} known-good references x
  4 retention levels x 5 categories; exact answer-likelihood anomaly score; normal-only calibrated
  threshold; isolated memory-scaling probe up to k = 24.

### Main systems finding
* Optimizations **do not simply multiply**. SDPA (vision encoder, −40%) and VisionZip (prefill) act
  on different stages and add; VisionZip interacts with the quantized-kernel path.
* AutoAWQ's fixed 1024-token dispatch made 75%-token pruning **17% slower in TTFT**; the fused kernel
  is 2.6x slower than dequantize + cuBLAS at ~1000 rows on this GPU. Re-tuning the threshold to the
  measured crossover (64 rows) + VisionZip-75%: **−27.8% latency, 99.6% of TextVQA accuracy, −27%
  energy**; POPE (VisionZip-50%): −38.7% latency at 100.2% accuracy.
* After pruning, the vision encoder is the bottleneck (37% → 53% of latency); decode ~64 ms/token.
* Engineering failure modes surfaced and fixed: fp16 overflow in the last ViT block (3.2% of TextVQA
  images → garbage) fixed by a bf16 vision tower; allocator-cache growth past VRAM and silent
  Windows system-memory spill (no OOM; k = 12 references at 100% tokens: 104 s/query vs 5.6 s
  VRAM-resident at 50%, ≈19x).

### Application result
* Max VRAM-resident reference count on 8 GB: **4 at 100% tokens → 8 / 12 / 16 at 75 / 50 / 25%**.
* More references improve discrimination: 1 → 8 references raise mean AUROC by **+0.08 to +0.13** at every retention level (paired 95% CIs exclude 0); compression at a fixed k ≤ 4 has no detectable cost; structural and logical anomalies behave alike.
* **8 references at 75% tokens** (VRAM-resident, 7.6 GB, 5.4 s/query): mean AUROC **0.697** [0.624, 0.769] vs **0.605** for the best uncompressed configuration that fits (k = 4) — paired **+0.092 [+0.026, +0.170]**. Absolute quality is modest (pushpins ≈ chance; the model's own answer is ~98% "ANOMALOUS"), so this is a research prototype, not a deployable inspector.

### Why relevant to CI0213
*(align this paragraph with the posting's exact wording)* The project is about making multimodal
foundation models usable under hard resource limits: it measures where time and memory actually go
on edge-class hardware, shows that numerical precision, attention kernels, runtime kernel dispatch
and memory residency interact with algorithmic compression, and connects those systems effects to an
industrial inspection capability (how much few-shot visual context fits in 8 GB).

### Future research directions
1. Vision-side token reduction (in-ViT pruning / adaptive resolution) — the new bottleneck and memory limit.
2. Automatic, shape-aware kernel selection for quantized models; generalization across GPUs.
3. Calibrated VLM anomaly scores, reference selection, and anomaly localization for inspection.
4. Embedded/edge GPUs (Jetson-class) with power caps; multi-model comparison (7B, other VLM families).

**Thesis:** efficient foundation-model deployment is not determined by model size or FLOPs alone —
compression, precision, attention kernels, runtime dispatch and memory residency interact in
hardware-specific ways, and understanding them expands application capability under a fixed budget.
