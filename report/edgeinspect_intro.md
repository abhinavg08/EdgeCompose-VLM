## 1. Introduction

Industrial visual inspection rarely has many labelled defects: defects are rare, varied and
expensive to collect, while images of *known-good* products are abundant. This makes
few-shot, reference-based inspection attractive: show a model a handful of normal products
and ask whether a new one deviates.

Modern vision-language models (VLMs) can compare images and reason about object counts,
arrangement and damage in natural language without task-specific training, which in
principle covers both *structural* defects (scratches, contamination, deformation) and
*logical* defects (a missing, extra or misplaced component, a wrong quantity). They are
also expensive: every image becomes hundreds of visual tokens, so a prompt with k reference
images and one query carries (k+1) times the visual-token load, increasing vision-encoder
time, prefill latency and memory. On an 8 GB consumer GPU this cost directly limits how
many references can be shown.

EdgeInspect-VLM tests whether the efficiency stack characterized in EdgeCompose-VLM
(INT4 AWQ weights, SDPA attention, tuned AWQ kernel dispatch, VisionZip visual-token
compression, stage-wise profiling) makes few-shot multimodal inspection practical on an
RTX 4060 Laptop GPU (8 GB), and how reference count and token compression trade off
against detection quality, latency and memory.

## 2. Related background

* **VLMs.** A vision encoder turns each image into visual tokens that a language model
  consumes together with text; Qwen2.5-VL supports multiple interleaved images per prompt.
* **Few-shot visual prompting.** Instead of fine-tuning, examples are supplied in-context;
  here the examples are only *normal* images (one-class / reference-based setting).
* **Industrial anomaly detection.** Classical approaches (PatchCore, EfficientAD, WinCLIP
  for zero/few-shot) score image patches against a memory of normal features. MVTec AD and
  VAND-style few-shot tracks evaluate image-level AUROC/F1-max at 1/2/4 shots.
* **Structural vs logical anomalies.** MVTec LOCO AD (Bergmann et al., IJCV 2022) adds
  *logical* anomalies — violations of composition constraints (e.g. a breakfast box with a
  missing tangerine, a pushpin compartment with two pins) — that are locally normal and
  require global reasoning, alongside conventional *structural* defects.
* **Visual-token compression.** Methods such as VisionZip prune/merge visual tokens after the
  vision encoder, shortening the LLM prompt; multi-image prompts multiply the benefit.
* **Resource-constrained inference.** At batch 1 on consumer GPUs, weight quantization
  bounds resident memory, while prompt length drives prefill time and activation memory.

## 3. System

```
references R1..Rk (train/good) ─┐
                                ├─> chat prompt: text + "Reference i (known-good):" <img> ... "Query image:" <img> + instruction
query Iq (test) ────────────────┘
      │
      ▼  EdgeCompose runner (Qwen2.5-VL-3B-Instruct-AWQ, SDPA, AWQ dispatch threshold 64)
 preprocess → ViT per image → VisionZip per image (retention r, same r for every image)
      → LLM prefill → greedy answer "NORMAL"/"ANOMALOUS" (Stage A, timed)
      → teacher-forced log P("NORMAL"), log P("ANOMALOUS") on the prompt KV cache (timed separately)
      → p_anomaly = sigmoid(logP(ANOMALOUS) − logP(NORMAL))
      → [if ANOMALOUS] Stage B: JSON {status, anomaly_type, issue, explanation} (not in benchmark latency)
```

* **Reuse.** EdgeInspect adds no new model code: `QwenVLRunner.run_images()` (the
  EdgeCompose runner generalized to interleaved multi-image prompts), EdgeCompose's
  VisionZip port, AWQ dispatch control, CUDA-synchronized stage timers, memory accounting and
  NVML power sampling.
* **Per-image compression.** VisionZip statistics are computed within each image (the ViT's
  full-attention blocks do not attend across images) and each image keeps the fraction r of
  its own tokens; surviving tokens keep their original M-RoPE positions.
* **Anomaly score.** Exact candidate likelihoods (all tokens of each answer) rather than a
  model-stated confidence; used for AUROC and F1-max. It is a ranking score, not calibrated.
