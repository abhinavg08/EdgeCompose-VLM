# EdgeCompose-VLM + EdgeInspect-VLM

**A hardware-aware study of efficient multimodal foundation-model inference on an RTX 4060 8 GB
GPU, extended to few-shot industrial anomaly inspection where visual-token compression increases
the usable reference context under a strict memory budget.**

Model: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (INT4) · GPU: RTX 4060 Laptop, 8 GB · batch 1 · no training.
This is a systems study and deployment framework, not a new compression or quantization algorithm:
AWQ, VisionZip, SDPA and FlashAttention are prior work (see [References](#references)).

## Key results (all measured in this repository)

| | Result |
|---|---|
| **Composition** | SDPA speeds up the *vision encoder* (−40%), VisionZip shortens *prefill*; their savings add. |
| **Kernel-dispatch interaction** | AutoAWQ's fixed 1024-token GEMM switch made 75%-token VisionZip **17% slower** in TTFT on TextVQA; re-tuning the threshold to the measured crossover + VisionZip 75% → **−27.8% latency at 99.6% of TextVQA accuracy, −27% GPU energy** (POPE, VisionZip 50%: −38.7% latency, 100.2% accuracy). |
| **Bottleneck shift** | After pruning, the vision encoder dominates (37% → 53% of latency); decode ≈64 ms/token everywhere. |
| **Numerical failure found & fixed** | fp16 overflow in the last ViT block broke 3.2% of TextVQA answers; bf16 vision tower → 16 failures to 0. |
| **Reference capacity on 8 GB** | Max VRAM-resident known-good references: **4 at 100% tokens → 8 / 12 / 16 at 75 / 50 / 25%**. |
| **Capacity → quality** | 1 → 8 references: **+0.08 to +0.13 AUROC** at every retention (paired CIs exclude 0); best VRAM-resident config (k = 8, 75%) **0.697 AUROC vs 0.605** for the best uncompressed config that fits (k = 4): **+0.093 [+0.026, +0.170]** (unrounded difference 0.0925). |
| **Honest limits** | Absolute inspection quality is modest (mean AUROC ≤ 0.72; pushpins ≈ chance); the VLM's own NORMAL/ANOMALOUS answer is ~98% "ANOMALOUS" and unusable without likelihood scoring + normal-only calibration. |

**Robustness:** a second, independent reference seed (k ∈ {1, 8} x 100/75/50%) reproduces the direction of every effect — k = 1 → 8 gains +0.04 to +0.08, 75% retention is free at k = 8 in both seeds (0.697 / 0.702) — but the reference gain at 75% is not significant for seed 1. Eager attention adds 0.2-0.5 GB and 61-94% latency without changing which configurations fit.

## Why it matters

Efficient deployment is usually argued from model size and FLOPs. On real constrained hardware
the outcome was decided by interactions: token pruning vs. the quantization library's kernel
choice, fp16 range vs. the vision tower's activations, caching-allocator growth vs. VRAM, and
silent system-memory spill vs. latency. Understanding those interactions turned a slower
configuration into a 28% faster one, and — for inspection — turned an 8 GB limit of 4 reference
images into 8-16, which measurably improves few-shot anomaly discrimination.

## Architecture

<p align="center">
  <img src="plots/architecture.svg" width="1200" alt="EdgeCompose and EdgeInspect share preprocessing, a bf16 vision encoder, per-image VisionZip compression, and AWQ INT4 prefill; VQA decoding and likelihood-based inspection feed measured analysis and deployment selection." />
</p>

Both tasks share the same instrumented runner. Compression acts on each image before
language-model prefill; inspection uses exact answer-likelihood scoring and a normal-only
threshold. The deployment selector chooses among measured configurations.

<details>
<summary>Editable Mermaid diagram</summary>

```mermaid
flowchart LR
    subgraph Inputs
        Q[query image]
        R["k known-good references<br/>(EdgeInspect)"]
        T[question / instruction]
    end
    subgraph Runner["Instrumented Qwen2.5-VL-3B-AWQ runner (edgecompose/models/qwen_vl.py)"]
        P[preprocess] --> V["vision encoder (bf16)<br/>SDPA or eager"]
        V --> C["VisionZip per image<br/>100/75/50/25% tokens"]
        C --> F["LLM prefill<br/>AWQ INT4, tuned dispatch"]
        F --> D[greedy decode + KV cache]
        F --> S["answer-likelihood score<br/>log P(ANOMALOUS) - log P(NORMAL)"]
    end
    Q --> P
    R --> P
    T --> P
    D --> L[("per-query logs: stages, TTFT, VRAM, energy")]
    S --> L
    L --> A["analysis: bootstrap CIs, interaction I = R_AB - R_A R_B,<br/>Pareto, residency (resident / spill / OOM)"]
    A --> O["optimize.py<br/>constraint-based selector over measured configs"]
```

</details>

## EdgeCompose findings (single-image VQA, 14 configs x 500 TextVQA + 500 POPE = 14,000 queries)

| TextVQA config | Accuracy* | TTFT | E2E | Peak alloc | Energy |
|---|---|---|---|---|---|
| C0 AWQ, SDPA, 100% tokens (baseline) | 0.800 | 957 ms | 1245 ms | 3507 MB | 92.4 J |
| C1 + VisionZip 75% (upstream AWQ dispatch) | 0.798 | **1119 ms (+17%)** | 1259 ms | 3446 MB | 96.2 J |
| T0 tuned dispatch, 100% | 0.800 | 843 ms | 976 ms (−21.6%) | 3507 MB | 73.5 J |
| **T1 tuned dispatch + VisionZip 75%** | **0.797** | 765 ms | **899 ms (−27.8%)** | 3446 MB | 67.6 J |
| T2 tuned dispatch + VisionZip 50% | 0.775 | 655 ms | 789 ms (−36.6%) | 3446 MB | 58.6 J |
| E0 eager attention, 100% | 0.800 | 1313 ms | 1468 ms (+17.9%) | 5216 MB (+1.7 GB) | 102.4 J |

\*Accuracy on the 484 samples without the (since fixed) fp16 overflow; see the erratum in the report.

* **Composition is stage-dependent:** SDPA × VisionZip ≈ additive (disjoint stages); tuned dispatch ×
  VisionZip interacts (synergy on TextVQA, interference on POPE) — individually measured gains do not
  predict composed gains in general.
* **Cause of the slowdown:** AutoAWQ runs a fused Triton W4A16 GEMM below 1024 rows and dequantize +
  cuBLAS above; on this GPU the fused kernel is 2.6x slower at ~1000 rows and only wins ≤ 32 rows
  (decode). Pruning moved prompts below the switch.
* **Quality:** VisionZip keeps 99.7 / 97.0 / 83.3% of TextVQA accuracy at 75 / 50 / 25% tokens; POPE is
  unchanged within noise; VisionZip beats uniform subsampling by +2.5 pp (50%) and +8.1 pp (25%).
* Pareto front (quality, latency, memory): tuned dispatch + VisionZip only. Details:
  [report/edgecompose_report.md](report/edgecompose_report.md).

## EdgeInspect use case (few-shot inspection on MVTec LOCO AD)

k normal references + 1 query → NORMAL/ANOMALOUS. 5 categories (breakfast box, juice bottle,
pushpins, screw bag, splicing connectors) with structural and logical anomalies; k ∈ {1,2,4,8} x
retention ∈ {100,75,50,25}%; 200 test queries per configuration (+ 10 normal validation images per
category for a normal-only decision threshold); AUROC as the primary metric.

| k | retention | mean AUROC [95% CI] | F1 @ normal-only threshold | E2E latency | VRAM footprint | residency |
|---|---|---|---|---|---|---|
| 1 | 100% | 0.594 [0.511, 0.670] | 0.357 | 1.35 s | 4.9 GB | resident |
| 4 | 100% | 0.605 [0.522, 0.686] | 0.323 | 3.33 s | 6.2 GB | resident (max k at 100%) |
| 8 | 100% | 0.723 [0.647, 0.793] | 0.586 | 8.25 s | 8.9 GB | **spills to system RAM** |
| **8** | **75%** | **0.697 [0.624, 0.769]** | 0.583 | 5.42 s | 7.6 GB | resident |
| 8 | 25% | 0.674 [0.601, 0.743] | 0.487 | 3.30 s | 5.7 GB | resident |

* More references help (k = 1 → 8: +0.08 to +0.13 AUROC at every retention); compression at a fixed
  k ≤ 4 has no detectable quality cost; 75% retention is free even at k = 8 (−0.026, n.s.).
* Structural and logical anomalies behave alike (e.g. 0.697 vs 0.697 at k = 8 / 75%).
* Categories differ: breakfast box up to 0.87, splicing connectors 0.77-0.85 at k = 8, pushpins ≈ chance.
* Details, CIs, per-category and structural/logical tables: [report/edgeinspect_report.md](report/edgeinspect_report.md).

## Hardware

NVIDIA GeForce RTX 4060 Laptop GPU (8188 MB, sm_89, 115 W), 15.7 GB RAM, Windows 11 (WDDM);
Python 3.9, PyTorch 2.7.1+cu118, Transformers 4.57.1, AutoAWQ 0.2.9 + triton-windows 3.3.1.
SDPA uses the memory-efficient kernel (this build has no FlashAttention kernel; flash-attn was not
installable). About 1.1 GB of VRAM is used by the CUDA context and other applications.
Exact environment: `requirements-lock.txt`, `results/system_info.json`.

## Reproduction

Run commands from the repository root in a Python 3.9 environment. The measured setup is
Windows 11 with the CUDA 11.8 PyTorch build; other platforms or dependency combinations are
not validated. `requirements.txt` lists direct dependencies; `requirements-lock.txt` is a
sanitized snapshot of the full research environment, including unrelated packages, rather
than a portable install recipe.

```powershell
# create and activate an isolated environment (Python 3.9 must already be installed)
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1

# environment (torch first, from the CUDA index; AutoAWQ without deps so torch is not replaced)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.1 accelerate==1.10.1 huggingface-hub==0.36.0 hf_xet numpy==2.0.2 datasets pyarrow pandas pillow pyyaml matplotlib nvidia-ml-py pytest
pip install --no-deps autoawq==0.2.9 zstandard triton-windows==3.3.1.post21
python scripts/system_check.py
python -m pytest tests -q          # 44 unit tests

# EdgeCompose (VQA)
python scripts/download_assets.py --repo Qwen/Qwen2.5-VL-3B-Instruct-AWQ
python scripts/prepare_data.py --n 200 1000
python scripts/smoke_test.py
python scripts/awq_kernel_bench.py
python scripts/run_sweep.py --sweep sweep_final.yaml --n 1000 --limit 500   # 14 configs, ~4.5 h; current bf16 vision default
python scripts/analyze.py --n 500 --main

# EdgeInspect: obtain mvtec_loco_anomaly_detection.tar.xz from MVTec (CC BY-NC-SA 4.0)
# and place it in hf_assets/mvtec_loco/ before the preparation command below.
python scripts/prepare_loco.py --final-per-label 20 10 10
python scripts/run_edgeinspect_sweep.py --steps memory grid seeds memory_eager    # ~6 h
python scripts/analyze_edgeinspect.py --tag final --main
python scripts/edgeinspect_examples.py

# deployment optimizer over measured results
python scripts/optimize.py --objective latency --min-quality 0.97 --relative-quality
python scripts/optimize.py --task industrial_inspection --max-vram-mb 7000 --max-latency-ms 6000 --min-auroc 0.65
```

Runs are resumable, every query is logged (`results/raw/`, `results/edgeinspect/raw/`), and the final
EdgeInspect grid is archived read-only with SHA-256 checksums. `PROJECT_STATUS.md` records every
decision, workaround and failed approach.

The project applies the AutoAWQ `PytorchGELUTanh` compatibility alias and sets
`MKL_THREADING_LAYER=SEQUENTIAL` before NumPy imports; no site-packages edits are required.
The language model uses fp16 activations, while the vision tower defaults to bf16 to avoid
the documented overflow. EdgeInspect releases the CUDA allocator cache per query to prevent
growth during long sweeps. FlashAttention is not required for the measured SDPA/eager setup.
Saved result files are included for analysis without downloading the model or datasets.
Existing result logs are resumed and completed rows skipped. To remeasure, use a separate
result tag or output directory and keep the published logs intact. The original EdgeCompose
sweep used an fp16 vision tower; `--vision-dtype float16` is available on `scripts/benchmark.py`
(not on the sweep wrapper) for reproducing that historical setting. The current bf16 default
addresses the erratum and is not a bit-for-bit re-run of the published fp16 sweep.

## Demo

```powershell
python scripts/inspect_demo.py --category breakfast_box --references 8 --token-retention 0.75 --query "hf_assets/mvtec_loco/mvtec_loco_anomaly_detection/breakfast_box/test/logical_anomalies/009.png"
```
Real saved run (`results/edgeinspect/demo_output.txt`; query chosen by a fixed rule, ground truth = logical anomaly):
```text
EdgeInspect-VLM
Model:            Qwen2.5-VL-3B-Instruct-AWQ (INT4 LLM, bf16 vision tower, SDPA, tuned AWQ dispatch)
Hardware:         NVIDIA GeForce RTX 4060 Laptop GPU (8188 MB)
Reference count:  8  (train/good: 271, 248, 267, 219, 145, 149, 056, 028; seed 0)
Token retention:  75%  (VisionZip per image; visual tokens 4500 -> 3375)
Prediction:       ANOMALOUS
Anomaly score:    +0.436  = log P(ANOMALOUS) - log P(NORMAL); normal-only calibrated threshold +0.319
Anomaly type:     logical
Issue:            extra nuts            (model-generated explanation, not verified)
Performance (classification pass, measured):
  TTFT:           5274 ms  (vision 2129 ms, prefill 2731 ms)
  Latency:        5752 ms
  Peak VRAM:      5475 MB allocated / 6348 MB reserved by the process
```
Inspection optimizer (real outputs: `results/edgeinspect/optimizer_examples.txt`):
```text
$ python scripts/optimize.py --task industrial_inspection --max-vram-mb 7000 --max-latency-ms 6000 --min-auroc 0.65
RECOMMENDED: reference count 8 | token retention 25% | sdpa | expected AUROC 0.674 [95% CI 0.601-0.743]
             median latency 3296 ms (TTFT 2900 ms) | peak VRAM 5742 MB | VRAM-resident
$ python scripts/optimize.py --task industrial_inspection --max-vram-mb 7000 --max-latency-ms 2000 --min-auroc 0.80
NO FEASIBLE CONFIGURATION (no measured configuration reaches AUROC 0.80; every rejection reason is listed)
```
Qualitative examples (real outputs, decision = normal-only threshold): `plots/edgeinspect/examples.png`.
This selected montage adapts MVTec LOCO AD images and is distributed under CC BY-NC-SA 4.0;
see [image attribution and license](plots/edgeinspect/examples.LICENSE.md).

## Main figures

| | |
|---|---|
| ![](plots/fig9_prefill_vs_length_500.png) **AWQ dispatch cliff** — prefill time drops at 1024 tokens | ![](plots/fig10_awq_kernel_crossover.png) **Kernel crossover** — fused GEMM wins only ≤ 32 rows |
| ![](plots/fig1_quality_vs_latency_500.png) **EdgeCompose quality vs latency** (Pareto ringed) | ![](plots/fig3_stage_latency_500.png) **Stage-wise latency** — vision encoder becomes the bottleneck |
| ![](plots/edgeinspect/fig3_reference_capacity.png) **Reference capacity** vs token retention (8 GB) | ![](plots/edgeinspect/fig1_auroc_vs_references.png) **Inspection AUROC vs number of references** |
| ![](plots/edgeinspect/fig2_auroc_vs_vram.png) **AUROC vs VRAM** — compression moves the feasible region | ![](plots/edgeinspect/fig6_auroc_vs_latency.png) **AUROC vs latency** (resident Pareto front) |

All figures: `plots/` (EdgeCompose, 10) and `plots/edgeinspect/` (EdgeInspect, 6 + examples).

## Limitations

* One GPU (Windows laptop) and one model family (Qwen2.5-VL-3B-AWQ); absolute latencies drift
  12-32% between sessions (within-session comparisons are interleaved and paired).
* FlashAttention-2 not evaluated; SDPA = memory-efficient kernel.
* Samples: 500 per VQA dataset; 200 inspection queries per configuration, one full + one partial
  reference seed; per-category inspection results are noisy.
* EdgeInspect uses a fixed, untuned prompt and ≤ 512 visual tokens per image; the VLM's answers are
  biased and its likelihood score is uncalibrated (threshold from 10 normal images); no localization;
  no factory data; MVTec LOCO AD is CC BY-NC-SA 4.0. The full dataset is not bundled;
  selected images appear only in the attributed qualitative montage.
* The reported EdgeCompose sweep ran the vision tower in fp16 (3.2% TextVQA overflow); corrected
  numbers are given on overflow-free samples, not from a full re-run.

## Reports

* [report/edgecompose_report.md](report/edgecompose_report.md) — EdgeCompose-VLM research report (methods, results, RQ1-RQ5, erratum).
* [report/edgeinspect_report.md](report/edgeinspect_report.md) — EdgeInspect-VLM research report (RQ1-RQ6, systems findings).
* [report/one_page_summary.md](report/one_page_summary.md) · [report/interview_prep.md](report/interview_prep.md) · [report/merl_application.md](report/merl_application.md)

## References

* Qwen2.5-VL (Bai et al., 2025); checkpoint `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
* AWQ (Lin et al., MLSys 2024); AutoAWQ (archived) with triton-windows kernels.
* VisionZip (Yang et al., CVPR 2025; github.com/dvlab-research/VisionZip, Qwen2.5-VL code).
* FlashAttention (Dao et al.); PyTorch `scaled_dot_product_attention`.
* TextVQA (Singh et al., CVPR 2019); POPE (Li et al., EMNLP 2023) — lmms-lab releases.
* MVTec LOCO AD (Bergmann et al., IJCV 2022), CC BY-NC-SA 4.0.

## Licensing and attribution

The repository has no overall project license. Adapted third-party code retains its upstream
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/`.
The downloaded 3B AWQ checkpoint uses the **Qwen Research License Agreement**, which must be
reviewed separately; model weights and the full datasets are excluded from Git.
