# PROJECT_STATUS — EdgeCompose-VLM + EdgeInspect-VLM

_Final: 2026-10-01 — ENGINEERING FROZEN (tag `edgecompose-edgeinspect-v1`)_

## Environment
- Hardware: NVIDIA RTX 4060 Laptop GPU (8188 MB, sm_89), 15.7 GB RAM, Windows 11 (WDDM); ~1.1 GB VRAM
  used by CUDA context + other apps.
- Python env: `python` in the research environment (Python 3.9.24).
- CUDA (torch build) 11.8 · PyTorch 2.7.1+cu118 · Transformers 4.57.1 · AutoAWQ 0.2.9 + triton-windows 3.3.1.
- Added to env (no-deps): autoawq, triton-windows, zstandard, nvidia-ml-py, hf_xet. Exact env: `requirements-lock.txt`.
- git: portable MinGit on PATH.

## EdgeCompose status — COMPLETE
- Model Qwen2.5-VL-3B-Instruct-AWQ (AWQ INT4 LLM fp16; vision tower bf16 by default after the erratum).
- VisionZip port + uniform control; SDPA / eager; AWQ dispatch upstream (1024) / tuned (64).
- Final sweep 14 configs x 500 TextVQA + 500 POPE = 14,000 queries, 0 failures.
- Outputs: `results/results.csv`, `results/aggregate/tables_500.md`, `plots/fig1-10`, `report/edgecompose_report.md`.
- Erratum: fp16 ViT overflow (16/500 TextVQA) → bf16; corrected quality `results/aggregate/textvqa_quality_clean484.csv`.

## EdgeInspect status — COMPLETE
- Dataset MVTec LOCO AD (1,568 test images); 70 deterministic reference/query manifests.
- Full grid seed 0: 5 cats x k{1,2,4,8} x r{100,75,50,25} x 50 = 4,000 rows, 0 failures; archived read-only
  with SHA-256: `results/edgeinspect/raw/archive_final_grid_seed0_2026-10-01/`.
- Seed variability seed 1: k{1,8} x r{100,75,50} x 5 cats = 1,500 rows, 0 failures.
- Memory scaling (SDPA, k ≤ 24) + eager check (k{4,8} x r{100,50}).
- Analysis: `results/edgeinspect/aggregate/final_tables.md` (Tables A-G), `final_seed1_contrasts.csv`,
  `edgeinspect_results.csv` (optimizer input), `plots/edgeinspect/fig1-6`, `examples.png`.
- Optimizer examples: `results/edgeinspect/optimizer_examples.txt`; demo: `results/edgeinspect/demo_output.txt`.
- Report: `report/edgeinspect_report.md` (~5,700 words).

## Application package — COMPLETE
`README.md` (combined), `report/merl_application.md` (CV entry, 120-word summary, GitHub text),
`report/one_page_summary.md`, `report/interview_prep.md`, `report/edgecompose_career.md`.

## Current blocker
None. Engineering frozen; no further experiments planned.

## Last successful commands
`python scripts/analyze_edgeinspect.py --tag final --main`; `python scripts/edgeinspect_examples.py`;
`python scripts/inspect_demo.py ...`; `python -m pytest tests -q` (44 passed).

## Key decisions (cumulative)
1. User conda env; no-deps additions. 2. AutoAWQ `PytorchGELUTanh` alias. 3. SDPA (no FA2 on this platform);
eager as the unoptimized reference. 4. VisionZip port, per-image for multi-image prompts. 5. Own greedy loop
(== generate(), repetition_penalty disabled). 6. Tuned AWQ dispatch 64 from microbenchmark. 7. bf16 vision
tower. 8. Interleaved, rotated measurement. 9. Energy = sampled NVML power x time. 10. MKL_THREADING_LAYER=
SEQUENTIAL. 11. EdgeInspect: answer-likelihood score; normal-only threshold (90th pct of 10 validation/good);
AUROC primary. 12. Per-query allocator release; residency classes resident / spill / OOM. 13. Seed-variability
subset k{1,8} x r{100,75,50}; eager check limited to k{4,8} x r{100,50}.

## Failed attempts / issues found (documented in the reports)
- huggingface_hub downloads stalled → curl downloader. flash-attn: no wheel, no nvcc.
- NVML energy counter implausible → sampled power. MKL OpenMP layer crashed numpy BLAS → sequential layer.
- AutoAWQ 1024-row dispatch made pruning slower → tuned threshold.
- fp16 ViT overflow → bf16 vision tower.
- Greedy answer ~98% "ANOMALOUS" → likelihood scoring + normal-only calibration.
- Allocator cache grew to 10.7 GB (> VRAM) → per-query release; affected rows archived in
  `results/edgeinspect/raw/archive_allocator_cache/` and excluded.
- Long background jobs were stopped by the harness time limit → remaining seed-1 work run per category (resumable).
- `scripts/inspect.py` would shadow stdlib `inspect` → `scripts/inspect_demo.py`.
