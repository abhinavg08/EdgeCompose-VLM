# PROJECT_STATUS — EdgeCompose-VLM + EdgeInspect-VLM

_Last updated: 2026-10-01 01:25_

## Environment
- Hardware: NVIDIA RTX 4060 Laptop GPU (8188 MB, sm_89), 15.7 GB RAM, Windows 11 (WDDM); ~1.1 GB VRAM
  used by CUDA context + other apps.
- Python env: `python` in the research environment (user-designated conda env, Python 3.9.24).
- CUDA (torch build) 11.8 · PyTorch 2.7.1+cu118 · Transformers 4.57.1 · AutoAWQ 0.2.9 + triton-windows 3.3.1.
- Added to the env (all `--no-deps`): autoawq, triton-windows, zstandard, nvidia-ml-py, hf_xet
  (snapshot before: `results/env_before_edgecompose.txt`; exact env: `requirements-lock.txt`).
- git: portable MinGit on PATH.

## EdgeCompose status — COMPLETE (frozen)
- Model: Qwen2.5-VL-3B-Instruct-AWQ; quantization AWQ INT4 (LLM fp16), vision tower bf16 (default since erratum).
- Token compression: VisionZip port (official Qwen2.5-VL logic) + uniform control; attention SDPA / eager;
  AWQ dispatch upstream (1024) / tuned (64).
- Profiling: stage timing, TTFT, per-query + isolated VRAM, NVML energy, decode profile, repeats.
- Final: 14 configs x 500 TextVQA + 500 POPE (14,000 queries, 0 failures) → `results/results.csv`,
  `results/aggregate/tables_500.md`, `plots/fig1-10`, `report/edgecompose_report.md`.
- Erratum: fp16 ViT overflow on 16/500 TextVQA images; bf16 fix validated; corrected quality in
  `results/aggregate/textvqa_quality_clean484.csv`.

## EdgeInspect status
- Dataset: MVTec LOCO AD extracted (1,568 test images; verified counts) ✔
- Reference sampler + 70 manifests (nested in k, deterministic) ✔
- Classification (Stage A) + exact answer-likelihood score + normal-only calibration ✔
- Evaluation: AUROC (primary), AUPRC, calibrated operating point, per-type, bootstrap + paired CIs ✔
- Memory-scaling probe (k up to 24): max resident k = 4/8/12/16 at 100/75/50/25% ✔
- Full grid (seed 0): 5 cats x k{1,2,4,8} x r{100,75,50,25} x 50 rows = 4,000 rows, 0 failures ✔
  archived read-only: `results/edgeinspect/raw/archive_final_grid_seed0_2026-10-01/` (SHA256SUMS.txt)
- Seed variability (seed 1: k{1,8} x r{100,75,50}): RUNNING
- Eager-attention capacity check (k{4,8} x r{100,50}): queued after seeds
- Optimizer (`--task industrial_inspection`, `--min-auroc`, residency-aware): implemented ✔; example outputs pending
- Demo (`scripts/inspect_demo.py`): implemented ✔; real run pending (GPU busy)
- Qualitative examples (`scripts/edgeinspect_examples.py`): implemented ✔; run pending
- Report `report/edgeinspect_report.md`: drafted with final seed-0 numbers; seed/eager/demo/examples markers pending
- README: restructured (combined); pending markers

## PAUSED (2026-10-01 ~01:50, at user request) — how to resume
- Seed-1 variability run stopped mid-way: breakfast_box complete (300 rows), juice_bottle partial
  (260/300 rows); pushpins, screw_bag, splicing_connectors not started. Runs are resumable (done rows skipped).
- Resume (≈75 min GPU + ≈5 min eager check):
  `python scripts/run_edgeinspect_sweep.py --steps seeds memory_eager`
- Then: `python scripts/analyze_edgeinspect.py --tag final --main` (default n_boot 1000),
  `python scripts/edgeinspect_examples.py`,
  `python scripts/inspect_demo.py --category pushpins --references 4 --token-retention 0.75 --query <test image> --json-out results/edgeinspect/demo_output.json`,
  optimizer examples (`--task industrial_inspection ...`), fill remaining `⟨EI:...⟩` markers in
  `report/edgeinspect_report.md`, `README.md`, `report/interview_prep.md` (grep for ⟨EI), run tests
  (update test count in README/merl_application.md), final commit + tag `edgecompose-edgeinspect-v1`,
  check `git status` clean. Then stop engineering.

## Current blocker
None (paused by user).

## Last successful command
`python scripts/analyze_edgeinspect.py --tag final --main`

## Last experiment
EdgeInspect full grid, seed 0 (finished 01:09); seed-1 variability run started 01:09.

## Latest result files
`results/edgeinspect/aggregate/final_tables.md`, `final_results.csv`, `final_paired_contrasts.csv`,
`edgeinspect_results.csv` (optimizer input), `plots/edgeinspect/fig1-6`.

## Next action
After seed 1 + eager: re-run analysis (n_boot 1000), run demo + examples, optimizer examples, fill report/
README markers, run tests, final commit + tag `edgecompose-edgeinspect-v1`, verify clean `git status`.
Then STOP engineering.

## Key decisions (cumulative)
1. User conda env; no-deps additions only. 2. AutoAWQ `PytorchGELUTanh` alias. 3. SDPA (no FA2 on this
platform); eager as the unoptimized reference. 4. VisionZip port, per-image for multi-image prompts.
5. Own greedy loop (== generate(), repetition_penalty disabled). 6. Tuned AWQ dispatch 64 from microbenchmark.
7. bf16 vision tower (erratum). 8. Interleaved, rotated measurement. 9. Energy = sampled NVML power x time.
10. MKL_THREADING_LAYER=SEQUENTIAL. 11. EdgeInspect: answer-likelihood score, normal-only threshold (90th pct
of 10 validation/good), AUROC primary. 12. Per-query allocator release; residency classes resident/spill/OOM.
13. Seed-variability subset k{1,8} x r{100,75,50} (runtime-justified).

## Failed attempts / issues found (all documented)
- huggingface_hub downloads stalled → curl downloader. flash-attn: no wheel, no nvcc.
- NVML energy counter implausible → sampled power. MKL OpenMP layer crashed numpy BLAS → sequential layer.
- AutoAWQ 1024-row dispatch made pruning slower → tuned threshold.
- fp16 ViT overflow (`!!!!`) → bf16 vision tower.
- EdgeInspect greedy answer ~98% "ANOMALOUS" → likelihood scoring + normal-only calibration.
- Allocator cache grew to 10.7 GB (> VRAM) in the interleaved run → per-query release; affected rows archived
  in `results/edgeinspect/raw/archive_allocator_cache/` and excluded.
- `scripts/inspect.py` would shadow stdlib `inspect` → `scripts/inspect_demo.py`.
