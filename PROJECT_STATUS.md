# PROJECT_STATUS — EdgeCompose-VLM

_Last updated: 2026-09-30 16:35 (session 1)_

## Environment (fixed)
- Python env: **`python` in the research environment** (user-designated conda env; Python 3.9.24)
- torch 2.7.1+cu118, transformers 4.57.1, accelerate 1.10.1
- Added to env (all `--no-deps`, torch untouched): `autoawq==0.2.9`, `triton-windows==3.3.1.post21`,
  `zstandard`, `nvidia-ml-py`, `hf_xet`; matplotlib 3.9.4 force-reinstalled (same version).
  Pre-change snapshot: `results/env_before_edgecompose.txt`; exact env: `requirements-lock.txt`.
- git: portable MinGit at `git.exe` (not on PATH).
- GPU: RTX 4060 Laptop, 8188 MB, sm_89, driver 560.94. RAM 15.7 GB.

## Completed stages
- [x] **Stage 0** — `scripts/system_check.py` → `results/system_info.json`.
- [x] **Stage 1** — `scripts/smoke_test.py` → `results/smoke_test.json`. Loop == HF generate() (pure greedy).
- [x] **Stage 2** — manifests `data/manifests/{textvqa,pope}_{200,1000}.jsonl` (seed 1234, nested, digests in manifest_info.json).
- [x] **Stage 3** — C0 baseline 20 → 200 samples.
- [x] **Stage 4 (MVP)** — VisionZip 100/75/50/25 under SDPA, 200 samples/dataset.
- [x] **Stage 5 (dev)** — eager vs SDPA x retention + uniform controls: `sweep_main.yaml`, n=200 (4000 queries, 0 failures).
      FA2: no wheel for py3.9/torch2.7/cu118/Windows, no CUDA toolkit → documented, skipped.
- [x] AWQ kernel microbenchmark (`scripts/awq_kernel_bench.py`) → crossover M=64; added tuned-dispatch arm T0-T3.
- [x] Isolated memory pass (`scripts/memory_profile.py --sweep sweep_final.yaml`) → `results/aggregate/memory_isolated.csv`.
- [x] Decode profile, 32 forced tokens, 40 TextVQA samples (`--tag decode32`).
- [ ] Repeat pass (3x, 40 samples, `--tag rep`) — running
- [ ] **Stage 6** — final sweep: `run_sweep.py --sweep sweep_final.yaml --n 1000 --limit 500` (14 configs, ~3.5 h)
- [ ] Stage 7 analysis on final data (`analyze.py --n 500 --main`), Stage 8 optimizer demo, docs, CV text

## Exact commands
```
$py = "python"
& $py scripts/run_sweep.py --sweep sweep_main.yaml --n 200                     # dev (done)
& $py scripts/awq_kernel_bench.py                                              # done
& $py scripts/memory_profile.py --sweep sweep_final.yaml --n 200 --k 10        # done
& $py scripts/benchmark.py --configs baseline.yaml token_25.yaml eager_baseline.yaml eager_token_25.yaml tuned_baseline.yaml tuned_token_25.yaml --datasets textvqa --n 200 --limit 40 --ignore-eos --tag decode32 --no-power   # done
& $py scripts/benchmark.py --configs baseline.yaml token_50.yaml token_25.yaml eager_baseline.yaml tuned_baseline.yaml tuned_token_50.yaml --datasets textvqa pope --n 200 --limit 40 --repeats 3 --tag rep --no-power
& $py scripts/run_sweep.py --sweep sweep_final.yaml --n 1000 --limit 500      # final
& $py scripts/analyze.py --n 200 ; & $py scripts/analyze.py --n 500 --main
```

## Key measured findings so far (dev, n=200 unless noted)
- Stage split (TextVQA, ~1000 visual tokens, SDPA): vision ~430-460 ms, prefill ~430 ms, decode 2-4 tokens.
- AutoAWQ dispatch: M<1024 → Triton split-K GEMM (slow), M≥1024 → dequant+cuBLAS. Whole-LLM linear time
  at M=1023: 646 ms fused vs 245 ms dequant (2.6x). Consequence: VisionZip 75% (≈780 tokens) prefill is
  SLOWER than 100% (≈1040 tokens). Tuned threshold 64 removes the inversion (T-arm).
- Decode ≈ 63 ms/token independent of KV length (278 vs 993) and attention backend → no decode benefit
  from token compression at these lengths.
- Eager attention: +~330 ms vision encoder and +1.9 GB peak alloc on TextVQA (5.39 vs 3.53 GB);
  token compression changes peak VRAM by < 75 MB (peak is in the vision encoder).

## Major decisions
1. **Env**: user's conda env (user instruction), only no-deps additions.
2. **AWQ on transformers 4.57**: alias `PytorchGELUTanh` (edgecompose/compat.py). triton-windows for kernels.
3. **Attention**: no SDPA-flash on Windows cu118 (SDPA = mem-efficient kernel); FA2 impossible here →
   attention axis = eager vs SDPA, switched in place via `set_attn_implementation`.
4. **VisionZip**: faithful port of the official Qwen2.5-VL selection logic (fallback step 2).
5. **Own generation loop**; pure greedy (checkpoint repetition_penalty=1.05 disabled for all configs).
6. **Pixel budget**: 256–1024 visual tokens/image.
7. **Downloads** via curl (`scripts/download_assets.py`) — huggingface_hub stalled.
8. **Interleaved measurement** of all configs per sample (rotated order).
9. **Energy** = NVML sampled power × time (counter over-reports on this laptop GPU).
10. **VRAM**: per-query peak *allocated* for tables; isolated pass for device footprint.
11. **MKL_THREADING_LAYER=SEQUENTIAL** — env's MKL Intel-OpenMP layer crashes numpy BLAS (0xc06d007f).
12. **Tuned AWQ dispatch (threshold 64)** added as a measured, configuration-level knob (not a new kernel).
13. Final sweep at 500 samples/dataset (prefix of the 1000 manifests) to bound runtime (~3.5 h).

## Failed approaches
- `pip install --dry-run autoawq` hung resolving a torch pin → `--no-deps`.
- `snapshot_download` (HTTP, hf_xet) stalled at 0 bytes → curl downloader.
- SDPA FlashAttention backend: "Torch was not compiled with flash attention".
- flash-attn: no binary wheel; no nvcc to build.
- NVML energy counter: implausible (>TGP) for sub-second windows.
- matplotlib crash traced to MKL OpenMP layer, not matplotlib (reinstall did not help).

## EdgeInspect-VLM (continuation; starts after EdgeCompose final analysis)
User request (2026-09-30 17:16): after EdgeCompose completes end to end, build EdgeInspect-VLM
(few-shot industrial inspection on MVTec LOCO AD) inside this repo, build order A0-A10.
- Dataset: MVTec LOCO AD (CC BY-NC-SA 4.0) downloading to `hf_assets/mvtec_loco/` from the
  official MVTec mydrive link; extraction deferred until the GPU sweep ends (CPU-heavy).
- Code written (CPU-only, while the GPU sweep runs):
  `edgeinspect/{datasets/mvtec_loco.py, references/sampler.py, prompts/inspection.py,
  inference/classify.py, inference/explain.py, metrics/anomaly.py, analysis/fewshot.py, runner.py}`,
  scripts `prepare_loco.py, inspect_smoke_test.py, run_edgeinspect.py, edgeinspect_memory.py,
  inspect_demo.py` (NOT inspect.py: that name shadows the stdlib `inspect` module), optimizer
  `--task industrial_inspection`. Tests: 43 passing.
- Runner generalized: `QwenVLRunner.run_images()` (multi-image, per-image VisionZip with
  per-image attention statistics, optional query exemption, exact class-likelihood scoring on a
  cropped KV cache, timed separately). `run()` is now a wrapper → re-verify EdgeCompose smoke test.
- Design: references from train/good only, nested in k per seed; queries stratified from test;
  SDPA + tuned AWQ dispatch (threshold 64); per-image budget default 512 visual tokens
  (to be confirmed by A3 timing probe).
- Next: A0 (re-run smoke_test.py), A1 prepare_loco.py, A3 inspect_smoke_test.py --probe, A4, A5.

## Open items
- Ollama (user has it) not usable for the core study (GGUF engine: no token pruning/backend switch/stage timing);
  listed as possible external reference in future work.
