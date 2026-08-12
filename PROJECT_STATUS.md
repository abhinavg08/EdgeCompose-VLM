# PROJECT_STATUS — EdgeCompose-VLM

_Last updated: 2026-09-30 (session 1)_

## Environment (fixed)
- Python env: **`python` in the research environment** (user-designated conda env; Python 3.9.24)
- torch 2.7.1+cu118, transformers 4.57.1, accelerate 1.10.1
- Added to env (all `--no-deps`, torch untouched): `autoawq==0.2.9`, `triton-windows==3.3.1.post21`,
  `zstandard`, `nvidia-ml-py`, `hf_xet`. Pre-change snapshot: `results/env_before_edgecompose.txt`.
- GPU: RTX 4060 Laptop, 8188 MB, sm_89, driver 560.94. RAM 15.7 GB.

## Completed stages
- [x] **Stage 0** — `python scripts/system_check.py` → `results/system_info.json`. CUDA OK.
- [x] **Stage 1** — `python scripts/smoke_test.py` → `results/smoke_test.json`.
  Qwen2.5-VL-3B-Instruct-AWQ loads fully on GPU (3251 MB allocated), answers correctly,
  peak 3.33 GB alloc / 3.48 GB reserved. Instrumented greedy loop == HF `generate()` (pure greedy).
  VisionZip 50% verified: 391 → 196 visual tokens (177 dominant + 19 contextual).
- [ ] Stage 2 — manifests (TextVQA val, POPE test) — parquets downloaded to `hf_assets/dataset/`
- [ ] Stage 3 — baseline C0 on 20 → 200 samples
- [ ] Stage 4 — VisionZip retention sweep 100/75/50/25 (MVP)
- [ ] Stage 5 — composition (SDPA vs eager; FA2 not installable → see decisions)
- [ ] Stage 6 — full sweep on larger fixed subsets
- [ ] Stage 7 — analysis/plots; Stage 8 — optimize.py; Stage 9 — optional extensions

## Exact commands
```
$py = "python"
& $py scripts/system_check.py
& $py scripts/download_assets.py --repo Qwen/Qwen2.5-VL-3B-Instruct-AWQ
& $py scripts/download_assets.py --repo lmms-lab/textvqa --repo-type dataset --include "data/validation-*"
& $py scripts/download_assets.py --repo lmms-lab/POPE --repo-type dataset --include "data/test-*"
& $py scripts/smoke_test.py
& $py scripts/prepare_data.py --n 200 1000
& $py scripts/benchmark.py --configs baseline.yaml --datasets textvqa pope --n 200 --limit 20
& $py scripts/run_sweep.py --sweep sweep_main.yaml --n 200
```

## Next action
Build manifests (`prepare_data.py --n 200 1000`), verify schema/IDs, run C0 on 20 samples.

## Major decisions
1. **Env**: user's conda env (user instruction), not a fresh venv. Only no-deps additions.
2. **AWQ on transformers 4.57**: AutoAWQ 0.2.9 imports removed `PytorchGELUTanh`; aliased to
   `GELUTanh` in `edgecompose/compat.py` (no site-packages edits). Triton-windows gives real
   AWQ kernels (without it AutoAWQ falls back to naive dequant+matmul).
   AutoAWQ kernel heuristic: `batch*seq >= 1024` → dequantize + fp16 cuBLAS matmul; else Triton split-K GEMM.
   This switches with prefill length → relevant to token-compression results.
3. **Attention**: Windows cu118 torch has **no SDPA flash kernel**; SDPA = mem-efficient kernel.
   flash-attn has no wheel for py3.9/torch2.7/cu118/Windows → FA2 arm skipped (fallback rule).
   To still study "optimized attention", the matrix uses **eager (unfused) vs SDPA** as the
   attention axis; switched in place via `set_attn_implementation` (verified on both sub-configs).
4. **VisionZip**: official repo has Qwen2.5-VL support (2025-05-26) but as a fork of the
   transformers-4.49 modeling file. Ported the selection logic only (fallback step 2,
   "isolate the compressor") into `edgecompose/compression/visionzip.py`, faithful to the
   reference (last-block softmax attention, head-mean/query-sum, 2x2 group mean, 5% contextual
   tokens via key-similarity merge, kept slots keep original M-RoPE positions).
5. **Own generation loop** (not `generate()`) for exact stage timing; verified identical outputs.
6. **Decoding**: pure greedy. Checkpoint generation_config has `repetition_penalty=1.05`
   (applied by HF even with do_sample=False) — disabled for all configs.
7. **Pixel budget**: min 256·28², max 1024·28² → ≤1024 visual tokens/image.
8. **Downloads**: `huggingface_hub` downloads stalled at 0 bytes (HTTP and xet); plain curl works
   → `scripts/download_assets.py` (curl, resumable) into `hf_assets/`.
9. **Interleaved measurement**: all configs sharing a model run per sample in rotated order.

## Failed approaches
- `pip install --dry-run autoawq` hung resolving a torch pin → installed with `--no-deps`.
- `snapshot_download` (HTTP, then hf_xet) stalled at 0 bytes → curl-based downloader.
- SDPA FlashAttention backend: "Torch was not compiled with flash attention" (Windows build).

## Open items / blockers
- Decode latency varied 740 → 1570 ms (14 tokens) between runs after passing the all-ones
  attention mask HF uses. Not yet attributed (mask-construction overhead vs laptop GPU clock
  state). Interleaved per-sample measurement protects config comparisons either way;
  revisit with repeats. Ollama (user has it) is not usable for the core study (GGUF engine:
  no token pruning, no backend switch, no stage timing) — possible external reference later.
