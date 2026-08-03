"""Stage 1 smoke test: one image + one question -> answer, with VRAM/latency.

Also cross-checks the instrumented greedy loop against HF `generate()` and, if
requested, runs one VisionZip-compressed query to verify visual-token reduction.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --image path/to.jpg --question "What is written on the sign?"
    python scripts/smoke_test.py --retention 0.5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
from PIL import Image, ImageDraw, ImageFont

from edgecompose.compression import build_compressor
from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.profiling import memory
from edgecompose.utils import RESULTS_DIR, setup_logging, write_json

logger = logging.getLogger("smoke_test")


def synthetic_image() -> Image.Image:
    """Deterministic test image with legible text (no download needed)."""
    img = Image.new("RGB", (640, 480), (235, 235, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 120, 580, 360], fill=(200, 30, 30), outline=(0, 0, 0), width=6)
    try:
        font = ImageFont.truetype("arial.ttf", 96)
    except OSError:
        font = ImageFont.load_default()
    d.text((120, 190), "STOP 42", fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--image", default=None)
    ap.add_argument("--question", default="What text is written on the red sign?")
    ap.add_argument("--retention", type=float, default=0.5, help="also test VisionZip at this ratio (1.0 to skip)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    setup_logging()

    image = Image.open(args.image).convert("RGB") if args.image else synthetic_image()
    runner = QwenVLRunner(args.model_id, attention_backend=args.attn)
    report = {"load": runner.describe()}
    report["load"]["quantization_config"] = str(report["load"]["quantization_config"])

    # warmup (Triton JIT, cuBLAS handles, allocator)
    for _ in range(2):
        runner.run(image, args.question, max_new_tokens=args.max_new_tokens)

    memory.reset_peak()
    out = runner.run(image, args.question, max_new_tokens=args.max_new_tokens)
    mem = memory.snapshot()
    logger.info("answer: %r", out.text)
    logger.info("stages (ms): %s | TTFT %.1f | total %.1f | tokens %d",
                {k: round(v, 1) for k, v in out.stages_ms.items()}, out.ttft_ms, out.total_latency_ms,
                out.generated_tokens)
    logger.info("visual tokens: %d (grid %s) | peak alloc %.0f MB | peak reserved %.0f MB",
                out.num_visual_tokens_before, out.image_grid_thw, mem["peak_allocated_mb"], mem["peak_reserved_mb"])

    ref = runner.reference_generate(image, args.question, max_new_tokens=args.max_new_tokens)
    match = ref == out.text
    logger.info("HF generate(): %r | matches instrumented loop: %s", ref, match)
    report["baseline"] = {**out.__dict__, **mem, "hf_generate_text": ref, "matches_hf_generate": match}

    if args.retention < 1.0:
        comp = build_compressor("visionzip", args.retention)
        runner.run(image, args.question, compressor=comp, max_new_tokens=args.max_new_tokens)
        memory.reset_peak()
        cout = runner.run(image, args.question, compressor=comp, max_new_tokens=args.max_new_tokens)
        cmem = memory.snapshot()
        logger.info("VisionZip r=%.2f: %r | tokens %d -> %d (dominant %d + contextual %d) | TTFT %.1f ms",
                    args.retention, cout.text, cout.num_visual_tokens_before, cout.num_visual_tokens_after,
                    cout.num_dominant, cout.num_contextual, cout.ttft_ms)
        report["visionzip"] = {**cout.__dict__, **cmem, "retention": args.retention}

    out_path = RESULTS_DIR / "smoke_test.json"
    write_json(report, out_path)
    logger.info("wrote %s", out_path)
    if not match:
        logger.warning("instrumented loop differs from HF generate(); inspect before benchmarking")


if __name__ == "__main__":
    main()
