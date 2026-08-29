"""EdgeInspect experiment runner (few-shot grid over reference count k x token retention r).

Protocol (mirrors EdgeCompose's benchmark engine):
* one model load; attention backend and AWQ dispatch fixed for the whole run;
* references come from saved manifests (nested in k, identical for every r);
* warmup on validation/good images (never used as reference or query);
* for every query, all (k, r) configurations run back-to-back in rotated order;
* every query is one JSONL row; CUDA OOM is recorded as status="OOM" (never dropped).
  After `oom_skip_after` OOMs of the same (k, r) within a category, remaining queries of
  that configuration are logged as OOM without re-running them (status="OOM",
  error_message="skipped: repeated OOM") to avoid wasting hours on an infeasible config;
* resumable: (seed, k, r, sample_id) rows already present are skipped;
* calibration: `calibration` validation/good images (disjoint from warmup, references and
  queries) are scored for every configuration (rows with role="calibration"); the analysis
  sets a normal-only decision threshold from them (no anomalous data is used).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

from edgecompose.compression import build_compressor
from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.profiling.power import PowerMonitor
from edgecompose.quant.awq_dispatch import set_dequant_threshold
from edgecompose.utils import RESULTS_DIR, append_jsonl, read_jsonl
from edgeinspect.datasets.mvtec_loco import LocoImage, load_category
from edgeinspect.inference.classify import classify
from edgeinspect.prompts.inspection import classification_content, render_text
from edgeinspect.references.sampler import load_queries, load_references

logger = logging.getLogger(__name__)
RAW_DIR = RESULTS_DIR / "edgeinspect" / "raw"


@dataclass
class InspectConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    attention_backend: str = "sdpa"
    awq_dequant_threshold: int = 64
    token_method: str = "visionzip"
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    compress_query: bool = True
    max_new_tokens: int = 6
    warmup: int = 2
    calibration: int = 10  # validation/good images per config for the normal-only decision threshold
    oom_skip_after: int = 3
    extra: Dict = field(default_factory=dict)


def load_image(img: LocoImage) -> Image.Image:
    return Image.open(img.abspath()).convert("RGB")


def build_runner(cfg: InspectConfig) -> QwenVLRunner:
    runner = QwenVLRunner(cfg.model_id, attention_backend=cfg.attention_backend,
                          min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
    set_dequant_threshold(cfg.awq_dequant_threshold)
    return runner


def _key(row: Dict) -> Tuple:
    return (row["reference_seed"], row["reference_count"], float(row["token_retention"]), row["sample_id"])


def run_category(
    runner: QwenVLRunner,
    cfg: InspectConfig,
    category: str,
    ks: Sequence[int],
    retentions: Sequence[float],
    seed: int,
    queries_name: str,
    tag: str,
    limit: int = 0,
    measure_power: bool = True,
) -> Path:
    queries: List[LocoImage] = load_queries(category, queries_name)
    if limit:
        # keep label balance when limiting: interleave labels
        by = {}
        for q in queries:
            by.setdefault(q.label, []).append(q)
        mixed = [x for tup in zip(*by.values()) for x in tup]
        queries = mixed[:limit]
    refs_all = load_references(category, max(ks), seed)
    ref_imgs_all = [load_image(r) for r in refs_all]
    out = RAW_DIR / f"{tag}__{category}__seed{seed}.jsonl"
    done = {_key(r) for r in read_jsonl(out)} if out.exists() else set()
    configs = [(k, r) for k in ks for r in retentions]
    power = PowerMonitor() if measure_power else None
    if power is not None and not power.available:
        power = None
    run_id = f"{_dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    oom_count: Dict[Tuple[int, float], int] = {c: 0 for c in configs}
    for r_ in read_jsonl(out) if out.exists() else []:
        if r_["status"] == "OOM":
            oom_count[(r_["reference_count"], float(r_["token_retention"]))] = oom_count.get(
                (r_["reference_count"], float(r_["token_retention"])), 0) + 1

    # warmup on validation normals (outside the evaluation set)
    val = load_category(category)["validation"][: cfg.warmup]
    t0 = time.perf_counter()
    for v in val:
        vi = load_image(v)
        for k, r in configs:
            if oom_count[(k, r)] >= cfg.oom_skip_after:
                continue
            res = classify(runner, category, ref_imgs_all[:k], vi, build_compressor(cfg.token_method, r),
                           cfg.compress_query, cfg.max_new_tokens)
            if res["status"] == "OOM":
                logger.warning("warmup OOM at k=%d r=%.2f", k, r)
    logger.info("%s seed%d warmup %.1fs", category, seed, time.perf_counter() - t0)

    calib = load_category(category)["validation"][cfg.warmup: cfg.warmup + cfg.calibration]
    items = [(q, "test") for q in queries] + [(c, "calibration") for c in calib]
    t_start = time.perf_counter()
    for i, (q, role) in enumerate(items):
        qimg = None
        rot = configs[i % len(configs):] + configs[: i % len(configs)]
        for k, r in rot:
            if (seed, k, float(r), q.sample_id) in done:
                continue
            row = {
                "run_id": run_id, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"), "tag": tag,
                "role": role,
                "category": category, "sample_id": q.sample_id, "query_image": q.path, "label": q.label,
                "binary_label": q.binary, "anomaly_type": q.anomaly_type, "reference_count": k,
                "reference_images": [x.sample_id for x in refs_all[:k]], "reference_seed": seed,
                "token_method": cfg.token_method if r < 1.0 else "none", "token_retention": float(r),
                "compress_query": cfg.compress_query, "attention_backend": cfg.attention_backend,
                "awq_dequant_threshold": cfg.awq_dequant_threshold, "max_pixels": cfg.max_pixels,
                "model_name": cfg.model_id,
            }
            if oom_count[(k, r)] >= cfg.oom_skip_after:
                row.update(status="OOM", error_message="skipped: repeated OOM for this (k, r) in this category")
                append_jsonl(row, out)
                continue
            if qimg is None:
                qimg = load_image(q)
            if power is not None:
                power.start()
            res = classify(runner, category, ref_imgs_all[:k], qimg, build_compressor(cfg.token_method, r),
                           cfg.compress_query, cfg.max_new_tokens)
            pw = power.stop() if power is not None else {}
            if res["status"] == "OOM":
                oom_count[(k, r)] += 1
                logger.warning("%s %s k=%d r=%.2f OOM (%d)", category, q.sample_id, k, r, oom_count[(k, r)])
            row.update(res)
            row["average_power_w"] = pw.get("average_power_w")
            row["energy_j"] = pw.get("energy_j") if res["status"] == "ok" else None
            if res["status"] == "ok":
                row["correct"] = int(res["prediction"] == q.binary)
            append_jsonl(row, out)
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            el = time.perf_counter() - t_start
            logger.info("  %s seed%d: %d/%d queries incl. calibration (%.0fs, eta %.0fs)", category, seed, i + 1,
                        len(items), el, el / (i + 1) * (len(items) - i - 1))
    if power is not None:
        power.close()
    return out


def save_run_config(cfg: InspectConfig, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = render_text(classification_content("pushpins", 2))
    path.write_text(json.dumps({**asdict(cfg), **kw, "prompt_example_k2": example}, indent=2))
