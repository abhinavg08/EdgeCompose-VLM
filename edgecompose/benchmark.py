"""Benchmark engine: evaluate configurations on a fixed manifest and log per-query rows.

Protocol (identical for every configuration):

1. Load the model once per (model, pixel budget) group.
2. Warm up every configuration of the group on `warmup` samples that are *not* part of
   the evaluation manifest (Triton JIT, cuBLAS/cuDNN handles, allocator growth).
3. For each evaluation sample, run every configuration of the group in a rotated order
   (config order shifts by one per sample). Interleaving means slow drifts (GPU clocks,
   temperature, background load) affect all configurations equally rather than biasing
   whichever configuration happens to run last.
4. Before every measured query: CUDA synchronize + reset peak memory stats; stages are
   timed with synchronized wall clocks (see profiling/latency.py); NVML energy is read
   around the query.
5. Each query becomes one JSONL row (see RESULT_FIELDS). Failures are logged as rows with
   status="error" and never silently dropped. Existing OK rows are skipped on resume.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from edgecompose.compression import build_compressor
from edgecompose.datasets.base import Sample
from edgecompose.evaluation.metrics import parse_yes_no, score_sample
from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.profiling import memory
from edgecompose.profiling.power import PowerMonitor
from edgecompose.utils import ExperimentConfig, append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

RESULT_FIELDS = [
    "run_id", "timestamp", "gpu_name", "torch_version", "cuda_version", "transformers_version",
    "model_name", "quantization", "config_name", "token_method", "token_retention", "attention_backend",
    "dataset", "category", "sample_id", "question", "prediction", "parsed_prediction", "ground_truth", "score",
    "num_visual_tokens_before", "num_visual_tokens_after", "num_dominant", "num_contextual", "prefill_seq_len",
    "image_grid_thw", "preprocess_ms", "vision_ms", "compress_ms", "prefill_ms", "decode_ms", "ttft_ms",
    "total_latency_ms", "generated_tokens", "tokens_per_second", "decode_tokens_per_second",
    "peak_allocated_mb", "peak_reserved_mb", "average_power_w", "energy_j", "energy_counter_j",
    "energy_sampled_j", "ignore_eos", "repeat", "status", "error_message",
]


@dataclass
class EnvInfo:
    gpu_name: str
    torch_version: str
    cuda_version: str
    transformers_version: str

    @classmethod
    def collect(cls) -> "EnvInfo":
        import transformers

        return cls(
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            torch_version=torch.__version__,
            cuda_version=str(torch.version.cuda),
            transformers_version=transformers.__version__,
        )


def raw_path(results_dir: Path, cfg: ExperimentConfig, dataset: str, n: int, tag: str = "") -> Path:
    suffix = f"__{tag}" if tag else ""
    return results_dir / "raw" / f"{cfg.name}__{dataset}_{n}{suffix}.jsonl"


def completed_ids(path: Path, repeat: int = 0) -> set:
    if not path.exists():
        return set()
    return {r["sample_id"] for r in read_jsonl(path) if r.get("status") == "ok" and r.get("repeat", 0) == repeat}


def _compressor_for(cfg: ExperimentConfig):
    kwargs = {"contextual_ratio": cfg.contextual_ratio} if cfg.token_method == "visionzip" else {}
    return build_compressor(cfg.token_method, cfg.token_retention, **kwargs)


def run_one(
    runner: QwenVLRunner,
    cfg: ExperimentConfig,
    sample: Sample,
    image,
    env: EnvInfo,
    power: Optional[PowerMonitor],
    run_id: str,
    ignore_eos: bool = False,
    repeat: int = 0,
) -> Dict:
    """Execute and score one (config, sample) query and return a result row."""
    runner.set_attention_backend(cfg.attention_backend)
    compressor = _compressor_for(cfg)
    row: Dict = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "gpu_name": env.gpu_name,
        "torch_version": env.torch_version,
        "cuda_version": env.cuda_version,
        "transformers_version": env.transformers_version,
        "model_name": cfg.model_id,
        "quantization": cfg.quantization,
        "config_name": cfg.name,
        "token_method": cfg.token_method,
        "token_retention": cfg.token_retention,
        "attention_backend": cfg.attention_backend,
        "dataset": sample.dataset,
        "category": sample.category,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "ground_truth": sample.answers if sample.dataset == "textvqa" else sample.answers[0],
        "ignore_eos": ignore_eos,
        "repeat": repeat,
    }
    try:
        memory.reset_peak()
        if power is not None:
            power.start()
        out = runner.run(image, sample.prompt, compressor=compressor, max_new_tokens=cfg.max_new_tokens,
                         ignore_eos=ignore_eos)
        pw = power.stop() if power is not None else {}
        mem = memory.snapshot()
        st = out.stages_ms
        decode_s = st.get("decode", 0.0) / 1e3
        row.update(
            prediction=out.text,
            parsed_prediction=parse_yes_no(out.text) if sample.dataset == "pope" else out.text,
            score=score_sample(sample.dataset, out.text, sample.answers),
            num_visual_tokens_before=out.num_visual_tokens_before,
            num_visual_tokens_after=out.num_visual_tokens_after,
            num_dominant=out.num_dominant,
            num_contextual=out.num_contextual,
            prefill_seq_len=out.prefill_seq_len,
            image_grid_thw=out.image_grid_thw,
            preprocess_ms=st.get("preprocess", 0.0),
            vision_ms=st.get("vision", 0.0),
            compress_ms=st.get("compress", 0.0),
            prefill_ms=st.get("prefill", 0.0),
            decode_ms=st.get("decode", 0.0),
            ttft_ms=out.ttft_ms,
            total_latency_ms=out.total_latency_ms,
            generated_tokens=out.generated_tokens,
            tokens_per_second=out.generated_tokens / (out.total_latency_ms / 1e3),
            decode_tokens_per_second=(out.decode_steps / decode_s) if decode_s > 0 and out.decode_steps else None,
            peak_allocated_mb=mem.get("peak_allocated_mb"),
            peak_reserved_mb=mem.get("peak_reserved_mb"),
            average_power_w=pw.get("average_power_w"),
            energy_j=pw.get("energy_j"),
            energy_counter_j=pw.get("energy_counter_j"),
            energy_sampled_j=pw.get("energy_sampled_j"),
            status="ok",
            error_message="",
        )
    except Exception as e:  # keep going; record the failure
        if power is not None:
            try:
                power.stop()
            except Exception:
                pass
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        logger.error("%s | %s failed: %s", cfg.name, sample.sample_id, e)
        row.update(status="oom" if is_oom else "error", error_message=f"{type(e).__name__}: {e}"[:500],
                   score=None, prediction=None)
        logger.debug(traceback.format_exc())
        torch.cuda.empty_cache()
    for k in RESULT_FIELDS:
        row.setdefault(k, None)
    return row


def run_group(
    runner: QwenVLRunner,
    configs: Sequence[ExperimentConfig],
    samples: List[Sample],
    warmup_samples: List[Sample],
    results_dir: Path,
    n_label: int,
    ignore_eos: bool = False,
    tag: str = "",
    repeats: int = 1,
    measure_power: bool = True,
) -> Dict[str, Path]:
    """Evaluate all `configs` (sharing one loaded model) on `samples`, interleaved per sample."""
    env = EnvInfo.collect()
    run_id = f"{_dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    power = PowerMonitor() if measure_power else None
    if power is not None and not power.available:
        power = None
    dataset = samples[0].dataset
    paths = {c.name: raw_path(results_dir, c, dataset, n_label, tag) for c in configs}

    # warmup: every config, samples outside the evaluation set
    t0 = time.perf_counter()
    for ws in warmup_samples:
        img = ws.load_image()
        for c in configs:
            runner.set_attention_backend(c.attention_backend)
            runner.run(img, ws.prompt, compressor=_compressor_for(c), max_new_tokens=c.max_new_tokens,
                       ignore_eos=ignore_eos)
    torch.cuda.synchronize()
    logger.info("warmup: %d samples x %d configs in %.1fs", len(warmup_samples), len(configs), time.perf_counter() - t0)

    for rep in range(repeats):
        done = {c.name: completed_ids(paths[c.name], rep) for c in configs}
        todo = [s for s in samples if any(s.sample_id not in done[c.name] for c in configs)]
        logger.info("repeat %d: %d/%d samples to run for %d configs", rep, len(todo), len(samples), len(configs))
        t_start = time.perf_counter()
        for i, s in enumerate(todo):
            img = s.load_image()
            k = i % len(configs)
            order = list(configs[k:]) + list(configs[:k])
            for c in order:
                if s.sample_id in done[c.name]:
                    continue
                row = run_one(runner, c, s, img, env, power, run_id, ignore_eos=ignore_eos, repeat=rep)
                append_jsonl(row, paths[c.name])
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                el = time.perf_counter() - t_start
                logger.info("  %s rep%d: %d/%d samples (%.1fs, eta %.0fs)", dataset, rep, i + 1, len(todo), el,
                            el / (i + 1) * (len(todo) - i - 1))
    if power is not None:
        power.close()
    return paths


def load_rows(paths: Sequence[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for p in paths:
        if Path(p).exists():
            rows.extend(read_jsonl(Path(p)))
    return rows


def dumps_config(cfg: ExperimentConfig) -> str:
    return json.dumps(cfg.to_dict(), sort_keys=True)
