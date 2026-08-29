"""Stage A: NORMAL / ANOMALOUS classification with a likelihood-based anomaly score.

Anomaly score. After the prompt is prefilled, the exact log-probabilities of the two
candidate answers are computed by teacher forcing on the prompt's KV cache (all tokens of
each candidate, not just the first):

    s_normal  = log P("NORMAL"    | references, query, prompt)
    s_anomaly = log P("ANOMALOUS" | references, query, prompt)
    p_anomaly = exp(s_anomaly) / (exp(s_anomaly) + exp(s_normal))

i.e. the model's relative preference between the two allowed answers. It is a ranking
score for AUROC/F1-max, not a calibrated probability. Scoring is timed separately and is
not part of TTFT/latency. The discrete prediction comes from the greedy answer; answers
that match neither class fall back to p_anomaly > 0.5 and are flagged.
"""
from __future__ import annotations

import gc
import math
import re
import time
from typing import Dict, List, Optional

import torch
from PIL import Image

from edgecompose.compression.base import TokenCompressor
from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.profiling import memory
from edgeinspect.prompts.inspection import CLASS_TOKENS, classification_content


def parse_label(text: str) -> Optional[int]:
    """1 = anomalous, 0 = normal, None = neither (checked in this order: ABNORMAL/ANOMAL, NOT NORMAL, NORMAL)."""
    t = text.strip().upper()
    if re.search(r"ABNORMAL|ANOMAL|DEFECT", t) or re.search(r"\bNOT\s+NORMAL\b", t):
        return 1
    if re.search(r"\bNORMAL\b", t):
        return 0
    return None


def p_anomaly_from_logprobs(lp: Dict[str, float]) -> Optional[float]:
    if "ANOMALOUS" not in lp or "NORMAL" not in lp:
        return None
    d = lp["NORMAL"] - lp["ANOMALOUS"]
    return 1.0 / (1.0 + math.exp(max(min(d, 700.0), -700.0)))


def classify(
    runner: QwenVLRunner,
    category: str,
    references: List[Image.Image],
    query: Image.Image,
    compressor: Optional[TokenCompressor] = None,
    compress_query: bool = True,
    max_new_tokens: int = 6,
    score: bool = True,
) -> Dict:
    """Run one inspection query; returns prediction, score, timings, tokens, memory, status."""
    k = len(references)
    content = classification_content(category, k)
    images = list(references) + [query]
    mask = [True] * k + [compress_query]
    memory.reset_peak()
    t0 = time.perf_counter()
    try:
        out = runner.run_images(images, content, compressor=compressor, max_new_tokens=max_new_tokens,
                                compress_image_mask=mask, score_candidates=CLASS_TOKENS if score else None)
    except torch.cuda.OutOfMemoryError as e:
        snap = memory.snapshot()
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "OOM", "error_message": str(e).split("\n")[0][:300],
                "peak_allocated_mb": snap.get("peak_allocated_mb"), "peak_reserved_mb": snap.get("peak_reserved_mb"),
                "wall_ms": (time.perf_counter() - t0) * 1e3}
    snap = memory.snapshot()
    pred = parse_label(out.text)
    p_anom = p_anomaly_from_logprobs(out.candidate_logprobs)
    parse_failed = pred is None
    if parse_failed:
        pred = int(p_anom is not None and p_anom > 0.5)
    st = out.stages_ms
    return {
        "status": "ok",
        "error_message": "",
        "raw_output": out.text,
        "prediction": pred,
        "prediction_label": "ANOMALOUS" if pred == 1 else "NORMAL",
        "parse_failed": parse_failed,
        "anomaly_score": p_anom,
        "logp_normal": out.candidate_logprobs.get("NORMAL"),
        "logp_anomalous": out.candidate_logprobs.get("ANOMALOUS"),
        "preprocess_ms": st.get("preprocess"), "vision_ms": st.get("vision"), "compress_ms": st.get("compress"),
        "prefill_ms": st.get("prefill"), "decode_ms": st.get("decode"), "score_ms": st.get("score"),
        "ttft_ms": out.ttft_ms, "total_latency_ms": out.total_latency_ms,
        "generated_tokens": out.generated_tokens,
        "tokens_per_second": out.generated_tokens / (out.total_latency_ms / 1e3),
        "peak_allocated_mb": snap.get("peak_allocated_mb"), "peak_reserved_mb": snap.get("peak_reserved_mb"),
        "visual_tokens_before": out.num_visual_tokens_before, "visual_tokens_after": out.num_visual_tokens_after,
        "per_image_tokens_before": out.per_image_tokens_before, "per_image_tokens_after": out.per_image_tokens_after,
        "prefill_seq_len": out.prefill_seq_len,
    }
