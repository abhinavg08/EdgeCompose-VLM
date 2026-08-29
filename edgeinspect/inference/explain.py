"""Stage B: structured explanation for queries classified ANOMALOUS (not part of benchmark latency)."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from PIL import Image

from edgecompose.compression.base import TokenCompressor
from edgecompose.models.qwen_vl import QwenVLRunner
from edgeinspect.prompts.inspection import explanation_content


def parse_json_object(text: str) -> Optional[Dict]:
    """Extract the first JSON object from model output (tolerates code fences / prose)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def explain(
    runner: QwenVLRunner,
    category: str,
    references: List[Image.Image],
    query: Image.Image,
    compressor: Optional[TokenCompressor] = None,
    compress_query: bool = True,
    max_new_tokens: int = 96,
) -> Dict:
    content = explanation_content(category, len(references))
    out = runner.run_images(list(references) + [query], content, compressor=compressor, max_new_tokens=max_new_tokens,
                            compress_image_mask=[True] * len(references) + [compress_query])
    obj = parse_json_object(out.text) or {}
    atype = str(obj.get("anomaly_type", "")).lower()
    return {
        "raw_explanation": out.text,
        "json_ok": bool(obj),
        "status": obj.get("status", "ANOMALOUS"),
        "anomaly_type": atype if atype in ("structural", "logical") else None,
        "issue": obj.get("issue"),
        "explanation": obj.get("explanation"),
        "explain_latency_ms": out.total_latency_ms,
        "explain_tokens": out.generated_tokens,
    }
