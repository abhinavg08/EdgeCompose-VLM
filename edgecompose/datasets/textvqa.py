"""TextVQA validation set (5,000 questions) from the `lmms-lab/textvqa` parquet release.

Each question has 10 human answers; scoring uses the standard VQA accuracy
(`edgecompose.evaluation.metrics.vqa_accuracy`). No OCR tokens are given to the model.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import List

import numpy as np

from edgecompose.datasets.base import IMAGE_DIR, SHORT_ANSWER_PROMPT, Sample, save_image_bytes

logger = logging.getLogger(__name__)
HF_REPO = "lmms-lab/textvqa"


def _read_split(parquet_dir: Path, split: str = "validation"):
    import pyarrow.parquet as pq
    import pyarrow as pa

    files = sorted(glob.glob(str(parquet_dir / "data" / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {parquet_dir}/data")
    return pa.concat_tables([pq.read_table(f) for f in files])


def build_textvqa_samples(parquet_dir: Path, n: int, seed: int, split: str = "validation") -> List[Sample]:
    """Deterministically select `n` questions (prefix of a seeded permutation)."""
    table = _read_split(parquet_dir, split)
    qids = [str(x) for x in table.column("question_id").to_pylist()]
    order = np.argsort(np.array(qids, dtype=object), kind="stable")  # canonical order by question_id
    perm = np.random.default_rng(seed).permutation(len(order))
    chosen = [int(order[i]) for i in perm[: min(n, len(order))]]
    logger.info("TextVQA %s: %d questions, selecting %d (seed=%d)", split, len(qids), len(chosen), seed)
    samples: List[Sample] = []
    for row_idx in chosen:
        row = table.slice(row_idx, 1).to_pylist()[0]
        img = row["image"]
        image_id = str(row.get("image_id") or row["question_id"])
        rel = save_image_bytes(img["bytes"], IMAGE_DIR / "textvqa" / f"{image_id}.jpg")
        q = row["question"].strip()
        samples.append(
            Sample(
                sample_id=f"textvqa_{row['question_id']}",
                dataset="textvqa",
                image_path=rel,
                question=q,
                prompt=q + SHORT_ANSWER_PROMPT,
                answers=list(row["answers"]),
                category="",
                meta={"image_id": image_id, "width": row.get("image_width"), "height": row.get("image_height")},
            )
        )
    return samples
