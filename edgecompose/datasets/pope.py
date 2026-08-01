"""POPE (Polling-based Object Probing Evaluation) from `lmms-lab/POPE` (test, 9,000 questions).

Three negative-sampling settings (random / popular / adversarial), 3,000 questions each,
over 500 COCO val2014 images, 50% yes / 50% no. Subsets are stratified: an equal number
of questions per setting, each a prefix of a seeded per-setting permutation.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import List

import numpy as np

from edgecompose.datasets.base import IMAGE_DIR, SHORT_ANSWER_PROMPT, Sample, save_image_bytes

logger = logging.getLogger(__name__)
HF_REPO = "lmms-lab/POPE"
CATEGORIES = ("adversarial", "popular", "random")


def _read(parquet_dir: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(parquet_dir / "data" / "test-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no POPE test parquet files under {parquet_dir}/data")
    return pa.concat_tables([pq.read_table(f) for f in files])


def build_pope_samples(parquet_dir: Path, n: int, seed: int) -> List[Sample]:
    table = _read(parquet_dir)
    cats = table.column("category").to_pylist()
    qids = [str(x) for x in table.column("question_id").to_pylist()]
    counts = [n // len(CATEGORIES) + (1 if i < n % len(CATEGORIES) else 0) for i in range(len(CATEGORIES))]
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for c, k in zip(CATEGORIES, counts):
        rows = sorted((i for i, cc in enumerate(cats) if cc == c), key=lambda i: (len(qids[i]), qids[i]))
        perm = rng.permutation(len(rows))  # drawn for every category regardless of n -> nested subsets
        chosen.extend(rows[j] for j in perm[:k])
    logger.info("POPE: %d questions, selecting %d (%s per category, seed=%d)", len(qids), len(chosen),
                dict(zip(CATEGORIES, counts)), seed)
    samples: List[Sample] = []
    for row_idx in chosen:
        row = table.slice(row_idx, 1).to_pylist()[0]
        src = str(row["image_source"])
        rel = save_image_bytes(row["image"]["bytes"], IMAGE_DIR / "pope" / f"{src}.jpg")
        q = row["question"].strip()
        samples.append(
            Sample(
                sample_id=f"pope_{row['category']}_{row['question_id']}",
                dataset="pope",
                image_path=rel,
                question=q,
                prompt=q + SHORT_ANSWER_PROMPT,
                answers=[str(row["answer"]).strip().lower()],
                category=str(row["category"]),
                meta={"image_source": src, "id": row.get("id")},
            )
        )
    return samples
