"""Dataset manifests: fixed, seeded sample lists shared by every configuration.

A manifest is a JSONL file (data/manifests/<dataset>_<n>.jsonl) whose rows are
`Sample`s with images extracted to data/images/<dataset>/. Subsets are prefixes of one
seeded permutation, so the 200-sample development subset is contained in every larger
subset and all configurations are scored on identical sample IDs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from edgecompose.utils import DATA_DIR, REPO_ROOT

MANIFEST_DIR = DATA_DIR / "manifests"
IMAGE_DIR = DATA_DIR / "images"
SHORT_ANSWER_PROMPT = "\nAnswer the question using a single word or phrase."


@dataclass
class Sample:
    sample_id: str
    dataset: str
    image_path: str  # relative to repo root
    question: str  # raw question
    prompt: str  # question + instruction actually sent to the model
    answers: List[str]
    category: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        return Image.open(REPO_ROOT / self.image_path).convert("RGB")


def manifest_path(dataset: str, n: int) -> Path:
    return MANIFEST_DIR / f"{dataset}_{n}.jsonl"


def write_manifest(samples: List[Sample], path: Path) -> str:
    """Write a manifest and return the sha256 of its sample-id list (for provenance)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return ids_digest([s.sample_id for s in samples])


def load_manifest(path: Path | str) -> List[Sample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest {path} not found - run scripts/prepare_data.py first")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Sample(**json.loads(line)))
    return out


def ids_digest(ids: List[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def save_image_bytes(data: bytes, dest: Path) -> str:
    """Write raw encoded image bytes once; return the repo-relative path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return dest.relative_to(REPO_ROOT).as_posix()
