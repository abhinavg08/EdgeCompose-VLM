"""MVTec LOCO AD loader.

Official layout (Bergmann et al., IJCV 2022)::

    <root>/<category>/train/good/*.png                  anomaly-free training images
    <root>/<category>/validation/good/*.png             anomaly-free validation images
    <root>/<category>/test/good/*.png                   normal test images
    <root>/<category>/test/logical_anomalies/*.png      logical anomalies
    <root>/<category>/test/structural_anomalies/*.png   structural anomalies
    <root>/<category>/ground_truth/<anomaly_type>/<id>/*.png   pixel masks (not used here)

Labels are taken verbatim from the directory structure: ``normal``,
``logical_anomaly``, ``structural_anomaly``; binary label = 0 for normal, 1 otherwise.
The dataset is licensed CC BY-NC-SA 4.0 (non-commercial use only).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from edgecompose.utils import REPO_ROOT

logger = logging.getLogger(__name__)

CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")
TEST_DIRS = {"good": "normal", "logical_anomalies": "logical_anomaly", "structural_anomalies": "structural_anomaly"}
DEFAULT_ROOT = REPO_ROOT / "hf_assets" / "mvtec_loco" / "mvtec_loco_anomaly_detection"


@dataclass(frozen=True)
class LocoImage:
    sample_id: str  # e.g. pushpins/test/logical_anomalies/012
    category: str
    split: str  # train | validation | test
    label: str  # normal | logical_anomaly | structural_anomaly
    path: str  # repo-relative (or absolute if outside the repo)

    @property
    def binary(self) -> int:
        return 0 if self.label == "normal" else 1

    @property
    def anomaly_type(self) -> str:
        return {"normal": "none", "logical_anomaly": "logical", "structural_anomaly": "structural"}[self.label]

    def to_dict(self) -> Dict:
        return {**asdict(self), "binary": self.binary, "anomaly_type": self.anomaly_type}

    def abspath(self) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else REPO_ROOT / p


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p.resolve())


def find_root(root: Optional[Path] = None) -> Path:
    """Locate the extracted dataset (directory containing the category folders)."""
    candidates = [Path(root)] if root else []
    candidates += [DEFAULT_ROOT, DEFAULT_ROOT.parent]
    for c in candidates:
        if c.exists() and all((c / cat).is_dir() for cat in CATEGORIES):
            return c
    raise FileNotFoundError(
        "MVTec LOCO AD not found. Download mvtec_loco_anomaly_detection.tar.xz from "
        "https://www.mvtec.com/company/research/datasets/mvtec-loco/downloads and run scripts/prepare_loco.py"
    )


def load_category(category: str, root: Optional[Path] = None) -> Dict[str, List[LocoImage]]:
    """Return {'train': [...], 'validation': [...], 'test': [...]} sorted by file name."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    base = find_root(root) / category
    out: Dict[str, List[LocoImage]] = {"train": [], "validation": [], "test": []}
    for split in ("train", "validation"):
        for p in sorted((base / split / "good").glob("*.png")):
            out[split].append(LocoImage(f"{category}/{split}/good/{p.stem}", category, split, "normal", _rel(p)))
    for d, label in TEST_DIRS.items():
        for p in sorted((base / "test" / d).glob("*.png")):
            out["test"].append(LocoImage(f"{category}/test/{d}/{p.stem}", category, "test", label, _rel(p)))
    if not out["train"] or not out["test"]:
        raise RuntimeError(f"{category}: empty train or test split under {base}")
    return out


def statistics(root: Optional[Path] = None) -> List[Dict]:
    """Per-category image counts by split and label (for logging / the report)."""
    rows = []
    for cat in CATEGORIES:
        data = load_category(cat, root)
        c = Counter(x.label for x in data["test"])
        rows.append({"category": cat, "train_good": len(data["train"]), "validation_good": len(data["validation"]),
                     "test_good": c["normal"], "test_logical": c["logical_anomaly"],
                     "test_structural": c["structural_anomaly"], "test_total": len(data["test"])})
    return rows
