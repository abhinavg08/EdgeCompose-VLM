"""Deterministic normal-reference and query sampling with saved manifests.

References come only from the anomaly-free ``train/good`` split, queries only from
``test``, so a query can never be its own reference. For a given (category, seed) the
reference sets are *nested*: the k=1 set is the first element of the k=2 set, etc. (prefix
of one seeded permutation), so changing k changes only how many references are shown, not
which ones. The same manifest is reused for every token-retention setting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from edgecompose.utils import DATA_DIR
from edgeinspect.datasets.mvtec_loco import LocoImage, load_category

MANIFEST_DIR = DATA_DIR / "manifests" / "edgeinspect"


def _seed_for(category: str, seed: int, salt: str) -> int:
    """Stable per-(category, purpose) seed derived from the user seed (no Python hash())."""
    h = hashlib.sha256(f"{category}|{salt}|{seed}".encode()).hexdigest()
    return int(h[:8], 16)


def sample_references(category: str, k: int, seed: int, root: Optional[Path] = None,
                      pool: str = "train") -> List[LocoImage]:
    """k normal reference images for `category` (nested across k for a fixed seed)."""
    data = load_category(category, root)
    candidates = [x for x in data[pool] if x.label == "normal"]
    if k > len(candidates):
        raise ValueError(f"{category}: requested k={k} but only {len(candidates)} normal {pool} images")
    perm = np.random.default_rng(_seed_for(category, seed, "references")).permutation(len(candidates))
    return [candidates[i] for i in perm[:k]]


def reference_manifest_path(category: str, k: int, seed: int) -> Path:
    return MANIFEST_DIR / f"{category}_k{k}_seed{seed}.json"


def save_reference_manifest(category: str, k: int, seed: int, root: Optional[Path] = None) -> Dict:
    refs = sample_references(category, k, seed, root)
    m = {"category": category, "k": k, "seed": seed, "pool": "train/good",
         "references": [r.to_dict() for r in refs]}
    p = reference_manifest_path(category, k, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2))
    return m


def load_references(category: str, k: int, seed: int) -> List[LocoImage]:
    p = reference_manifest_path(category, k, seed)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/prepare_loco.py")
    m = json.loads(p.read_text())
    return [LocoImage(r["sample_id"], r["category"], r["split"], r["label"], r["path"]) for r in m["references"]]


def sample_queries(category: str, per_label: Dict[str, int], seed: int, root: Optional[Path] = None) -> List[LocoImage]:
    """Stratified query subset from the test split (per label; prefix of a seeded permutation)."""
    data = load_category(category, root)
    out: List[LocoImage] = []
    for label, n in per_label.items():
        pool = [x for x in data["test"] if x.label == label]
        perm = np.random.default_rng(_seed_for(category, seed, f"queries|{label}")).permutation(len(pool))
        out.extend(pool[i] for i in perm[: min(n, len(pool))])
    return out


def query_manifest_path(category: str, name: str) -> Path:
    return MANIFEST_DIR / f"{category}_queries_{name}.json"


def save_query_manifest(category: str, name: str, per_label: Dict[str, int], seed: int,
                        root: Optional[Path] = None) -> Dict:
    qs = sample_queries(category, per_label, seed, root)
    m = {"category": category, "name": name, "seed": seed, "per_label": per_label, "queries": [q.to_dict() for q in qs]}
    p = query_manifest_path(category, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2))
    return m


def load_queries(category: str, name: str) -> List[LocoImage]:
    p = query_manifest_path(category, name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/prepare_loco.py")
    m = json.loads(p.read_text())
    return [LocoImage(q["sample_id"], q["category"], q["split"], q["label"], q["path"]) for q in m["queries"]]
