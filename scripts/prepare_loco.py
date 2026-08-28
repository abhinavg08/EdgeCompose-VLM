"""EdgeInspect stage A1/A2: extract MVTec LOCO AD, print statistics, write manifests.

    python scripts/prepare_loco.py                       # extract (if needed) + stats + manifests
    python scripts/prepare_loco.py --stats-only

Manifests (data/manifests/edgeinspect/):
  <category>_k{1,2,4,8}_seed{0,1,2}.json   normal references from train/good (nested in k)
  <category>_queries_dev.json               stratified test subset for development
  <category>_queries_final.json             stratified test subset for the full sweep
The archive must be obtained from MVTec (CC BY-NC-SA 4.0):
https://www.mvtec.com/company/research/datasets/mvtec-loco/downloads
"""
from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
from edgecompose.utils import REPO_ROOT, RESULTS_DIR, setup_logging, write_json
from edgeinspect.datasets.mvtec_loco import CATEGORIES, find_root, statistics
from edgeinspect.references.sampler import (
    MANIFEST_DIR, load_references, save_query_manifest, save_reference_manifest,
)

logger = logging.getLogger("prepare_loco")
ARCHIVE = REPO_ROOT / "hf_assets" / "mvtec_loco" / "mvtec_loco_anomaly_detection.tar.xz"


def extract(archive: Path, dest: Path) -> None:
    logger.info("extracting %s -> %s (this takes a few minutes)", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dev-per-label", type=int, nargs=3, default=[50, 25, 25],
                    metavar=("GOOD", "LOGICAL", "STRUCTURAL"))
    ap.add_argument("--final-per-label", type=int, nargs=3, default=[30, 15, 15],
                    metavar=("GOOD", "LOGICAL", "STRUCTURAL"))
    ap.add_argument("--query-seed", type=int, default=1234)
    args = ap.parse_args()
    setup_logging()

    try:
        root = find_root()
    except FileNotFoundError:
        archive = Path(args.archive)
        if not archive.exists():
            raise
        extract(archive, archive.parent / "mvtec_loco_anomaly_detection")
        root = find_root()
    logger.info("dataset root: %s", root)

    stats = statistics(root)
    for r in stats:
        logger.info("%-20s train %3d | val %3d | test good %3d  logical %3d  structural %3d  (total %d)",
                    r["category"], r["train_good"], r["validation_good"], r["test_good"], r["test_logical"],
                    r["test_structural"], r["test_total"])
    write_json({"root": str(root), "categories": stats}, RESULTS_DIR / "edgeinspect" / "loco_statistics.json")
    if args.stats_only:
        return

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        for seed in args.seeds:
            for k in args.ks:
                save_reference_manifest(cat, k, seed, root)
            # nested check: k-reference set is a prefix of the largest set
            big = load_references(cat, max(args.ks), seed)
            for k in args.ks:
                assert [r.sample_id for r in load_references(cat, k, seed)] == [r.sample_id for r in big[:k]]
        for name, per in (("dev", args.dev_per_label), ("final", args.final_per_label)):
            m = save_query_manifest(cat, name, {"normal": per[0], "logical_anomaly": per[1],
                                                "structural_anomaly": per[2]}, args.query_seed, root)
            logger.info("%-20s %s queries: %d", cat, name, len(m["queries"]))
    logger.info("manifests in %s", MANIFEST_DIR)
    # determinism self-check: regenerate one manifest and compare
    p = MANIFEST_DIR / "pushpins_k4_seed0.json"
    before = json.loads(p.read_text())
    save_reference_manifest("pushpins", 4, 0, root)
    assert json.loads(p.read_text()) == before, "reference sampling is not deterministic"
    logger.info("determinism check passed (pushpins k=4 seed=0 regenerated identically)")


if __name__ == "__main__":
    main()
