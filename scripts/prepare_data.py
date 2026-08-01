"""Stage 2: build deterministic evaluation manifests for TextVQA and POPE.

    python scripts/prepare_data.py --n 200            # development subsets
    python scripts/prepare_data.py --n 1000 --datasets textvqa pope

Parquet files are read from hf_assets/ (see scripts/download_assets.py), downloaded on
demand. Images are extracted once to data/images/. Manifest provenance (seed, sample-ID
digest) is recorded in data/manifests/manifest_info.json.
"""
from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
from download_assets import download, local_repo_dir
from edgecompose.datasets import pope, textvqa
from edgecompose.datasets.base import MANIFEST_DIR, load_manifest, manifest_path, write_manifest
from edgecompose.utils import setup_logging

logger = logging.getLogger("prepare_data")


def ensure_parquets(repo: str, pattern: str):
    d = local_repo_dir(repo, "dataset")
    if not list(d.glob(pattern)):
        download(repo, "dataset", [pattern])
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[200])
    ap.add_argument("--datasets", nargs="+", default=["textvqa", "pope"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    setup_logging()

    info_path = MANIFEST_DIR / "manifest_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for ds in args.datasets:
        for n in args.n:
            if ds == "textvqa":
                d = ensure_parquets(textvqa.HF_REPO, "data/validation-*.parquet")
                samples = textvqa.build_textvqa_samples(d, n, args.seed)
            elif ds == "pope":
                d = ensure_parquets(pope.HF_REPO, "data/test-*.parquet")
                samples = pope.build_pope_samples(d, n, args.seed)
            else:
                raise SystemExit(f"unknown dataset {ds}")
            path = manifest_path(ds, n)
            digest = write_manifest(samples, path)
            again = load_manifest(path)  # round-trip check
            assert [s.sample_id for s in again] == [s.sample_id for s in samples]
            info[path.name] = {"dataset": ds, "n": len(samples), "seed": args.seed, "ids_sha256_16": digest}
            logger.info("%s: %d samples, id digest %s", path.name, len(samples), digest)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
