"""Download model/dataset files from the Hugging Face Hub with resumable curl transfers.

On the development machine `huggingface_hub`'s own downloader (HTTP and hf_xet) stalled
at 0 bytes while plain `curl` against the same resolve URLs worked, so this script lists
repo files through the Hub API and fetches each with `curl -C -` (resumable). Files are
stored under `hf_assets/<repo_type>/<repo_id>/`, a normal directory usable by
`from_pretrained(<path>)`.

    python scripts/download_assets.py --repo Qwen/Qwen2.5-VL-3B-Instruct-AWQ
    python scripts/download_assets.py --repo lmms-lab/textvqa --repo-type dataset --include "data/validation-*"
"""
from __future__ import annotations

import argparse
import fnmatch
import logging
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from edgecompose.utils import REPO_ROOT, setup_logging

logger = logging.getLogger("download_assets")
ASSET_DIR = REPO_ROOT / "hf_assets"


def local_repo_dir(repo: str, repo_type: str = "model") -> Path:
    return ASSET_DIR / repo_type / repo.replace("/", "__")


def download(repo: str, repo_type: str, include: list[str], revision: str = "main") -> Path:
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "dataset":
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        base = f"https://huggingface.co/datasets/{repo}/resolve/{info.sha}"
    else:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        base = f"https://huggingface.co/{repo}/resolve/{info.sha}"
    out_dir = local_repo_dir(repo, repo_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".revision").write_text(info.sha or revision)
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is None:
        raise RuntimeError("curl not found on PATH")
    for s in info.siblings:
        name = s.rfilename
        if include and not any(fnmatch.fnmatch(name, pat) for pat in include):
            continue
        dest = out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if s.size is not None and dest.exists() and dest.stat().st_size == s.size:
            logger.info("ok   %s (%.1f MB)", name, s.size / 1e6)
            continue
        logger.info("get  %s (%.1f MB)", name, (s.size or 0) / 1e6)
        cmd = [curl, "-sS", "-L", "--retry", "10", "--retry-delay", "3", "-C", "-", "-o", str(dest), f"{base}/{name}"]
        for attempt in range(5):
            r = subprocess.run(cmd)
            if s.size is None or (dest.exists() and dest.stat().st_size == s.size):
                break
            logger.warning("retrying %s (attempt %d, rc=%d)", name, attempt + 1, r.returncode)
        if s.size is not None and dest.stat().st_size != s.size:
            raise RuntimeError(f"size mismatch for {name}: {dest.stat().st_size} != {s.size}")
    logger.info("done -> %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    setup_logging()
    download(args.repo, args.repo_type, args.include, args.revision)


if __name__ == "__main__":
    main()
