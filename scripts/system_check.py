"""Stage 0: verify Python/CUDA/GPU and write results/system_info.json."""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
from edgecompose.attention.backends import flash_attn_available, probe_sdpa_kernels
from edgecompose.utils import RESULTS_DIR, collect_system_info, setup_logging, write_json

logger = logging.getLogger("system_check")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS_DIR / "system_info.json"))
    args = ap.parse_args()
    setup_logging()
    info = collect_system_info()
    info["sdpa_kernels"] = probe_sdpa_kernels()
    info["flash_attn_usable"] = flash_attn_available()
    for k, v in info.items():
        logger.info("%-28s %s", k, v)
    if not info.get("cuda_available"):
        raise SystemExit("CUDA is not available - Stage 0 failed")
    write_json(info, __import__("pathlib").Path(args.out))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
