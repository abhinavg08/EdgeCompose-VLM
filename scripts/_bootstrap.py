"""Make the repository importable when scripts are run as `python scripts/x.py`.

Also sets MKL_THREADING_LAYER before numpy can be imported (see edgecompose/__init__.py).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
