import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")  # see edgecompose/__init__.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
