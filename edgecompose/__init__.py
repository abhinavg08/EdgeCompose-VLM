"""EdgeCompose-VLM: composing VLM inference optimizations on resource-constrained GPUs.

Importing this package applies environment-level settings that must be in place
before `transformers` is imported (disable TensorFlow/JAX backends, AWQ compat shim).
"""
import os

# The host conda env ships TensorFlow; stop transformers from importing it (slow import,
# and TF may try to grab GPU memory which would contaminate VRAM measurements).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# The host env's conda numpy links MKL 2025.3 whose default Intel-OpenMP threading layer
# cannot resolve libiomp5md (not installed) -> any numpy BLAS call (np.dot, matplotlib
# transforms) dies with 0xc06d007f. The sequential layer avoids OpenMP entirely.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from edgecompose.compat import apply_compat_shims  # noqa: E402

apply_compat_shims()

__version__ = "0.1.0"
