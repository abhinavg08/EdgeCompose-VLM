"""Attention backend selection and capability probing.

Backends (HF `attn_implementation`):

* ``eager``  - explicit matmul + softmax + matmul (materialises the S x S score matrix).
  Used as the *unoptimized* reference so that "optimized attention" has a measurable effect.
* ``sdpa``   - `torch.nn.functional.scaled_dot_product_attention`. PyTorch dispatches to
  FlashAttention, memory-efficient (CUTLASS) or math kernels. On the Windows cu118 build
  used here the flash kernel is *not compiled in*, so SDPA resolves to the
  memory-efficient kernel (probed and recorded below).
* ``flash_attention_2`` - Dao-AILab flash-attn package; optional, only if importable.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)

SUPPORTED = ("eager", "sdpa", "flash_attention_2")


def probe_sdpa_kernels(head_dim: int = 128, seq: int = 256) -> Dict[str, bool]:
    """Return which SDPA kernels can actually execute on the current GPU/build."""
    out: Dict[str, bool] = {}
    if not torch.cuda.is_available():
        return out
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 8, seq, head_dim, dtype=torch.float16, device="cuda")
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            out[name] = True
        except Exception:
            out[name] = False
    return out


def flash_attn_available() -> bool:
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover - platform dependent
        logger.warning("flash_attn present but failed to import: %s", e)
        return False


def resolve_backend(requested: str) -> str:
    """Validate a requested backend; raise if FA2 is requested but unusable."""
    requested = requested.lower()
    if requested not in SUPPORTED:
        raise ValueError(f"attention backend must be one of {SUPPORTED}, got {requested!r}")
    if requested == "flash_attention_2" and not flash_attn_available():
        raise RuntimeError("flash_attention_2 requested but the flash_attn package is not usable")
    return requested


def describe_backend(backend: str) -> str:
    """Human-readable description of the kernel that will actually run."""
    if backend != "sdpa":
        return backend
    kernels = probe_sdpa_kernels()
    for k in ("flash", "mem_efficient", "math"):
        if kernels.get(k):
            return f"sdpa[{k}]"
    return "sdpa[unknown]"
