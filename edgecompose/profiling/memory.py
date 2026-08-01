"""CUDA memory accounting helpers (PyTorch caching-allocator view).

`peak_allocated_mb` (live tensors, peak within the query) is the primary per-query VRAM
metric: it is independent of allocator history. `peak_reserved_mb` includes blocks cached
from earlier (possibly larger) queries, so after a peak reset it reports max(current cache,
this query's need) - it is a run-level footprint, not attributable to one configuration.
The device-level footprint additionally includes the CUDA context (~0.3-0.5 GB).
"""
from __future__ import annotations

from typing import Dict

import torch

MB = 1024.0**2


def reset_peak() -> None:
    """Reset peak statistics; call immediately before each measured query."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def snapshot() -> Dict[str, float]:
    """Current and peak allocated/reserved memory in MB."""
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / MB,
        "reserved_mb": torch.cuda.memory_reserved() / MB,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
    }


def device_used_mb() -> float:
    """Device-wide used memory as seen by the driver (includes CUDA context + other processes)."""
    if not torch.cuda.is_available():
        return float("nan")
    free, total = torch.cuda.mem_get_info()
    return (total - free) / MB
