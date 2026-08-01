"""CUDA-synchronized wall-clock timing of inference stages.

Every stage boundary calls `torch.cuda.synchronize()` so that asynchronous kernel
launches are attributed to the stage that issued them. This adds a few microseconds of
host-side overhead per boundary (a handful of boundaries per query), which is negligible
relative to the millisecond-scale stages being measured.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    """Accumulates synchronized wall-clock durations (ms) per named stage."""

    def __init__(self, enabled_sync: bool = True) -> None:
        self.enabled_sync = enabled_sync
        self.stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.enabled_sync:
            _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled_sync:
                _sync()
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def get(self, name: str) -> float:
        return self.stages.get(name, 0.0)

    def total(self) -> float:
        return sum(self.stages.values())
