"""GPU power/energy measurement through NVML.

Two estimators are recorded per query:

* ``energy_counter_j`` - delta of `nvmlDeviceGetTotalEnergyConsumption` (mJ counter,
  Volta+). Recorded for audit only: on the development laptop GPU it over-reports for
  sub-second windows (implied average power above the board limit).
* ``energy_sampled_j`` - mean of `nvmlDeviceGetPowerUsage` samples x window length, taken
  by a background thread. NVML power readings are themselves averaged by the driver over
  a window, so this is also approximate.

Energy is an optional metric in this project; it never blocks a benchmark run.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PowerMonitor:
    """Background NVML power sampler with an energy-counter cross-check."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self.available = False
        self._nvml = None
        self._handle = None
        self._samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._e0: Optional[int] = None
        self._has_counter = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self.available = True
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self._has_counter = True
            except Exception:
                self._has_counter = False
        except Exception as e:  # pragma: no cover - hardware dependent
            logger.warning("NVML power telemetry unavailable: %s", e)

    def _run(self) -> None:
        assert self._nvml is not None
        while not self._stop.is_set():
            try:
                p = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append((time.perf_counter(), p))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self) -> None:
        if not self.available:
            return
        self._samples = []
        self._stop.clear()
        if self._has_counter:
            self._e0 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        if not self.available or self._thread is None:
            return {"average_power_w": None, "energy_j": None, "energy_sampled_j": None, "energy_counter_j": None}
        t1 = time.perf_counter()
        e1 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) if self._has_counter else None
        self._stop.set()
        self._thread.join()
        self._thread = None
        samples = self._samples
        duration = t1 - self._t0
        sampled = None
        avg = None
        if samples:
            # Samples are (near-)uniformly spaced, so mean power x window length equals the
            # trapezoidal integral up to edge effects, and it covers the whole window.
            avg = sum(p for _, p in samples) / len(samples)
            sampled = avg * duration
        counter = (e1 - self._e0) / 1000.0 if (e1 is not None and self._e0 is not None) else None
        # On the RTX 4060 Laptop the energy counter implied ~160 W averages over ~1 s
        # windows (above the 115 W TGP), i.e. it is not trustworthy for sub-second queries.
        # The sampled estimate is therefore the reported value; the counter is kept for audit.
        energy = sampled
        return {
            "average_power_w": avg,
            "energy_j": energy,
            "energy_sampled_j": sampled,
            "energy_counter_j": counter,
        }

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
