"""Shared utilities: paths, logging, seeding, config loading, system info."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_model_path(model_id: str) -> str:
    """Prefer a local copy in hf_assets/ (scripts/download_assets.py), else the Hub id."""
    local = REPO_ROOT / "hf_assets" / "model" / model_id.replace("/", "__")
    if (local / "config.json").exists():
        return str(local)
    return model_id


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure structured (timestamp | level | module | message) logging."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs (decoding itself is greedy/deterministic)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class ExperimentConfig:
    """One point in the experiment matrix (see configs/*.yaml)."""

    name: str
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    quantization: str = "awq-int4"
    attention_backend: str = "sdpa"  # sdpa | eager | flash_attention_2
    token_method: str = "none"  # none | visionzip | uniform
    token_retention: float = 1.0
    contextual_ratio: float = 0.05  # VisionZip contextual tokens as a fraction of all visual tokens
    awq_dequant_threshold: int = 1024  # AutoAWQ: dequant+cuBLAS when batch*seq >= this (upstream 1024)
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1024 * 28 * 28
    max_new_tokens: int = 32
    warmup: int = 5
    seed: int = 1234
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Read YAML tolerating a UTF-8 BOM (Windows PowerShell 5.1 writes one)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Path | str) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML. Unknown keys are kept in `extra`."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    raw = load_yaml(path)
    known = {k: v for k, v in raw.items() if k in ExperimentConfig.__dataclass_fields__}
    extra = {k: v for k, v in raw.items() if k not in ExperimentConfig.__dataclass_fields__}
    known.setdefault("name", path.stem)
    cfg = ExperimentConfig(**known)
    cfg.extra.update(extra)
    return cfg


def collect_system_info() -> Dict[str, Any]:
    """Collect hardware/software provenance for results/system_info.json."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import psutil  # optional

        info["system_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            info["system_ram_gb"] = round(stat.ullTotalPhys / 1024**3, 1)
        except Exception:
            info["system_ram_gb"] = None
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_mb"] = round(props.total_memory / 1024**2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            info["gpu_sm_count"] = props.multi_processor_count
    except ImportError:
        info["torch_version"] = None
    for pkg in ("transformers", "accelerate", "autoawq", "triton-windows", "triton", "flash-attn", "nvidia-ml-py"):
        try:
            from importlib.metadata import version

            info[f"{pkg}_version"] = version(pkg)
        except Exception:
            info[f"{pkg}_version"] = None
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(info["nvidia_driver"], bytes):
            info["nvidia_driver"] = info["nvidia_driver"].decode()
        try:
            info["gpu_power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception:
            info["gpu_power_limit_w"] = None
        pynvml.nvmlShutdown()
    except Exception:
        info["nvidia_driver"] = None
    return info


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
