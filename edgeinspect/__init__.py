"""EdgeInspect-VLM: few-shot industrial visual inspection on top of EdgeCompose-VLM.

Reuses EdgeCompose's instrumented Qwen2.5-VL-3B-AWQ runner, visual-token compression,
attention/AWQ-dispatch settings and profiling; adds MVTec LOCO AD data handling,
normal-reference sampling, inspection prompts, likelihood-based anomaly scores and
anomaly-detection metrics.
"""
import edgecompose  # noqa: F401  (environment + compatibility shims)

__version__ = "0.1.0"
