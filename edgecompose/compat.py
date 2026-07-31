"""Compatibility shims between pinned third-party packages.

AutoAWQ 0.2.9 (the last release; the project is archived) imports
`transformers.activations.PytorchGELUTanh`, which was renamed to `GELUTanh` in
recent transformers releases. Transformers' AWQ integration imports `awq`, so the
missing symbol breaks loading of *any* AWQ checkpoint. We alias the old name
instead of patching site-packages so the host environment stays untouched.
"""
from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def apply_compat_shims() -> None:
    """Install idempotent aliases required by AutoAWQ on newer transformers."""
    try:
        import transformers.activations as act
    except Exception:  # transformers missing: nothing to shim
        return
    if not hasattr(act, "PytorchGELUTanh") and hasattr(act, "GELUTanh"):
        act.PytorchGELUTanh = act.GELUTanh  # type: ignore[attr-defined]
    # AutoAWQ prints a long deprecation banner on import; it is not actionable here.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"awq.*")
