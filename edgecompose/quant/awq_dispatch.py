"""Configurable AutoAWQ GEMM dispatch threshold.

AutoAWQ 0.2.9 (`awq/modules/linear/gemm.py`) chooses the kernel for a W4A16 linear layer
from the number of activation rows ``M = batch * seq``:

    M >= 1024 : dequantize the whole INT4 weight to FP16 (Triton) + cuBLAS FP16 GEMM
    M <  1024 : fused Triton split-K W4A16 GEMM

The constant 1024 is hard-coded. On the RTX 4060 the fused Triton GEMM is much slower
than dequantize + cuBLAS for the medium M values that visual-token compression produces,
so token pruning can *increase* prefill time. This module replaces the autograd
function's forward with an identical copy whose threshold is configurable
(`set_dequant_threshold`). With the default (1024) behaviour is byte-for-byte upstream.
This is a kernel-selection setting of existing kernels, not a new kernel.
"""
# Adapted from AutoAWQ awq/modules/linear/gemm.py (MIT License).
# Copyright (c) 2023 MIT HAN Lab.
# Modified for EdgeCompose: configurable GEMM dispatch threshold.
# Full upstream license: third_party/AutoAWQ-MIT.txt.
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

UPSTREAM_THRESHOLD = 1024
_state = {"threshold": UPSTREAM_THRESHOLD, "patched": False}


def _forward(ctx, x, qweight, qzeros, scales, w_bit=4, group_size=128, bias=None, out_features=0):
    from awq.modules.linear import gemm as g

    ctx.save_for_backward(x, qweight, qzeros, scales, bias)
    ctx.out_features = out_features
    out_shape = x.shape[:-1] + (out_features,)
    x = x.to(torch.float16)
    if x.shape[0] == 0:
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    use_dequant = x.shape[0] * x.shape[1] >= _state["threshold"]
    if g.awq_ext is not None:
        if use_dequant:
            out = g.awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0, 0, 0, False)
            out = torch.matmul(x, out)
        else:
            out = g.awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8)
    elif g.TRITON_AVAILABLE:
        if use_dequant:
            out = g.awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x, out.to(x.dtype))
        else:
            out = g.awq_gemm_triton(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8)
    else:
        out = g.dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)
    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)
    if len(out.shape) == 2:
        out = out.unsqueeze(0)
    return out


def set_dequant_threshold(threshold: int) -> None:
    """Use dequantize+cuBLAS when batch*seq >= threshold (upstream: 1024)."""
    threshold = int(threshold)
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    if not _state["patched"]:
        from awq.modules.linear import gemm as g

        g.WQLinearMMFunction.forward = staticmethod(_forward)
        _state["patched"] = True
    if threshold != _state["threshold"]:
        logger.debug("AWQ dequant threshold %d -> %d", _state["threshold"], threshold)
    _state["threshold"] = threshold


def get_dequant_threshold() -> int:
    return _state["threshold"]
