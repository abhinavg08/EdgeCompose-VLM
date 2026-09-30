"""VisionZip for Qwen2.5-VL, isolated from the reference implementation.

Reference: Yang et al., "VisionZip: Longer is Better but Not Necessary in Vision Language
Models" (CVPR 2025), code: https://github.com/dvlab-research/VisionZip
(`Qwen2_5_VL/qwen2_5vl_visionzip.py`, released 2025-05-26, "modified from Transformers
4.49.0").

The reference ships a full fork of the transformers-4.49 modeling file, which does not
run on the transformers 4.57 in our environment. Following the project's fallback
hierarchy ("isolate the token compressor from its larger framework") we port only the
selection logic and keep the unmodified upstream model. The port mirrors the reference:

1. Attention capture: softmax attention weights of the *last* vision block (a
   full-attention block in Qwen2.5-VL-3B: `fullatt_block_indexes=[7,15,23,31]`),
   computed on RoPE-rotated q/k.
2. Importance: mean over heads, sum over queries -> per patch; mean over each 2x2
   merge group -> per LLM visual token; reordered from window order to raster order.
3. Keys: rotated keys averaged over each 2x2 merge group and over heads.
4. Dominant tokens: top-`dominant_num` tokens by importance.
5. Contextual tokens: remaining tokens are split into `contextual_num` uniformly spaced
   targets; every other remaining token is assigned to its most cosine-similar target
   (on keys) and the target embedding becomes `target + mean(assigned)` (as in the
   reference code).
6. Budget: the reference uses `contextual_num = max(int(0.05 N), 1)` and
   `dominant_num = int(r N) - contextual_num` (e.g. 0.65+0.05 for 70%, 0.45+0.05 for
   50%). We use the same split for every retention ratio r.

Only the positional handling differs by construction: contextual tokens live in their
target token's slot and every surviving slot keeps its original M-RoPE position. Decode
positions continue from the (unchanged) maximum position, i.e. identical to the
uncompressed model, rather than being shifted by the number of removed tokens.
"""
# Adapted from VisionZip's Qwen2.5-VL implementation (Apache-2.0).
# Copyright 2025 Senqiao Yang.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified for EdgeCompose: isolated compressor, original M-RoPE positions, and
# per-image attention segments. See THIRD_PARTY_NOTICES.md and third_party/APACHE-2.0.txt.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from edgecompose.compression.base import CompressionResult, TokenCompressor, VisionFeatures


class VisionZipCompression(TokenCompressor):
    name = "visionzip"
    needs_attention = True

    def __init__(self, retention: float, contextual_ratio: float = 0.05) -> None:
        super().__init__(retention)
        self.contextual_ratio = contextual_ratio

    def split_budget(self, n: int) -> tuple[int, int]:
        k = self.target_count(n)
        contextual = max(int(self.contextual_ratio * n), 1)
        if contextual >= k:  # tiny budgets: keep at least one dominant token
            contextual = max(k // 10, 0) if k > 1 else 0
        dominant = k - contextual
        return dominant, contextual

    @torch.no_grad()
    def compress(self, feats: VisionFeatures) -> CompressionResult:
        if feats.importance is None or feats.keys is None:
            raise RuntimeError("VisionZip requires captured vision attention (importance/keys)")
        n = feats.num_tokens
        if self.is_identity():
            return super().compress(feats)
        dominant_num, contextual_num = self.split_budget(n)
        device = feats.embeds.device

        dom_idx = torch.topk(feats.importance.float(), dominant_num).indices
        dom_mask = torch.zeros(n, dtype=torch.bool, device=device)
        dom_mask[dom_idx] = True
        rest_idx = torch.nonzero(~dom_mask, as_tuple=False).squeeze(-1)  # sorted raster order

        keep_mask = dom_mask.clone()
        new_embeds = feats.embeds.clone()
        n_ctx = 0
        if contextual_num > 0 and rest_idx.numel() > 0:
            contextual_num = min(contextual_num, int(rest_idx.numel()))
            metric = F.normalize(feats.keys[rest_idx].float(), dim=-1)
            step = max(1, metric.shape[0] // contextual_num)
            target_local = torch.arange(0, metric.shape[0], step, device=device)[:contextual_num]
            is_target = torch.zeros(metric.shape[0], dtype=torch.bool, device=device)
            is_target[target_local] = True
            merge_local = torch.nonzero(~is_target, as_tuple=False).squeeze(-1)

            target_idx = rest_idx[target_local]
            target_hidden = feats.embeds[target_idx].float()
            if merge_local.numel() > 0:
                sim = metric[merge_local] @ metric[target_local].T  # [M, C]
                assign = sim.argmax(dim=1)
                agg = torch.zeros_like(target_hidden)
                agg.index_add_(0, assign, feats.embeds[rest_idx[merge_local]].float())
                counts = torch.bincount(assign, minlength=target_local.numel()).clamp(min=1).unsqueeze(-1)
                contextual = target_hidden + agg / counts
            else:
                contextual = target_hidden
            new_embeds[target_idx] = contextual.to(new_embeds.dtype)
            keep_mask[target_idx] = True
            n_ctx = int(target_idx.numel())

        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        return CompressionResult(keep_idx, new_embeds[keep_idx], dominant_num, n_ctx)


@torch.no_grad()
def attention_importance_and_keys(
    q: torch.Tensor,
    k: torch.Tensor,
    window_index: torch.Tensor,
    merge_unit: int = 4,
    scale: Optional[float] = None,
    segments: Optional[list] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VisionZip statistics from a full-attention vision block.

    Args:
        q, k: RoPE-rotated query/key tensors ``[S, H, d]`` in the encoder's *window* order.
        window_index: permutation used by Qwen2.5-VL to go raster->window order for merged tokens.
        merge_unit: patches per LLM token (2x2 = 4).
        segments: optional ``[(start, end), ...]`` patch ranges (window order) of individual
            images; attention is computed within each segment only (the full-attention block's
            ``cu_seqlens`` do the same). Default: one segment spanning all patches.
    Returns:
        importance ``[N]`` and keys ``[N, d]`` in raster order (N = S / merge_unit).
    """
    s, h, d = q.shape
    scale = scale if scale is not None else d**-0.5
    col_sum = torch.zeros(s, dtype=torch.float32, device=q.device)
    for a, b in (segments or [(0, s)]):
        # One head at a time keeps the S x S matrix at <= 64 MB for S = 4096 patches.
        for hh in range(h):
            logits = (q[a:b, hh, :] @ k[a:b, hh, :].T).float() * scale
            attn = torch.softmax(logits, dim=-1)
            col_sum[a:b] += attn.sum(dim=0)
            del logits, attn
    importance_patch = col_sum / h  # mean over heads, summed over queries
    importance_w = importance_patch.view(s // merge_unit, merge_unit).mean(dim=-1)
    keys_w = k.float().view(s // merge_unit, merge_unit, h, d).mean(dim=1).mean(dim=1)  # [N, d]
    reverse = torch.argsort(window_index.to(q.device))
    return importance_w[reverse], keys_w[reverse]
