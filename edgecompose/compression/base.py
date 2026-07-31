"""Visual-token compression interface.

A compressor receives the vision encoder's output tokens for one image (post
PatchMerger, i.e. exactly the tokens the language model would consume) in raster order,
and returns which of the image-token *slots* of the LLM sequence survive together with
the embeddings to place there. Surviving slots keep their original 3-D M-RoPE position
ids, so a compressed sequence is literally the uncompressed sequence with some image
tokens deleted (and, for VisionZip, some tokens replaced by merged "contextual" tokens).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VisionFeatures:
    """Vision-encoder outputs for a single image, all in raster (row-major) token order."""

    embeds: torch.Tensor  # [N, D_llm] merged visual tokens fed to the LLM
    grid_thw: torch.Tensor  # [1, 3] patch grid (t, h, w) before 2x2 merging
    importance: Optional[torch.Tensor] = None  # [N] attention received (VisionZip)
    keys: Optional[torch.Tensor] = None  # [N, d_head] head-averaged keys (VisionZip)

    @property
    def num_tokens(self) -> int:
        return int(self.embeds.shape[0])


@dataclass
class CompressionResult:
    keep_indices: torch.Tensor  # [K] sorted raster indices of image slots that survive
    embeds: torch.Tensor  # [K, D_llm] embeddings for those slots (same order)
    num_dominant: int
    num_contextual: int


class TokenCompressor:
    """Base class. `needs_attention` asks the runner to capture vision attention."""

    name: str = "none"
    needs_attention: bool = False

    def __init__(self, retention: float = 1.0) -> None:
        if not 0.0 < retention <= 1.0:
            raise ValueError(f"retention must be in (0, 1], got {retention}")
        self.retention = retention

    def target_count(self, n: int) -> int:
        return max(1, min(n, int(round(self.retention * n))))

    def is_identity(self) -> bool:
        return self.retention >= 1.0

    def compress(self, feats: VisionFeatures) -> CompressionResult:
        n = feats.num_tokens
        idx = torch.arange(n, device=feats.embeds.device)
        return CompressionResult(idx, feats.embeds, n, 0)


class NoCompression(TokenCompressor):
    name = "none"

    def __init__(self, retention: float = 1.0) -> None:
        super().__init__(1.0)


class UniformCompression(TokenCompressor):
    """Deterministic control: keep K tokens evenly spaced in raster order (no attention)."""

    name = "uniform"

    def compress(self, feats: VisionFeatures) -> CompressionResult:
        n = feats.num_tokens
        k = self.target_count(n)
        if k >= n:
            return super().compress(feats)
        idx = torch.linspace(0, n - 1, steps=k, device=feats.embeds.device).round().long().unique()
        return CompressionResult(idx, feats.embeds[idx], int(idx.numel()), 0)


def build_compressor(method: str, retention: float, **kwargs) -> TokenCompressor:
    """Factory used by configs: method in {none, visionzip, uniform}."""
    method = (method or "none").lower()
    if method == "none" or retention >= 1.0:
        return NoCompression()
    if method == "uniform":
        return UniformCompression(retention)
    if method == "visionzip":
        from edgecompose.compression.visionzip import VisionZipCompression

        return VisionZipCompression(retention, **kwargs)
    raise ValueError(f"unknown token compression method: {method}")
