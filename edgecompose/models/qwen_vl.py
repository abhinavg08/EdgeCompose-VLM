"""Instrumented Qwen2.5-VL inference with pluggable visual-token compression.

Instead of calling `model.generate()` (which hides stage boundaries), the runner drives
the unmodified Hugging Face sub-modules directly:

    preprocess : chat template + image resize/patchify (CPU) + host->device copy
    vision     : `model.model.visual` (ViT + PatchMerger) on GPU
    compress   : text embedding lookup, image-token fusion, M-RoPE position ids,
                 and (optionally) token compression incl. its attention statistics
    prefill    : one forward of `model.model.language_model` over the whole prompt,
                 LM head on the last position, greedy argmax -> first token
    decode     : greedy autoregressive loop with a KV cache until EOS/max_new_tokens

TTFT = preprocess + vision + compress + prefill. Every boundary is CUDA-synchronized.

Decoding is pure greedy argmax (no repetition penalty / sampling), identical for every
configuration. Note: the AWQ decode GEMM (Triton, split-K) is not bit-deterministic across
different call sequences (logit jitter ~0.04), but repeated runs of the same query are
reproducible in practice (verified 3/3 in the smoke-test diagnostics).
`reference_generate()` runs HF `generate()` so the smoke test can verify that the custom
loop produces identical greedy outputs at 100% retention.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

import edgecompose  # noqa: F401  (env + compat shims before transformers import)
from edgecompose.attention.backends import resolve_backend
from edgecompose.compression.base import NoCompression, TokenCompressor, VisionFeatures
from edgecompose.compression.visionzip import attention_importance_and_keys
from edgecompose.profiling import memory
from edgecompose.profiling.latency import StageTimer
from edgecompose.utils import resolve_model_path

logger = logging.getLogger(__name__)


@dataclass
class InferenceOutput:
    text: str
    generated_tokens: int
    stages_ms: Dict[str, float]
    ttft_ms: float
    total_latency_ms: float
    num_visual_tokens_before: int
    num_visual_tokens_after: int
    num_dominant: int
    num_contextual: int
    prefill_seq_len: int
    image_grid_thw: List[int]
    token_ids: List[int] = field(default_factory=list)
    per_image_tokens_before: List[int] = field(default_factory=list)
    per_image_tokens_after: List[int] = field(default_factory=list)
    candidate_logprobs: Dict[str, float] = field(default_factory=dict)

    @property
    def decode_steps(self) -> int:
        return max(self.generated_tokens - 1, 0)


class _AttnCapture:
    """Forward-pre-hook that stores the inputs of a vision attention module (no compute)."""

    def __init__(self) -> None:
        self.hidden: Optional[torch.Tensor] = None
        self.position_embeddings: Optional[tuple] = None

    def __call__(self, module, args, kwargs):  # noqa: ANN001
        self.hidden = args[0] if args else kwargs["hidden_states"]
        self.position_embeddings = kwargs.get("position_embeddings")
        return None

    def clear(self) -> None:
        self.hidden = None
        self.position_embeddings = None


class QwenVLRunner:
    """Loads Qwen2.5-VL once and serves instrumented single-image queries (batch size 1)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        attention_backend: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = model_id
        self.attention_backend = resolve_backend(attention_backend)
        self.device = torch.device(device)
        self.dtype = dtype

        memory.reset_peak()
        before = memory.device_used_mb()
        t0 = time.perf_counter()
        path = resolve_model_path(model_id)
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            attn_implementation=self.attention_backend,
            device_map={"": self.device.index or 0},
        ).eval()
        torch.cuda.synchronize()
        self.load_time_s = time.perf_counter() - t0
        snap = memory.snapshot()
        self.model_load_allocated_mb = snap.get("allocated_mb", float("nan"))
        self.model_load_device_mb = memory.device_used_mb() - before
        logger.info(
            "loaded %s (attn=%s) in %.1fs; allocated %.0f MB",
            model_id, self.attention_backend, self.load_time_s, self.model_load_allocated_mb,
        )

        cfg = self.model.config
        self.image_token_id: int = cfg.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if self.processor.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.processor.tokenizer.eos_token_id)

        self.visual = self.model.model.visual
        self.lm = self.model.model.language_model
        self.lm_head = self.model.lm_head
        full_idx = list(self.visual.fullatt_block_indexes)
        self._capture_block = max(full_idx)
        if self._capture_block != len(self.visual.blocks) - 1:
            logger.warning("last vision block is windowed; VisionZip uses full-attn block %d", self._capture_block)
        self._capture = _AttnCapture()
        self._capture_handle = None

    # ------------------------------------------------------------------ helpers
    def set_attention_backend(self, backend: str) -> None:
        """Switch eager/sdpa/fa2 in place (vision tower + LLM) without reloading weights.

        Uses HF `set_attn_implementation`; verified afterwards on both sub-configs because
        HF marks changed sub-configs with a flag and could silently skip them.
        """
        backend = resolve_backend(backend)
        if backend == self.attention_backend:
            return
        self.model.set_attn_implementation(backend)
        for cfg in (self.model.config, self.visual.config, self.lm.config):
            if hasattr(cfg, "_attn_was_changed"):
                delattr(cfg, "_attn_was_changed")
        got = (self.visual.config._attn_implementation, self.lm.config._attn_implementation)
        if got != (backend, backend):
            raise RuntimeError(f"attention switch to {backend} failed: vision/text = {got}")
        self.attention_backend = backend

    def build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _set_capture(self, enabled: bool) -> None:
        if enabled and self._capture_handle is None:
            attn = self.visual.blocks[self._capture_block].attn
            self._capture_handle = attn.register_forward_pre_hook(self._capture, with_kwargs=True)
        elif not enabled and self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None
            self._capture.clear()

    def _visionzip_stats(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VisionZip statistics for all images; attention is computed *within* each image,
        matching the full-attention block (whose cu_seqlens separate images)."""
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

        attn = self.visual.blocks[self._capture_block].attn
        hs = self._capture.hidden
        s = hs.shape[0]
        q, k, _ = attn.qkv(hs).reshape(s, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = self._capture.position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        window_index, _ = self.visual.get_window_index(grid_thw)
        patches = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()  # per image, contiguous in window order
        bounds, start = [], 0
        for n in patches:
            bounds.append((start, start + int(n)))
            start += int(n)
        imp, keys = attention_importance_and_keys(q, k, window_index, self.visual.spatial_merge_unit, attn.scaling,
                                                  segments=bounds)
        self._capture.clear()
        return imp, keys

    # ------------------------------------------------------------------ main API
    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        question: str,
        compressor: Optional[TokenCompressor] = None,
        max_new_tokens: int = 32,
        ignore_eos: bool = False,
    ) -> InferenceOutput:
        """Answer one question about one image, returning text and per-stage timings."""
        content = [{"type": "image"}, {"type": "text", "text": question}]
        return self.run_images([image], content, compressor=compressor, max_new_tokens=max_new_tokens,
                               ignore_eos=ignore_eos)

    @torch.inference_mode()
    def run_images(
        self,
        images: List[Image.Image],
        content: List[Dict[str, Any]],
        compressor: Optional[TokenCompressor] = None,
        max_new_tokens: int = 32,
        ignore_eos: bool = False,
        compress_image_mask: Optional[List[bool]] = None,
        score_candidates: Optional[Dict[str, str]] = None,
    ) -> InferenceOutput:
        """General multi-image query (batch size 1).

        Args:
            images: images in the order their ``{"type": "image"}`` placeholders appear in ``content``.
            content: chat-message content list (interleaved text and image placeholders).
            compressor: applied independently to every image (same retention per image).
            compress_image_mask: optional per-image flags; False keeps that image uncompressed.
            score_candidates: optional {label: text}; after generation, the exact log-probability
                of each candidate continuation is computed by teacher forcing on the prompt's KV
                cache (timed separately as ``score`` and excluded from latency/TTFT).
        """
        compressor = compressor or NoCompression()
        use_capture = compressor.needs_attention and not compressor.is_identity()
        self._set_capture(use_capture)
        timer = StageTimer()

        with timer.stage("preprocess"):
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=list(images), return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device, non_blocking=True)
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.visual.dtype, non_blocking=True)
            grid_thw = inputs["image_grid_thw"].to(self.device)

        with timer.stage("vision"):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        with timer.stage("compress"):
            position_ids, _ = self.model.model.get_rope_index(input_ids, grid_thw, None, attention_mask=None)
            embeds = self.lm.embed_tokens(input_ids)
            image_mask = input_ids[0] == self.image_token_id
            image_slots = torch.nonzero(image_mask, as_tuple=False).squeeze(-1)
            n_before = int(image_slots.numel())
            if n_before != image_embeds.shape[0]:
                raise RuntimeError(f"image token/feature mismatch: {n_before} vs {image_embeds.shape[0]}")
            image_embeds = image_embeds.to(embeds.dtype)
            importance = keys = None
            if use_capture:
                importance, keys = self._visionzip_stats(grid_thw)
            merge = self.visual.spatial_merge_size ** 2
            per_image = [int(n) // merge for n in (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()]
            keep_idx, keep_emb, per_after = [], [], []
            n_dom = n_ctx = 0
            off = 0
            for i, n in enumerate(per_image):
                sl = slice(off, off + n)
                feats = VisionFeatures(embeds=image_embeds[sl], grid_thw=grid_thw[i:i + 1],
                                       importance=None if importance is None else importance[sl],
                                       keys=None if keys is None else keys[sl])
                comp = compressor if (compress_image_mask is None or compress_image_mask[i]) else NoCompression()
                res = comp.compress(feats)
                keep_idx.append(res.keep_indices + off)
                keep_emb.append(res.embeds)
                per_after.append(int(res.keep_indices.numel()))
                n_dom += res.num_dominant
                n_ctx += res.num_contextual
                off += n
            keep_idx_t = torch.cat(keep_idx)
            keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=self.device)
            keep[image_slots] = False
            keep[image_slots[keep_idx_t]] = True
            embeds[0, image_slots[keep_idx_t]] = torch.cat(keep_emb)
            embeds = embeds[:, keep]
            position_ids = position_ids[:, :, keep]
            next_pos = int(position_ids.max().item()) + 1
            seq_len = int(embeds.shape[1])

        with timer.stage("prefill"):
            from transformers import DynamicCache

            cache = DynamicCache(config=self.lm.config)
            # All-ones 2-D mask exactly as HF generate() passes it; this makes the loop
            # bit-identical to generate() (verified by scripts/debug_equivalence.py).
            attn_mask = torch.ones(1, seq_len + max_new_tokens, dtype=torch.long, device=self.device)
            out = self.lm(
                inputs_embeds=embeds,
                attention_mask=attn_mask[:, :seq_len],
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(seq_len, device=self.device),
            )
            logits = self.lm_head(out.last_hidden_state[:, -1:, :])
            next_tok = logits.argmax(dim=-1)  # [1, 1]
            first_id = int(next_tok.item())
        ttft = timer.total()

        generated = [first_id]
        with timer.stage("decode"):
            cur = next_tok
            step = 0
            while len(generated) < max_new_tokens and (ignore_eos or generated[-1] not in self.eos_ids):
                pos = torch.full((3, 1, 1), next_pos + step, dtype=torch.long, device=self.device)
                out = self.lm(
                    input_ids=cur,
                    attention_mask=attn_mask[:, : seq_len + step + 1],
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([seq_len + step], device=self.device),
                )
                cur = self.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
                generated.append(int(cur.item()))
                step += 1
        total_ms = timer.total()

        cand_logprobs: Dict[str, float] = {}
        if score_candidates:
            with timer.stage("score"):
                first_logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
                for label, cand_text in score_candidates.items():
                    ids = self.processor.tokenizer(cand_text, add_special_tokens=False)["input_ids"]
                    lp = float(first_logp[ids[0]])
                    if len(ids) > 1:
                        cache.crop(seq_len)  # drop generated tokens; keep the prompt KV cache
                        n = len(ids) - 1
                        pos = (next_pos + torch.arange(n, device=self.device)).view(1, 1, -1).expand(3, 1, -1)
                        out = self.lm(
                            input_ids=torch.tensor([ids[:-1]], device=self.device),
                            attention_mask=torch.ones(1, seq_len + n, dtype=torch.long, device=self.device),
                            position_ids=pos,
                            past_key_values=cache,
                            use_cache=True,
                            cache_position=torch.arange(seq_len, seq_len + n, device=self.device),
                        )
                        lps = torch.log_softmax(self.lm_head(out.last_hidden_state[0]).float(), dim=-1)
                        lp += float(sum(lps[j, ids[j + 1]] for j in range(n)))
                    cand_logprobs[label] = lp
        del cache

        text_ids = [t for t in generated if t not in self.eos_ids] if not ignore_eos else generated
        answer = self.processor.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        stages = dict(timer.stages)
        return InferenceOutput(
            text=answer,
            generated_tokens=len(generated),
            stages_ms=stages,
            ttft_ms=ttft,
            total_latency_ms=total_ms,
            num_visual_tokens_before=n_before,
            num_visual_tokens_after=int(keep_idx_t.numel()),
            num_dominant=n_dom,
            num_contextual=n_ctx,
            prefill_seq_len=seq_len,
            image_grid_thw=[int(x) for x in grid_thw[0].tolist()],
            token_ids=generated,
            per_image_tokens_before=per_image,
            per_image_tokens_after=per_after,
            candidate_logprobs=cand_logprobs,
        )

    @torch.inference_mode()
    def reference_generate(self, image: Image.Image, question: str, max_new_tokens: int = 32) -> str:
        """Plain HF `generate()` (pure greedy) for correctness cross-checks.

        The checkpoint's generation_config sets repetition_penalty=1.05, which HF applies
        even with do_sample=False. The benchmark uses pure greedy decoding (no penalty),
        so the reference disables it for a like-for-like comparison.
        """
        text = self.build_prompt(question)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                                  temperature=None, top_p=None, top_k=None)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()

    def describe(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "attention_backend": self.attention_backend,
            "model_load_time_s": self.load_time_s,
            "model_load_allocated_mb": self.model_load_allocated_mb,
            "model_load_device_mb": self.model_load_device_mb,
            "quantization_config": getattr(self.model.config, "quantization_config", None),
        }
