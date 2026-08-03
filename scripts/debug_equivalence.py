"""Diagnose differences between the instrumented loop and HF generate().

Teacher-forces HF's greedy continuation through (a) the HF model forward and (b) our
language-model path (inputs_embeds + explicit 3-D positions, with/without an all-ones
attention mask) and reports max |logit diff| plus the top-1/top-2 margin at each step.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
import torch

from edgecompose.models.qwen_vl import QwenVLRunner
from smoke_test import synthetic_image


@torch.inference_mode()
def main() -> None:
    r = QwenVLRunner()
    img = synthetic_image()
    q = "What text is written on the red sign?"
    inputs = r.processor(text=[r.build_prompt(q)], images=[img], return_tensors="pt").to(r.device)
    gen = r.model.generate(**inputs, max_new_tokens=32, do_sample=False)
    L = inputs["input_ids"].shape[1]
    new = gen[0, L:]
    print("HF tokens:", r.processor.tokenizer.decode(new))
    full_ids = gen[:, :-1] if new[-1].item() in r.eos_ids else gen
    # (a) HF forward over the whole teacher-forced sequence
    hf = r.model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                 pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]).logits[0].float()
    # (b) our path, one full forward (no cache) with explicit positions
    pos, _ = r.model.model.get_rope_index(full_ids, inputs["image_grid_thw"], None, attention_mask=None)
    emb = r.lm.embed_tokens(full_ids)
    img_emb = r.visual(inputs["pixel_values"].to(r.visual.dtype), grid_thw=inputs["image_grid_thw"])
    emb[full_ids == r.image_token_id] = img_emb.to(emb.dtype)
    for mask_name, mask in (("no-mask", None), ("ones-mask", torch.ones_like(full_ids))):
        ours = r.lm_head(r.lm(inputs_embeds=emb, position_ids=pos, attention_mask=mask).last_hidden_state)[0].float()
        d = (ours[L - 1:] - hf[L - 1:]).abs().max(dim=-1).values
        print(f"[{mask_name}] max |dlogit| per generated step:", [round(x, 3) for x in d.tolist()])
    top2 = hf[L - 1:].topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).tolist()
    print("HF top1-top2 margin per step:", [round(x, 3) for x in margins])
    # (c) our cached incremental loop with teacher forcing
    out = r.run(img, q, max_new_tokens=32)
    print("ours tokens:", r.processor.tokenizer.decode(out.token_ids))


if __name__ == "__main__":
    main()
