"""Locate fp16 overflow (NaN/Inf) for inputs that produce degenerate '!!!!' outputs.

Hooks every vision block, the merger and every LLM decoder layer, and reports the first
module whose output contains non-finite values plus the max |activation| per stage.

    python scripts/debug_overflow.py
"""
from __future__ import annotations

import glob
import json

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
import pandas as pd
import torch
from PIL import Image

from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.utils import REPO_ROOT


def main() -> None:
    d = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(REPO_ROOT / "results/raw/C0_sdpa_r100__textvqa_500.jsonl"))])
    bad = d[d["prediction"].astype(str).str.contains("!!!")]["sample_id"].tolist()
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(REPO_ROOT / "data/manifests/textvqa_1000.jsonl")}
    r = QwenVLRunner()
    stats = {}

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                stats.setdefault(name, []).append((float(t.float().abs().max()), bool(torch.isfinite(t).all())))
        return f

    hs = [b.register_forward_hook(hook(f"vit.block{i}")) for i, b in enumerate(r.visual.blocks)]
    hs.append(r.visual.merger.register_forward_hook(hook("vit.merger")))
    hs += [l.register_forward_hook(hook(f"llm.layer{i}")) for i, l in enumerate(r.lm.layers)]
    for sid in bad[:3]:
        m = man[sid]
        img = Image.open(REPO_ROOT / m["image_path"]).convert("RGB")
        stats.clear()
        out = r.run(img, m["prompt"], max_new_tokens=4)
        first_bad = next((n for n, v in stats.items() if not all(ok for _, ok in v)), None)
        vit_max = max(v[0][0] for n, v in stats.items() if n.startswith("vit.block"))
        llm_max = max(v[0][0] for n, v in stats.items() if n.startswith("llm."))
        print(f"{sid}: out={out.text!r} size={img.size} first_nonfinite={first_bad} vit_max={vit_max:.0f} "
              f"merger_max={stats['vit.merger'][0][0]:.0f} llm_prefill_max={llm_max:.0f}")
        top = sorted(((v[0][0], n) for n, v in stats.items()), reverse=True)[:5]
        print("   largest |act|:", [(n, round(a)) for a, n in top])
    for h in hs:
        h.remove()


if __name__ == "__main__":
    main()
