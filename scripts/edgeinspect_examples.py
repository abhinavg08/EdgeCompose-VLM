"""EdgeInspect qualitative examples (illustration only; no localization benchmark).

Selects 4-5 examples from the final grid (seed 0) with fixed rules, using the normal-only
calibrated decision (score = log P(ANOMALOUS) - log P(NORMAL) vs the per-(category, k, r)
threshold), then renders references + query with the measured score/latency/VRAM and a
Stage-B explanation generated now (explanation time is reported separately).

Selection rules (display configuration k=4, r=75% unless stated):
  1. easy normal        - normal query, correct, lowest margin (score - threshold)
  2. clear structural   - structural anomaly, correct, highest margin
  3. clear logical      - logical anomaly, correct, highest margin
  4. more references    - anomaly missed at k=1/100% but detected at k=8/75%, largest margin gain
  5. compression hurts  - anomaly detected at k=4/100% but missed at k=4/25%, largest margin loss

    python scripts/edgeinspect_examples.py
"""
from __future__ import annotations

import logging
import textwrap

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from edgecompose.compression import build_compressor  # noqa: E402
from edgecompose.utils import PLOTS_DIR, REPO_ROOT, RESULTS_DIR, setup_logging, write_json  # noqa: E402
from edgeinspect.inference.explain import explain  # noqa: E402
from edgeinspect.references.sampler import load_references  # noqa: E402
from edgeinspect.runner import InspectConfig, build_runner, load_image  # noqa: E402
from edgeinspect.datasets.mvtec_loco import LocoImage  # noqa: E402

logger = logging.getLogger("edgeinspect_examples")
AGG = RESULTS_DIR / "edgeinspect" / "aggregate"


def table() -> pd.DataFrame:
    q = pd.read_csv(AGG / "final_per_query.csv")
    q = q[(q["reference_seed"] == 0) & (q["role"] == "test") & (q["status"] == "ok")].copy()
    th = pd.read_csv(AGG / "final_per_category.csv")[["category", "seed", "k", "retention", "threshold_llr"]]
    th = th[th["seed"] == 0].rename(columns={"k": "reference_count", "retention": "token_retention"})
    q = q.merge(th.drop(columns="seed"), on=["category", "reference_count", "token_retention"], how="left")
    q["llr"] = q["logp_anomalous"] - q["logp_normal"]
    q["margin"] = q["llr"] - q["threshold_llr"]
    q["pred_cal"] = (q["margin"] > 0).astype(int)
    return q


def cfg(q: pd.DataFrame, k: int, r: float) -> pd.DataFrame:
    return q[(q["reference_count"] == k) & (q["token_retention"] == r)].set_index("sample_id")


def _first_new_category(ranked_ids, used: set) -> str:
    """Best-ranked sample whose category is not yet shown (falls back to the overall best)."""
    for sid in ranked_ids:
        if sid.split("/")[0] not in used:
            return sid
    return ranked_ids[0]


def select(q: pd.DataFrame) -> list:
    picks, used = [], set()

    def add(story, ranked, k, r):
        if len(ranked):
            sid = _first_new_category(list(ranked), used)
            used.add(sid.split("/")[0])
            picks.append((story, sid, k, r))

    d = cfg(q, 4, 0.75)
    add("easy normal", d[(d["label"] == "normal") & (d["pred_cal"] == 0)].sort_values("margin").index, 4, 0.75)
    for lab, name in (("structural_anomaly", "clear structural anomaly"), ("logical_anomaly", "clear logical anomaly")):
        add(name, d[(d["label"] == lab) & (d["pred_cal"] == 1)].sort_values("margin", ascending=False).index, 4, 0.75)
    k1, k8 = cfg(q, 1, 1.0), cfg(q, 8, 0.75)
    j = k1[["margin", "binary_label"]].join(k8[["margin"]], rsuffix="_k8").dropna()
    j = j[(j["binary_label"] == 1) & (j["margin"] < 0) & (j["margin_k8"] > 0)]
    add("improved by more references (k=1/100% -> k=8/75%)",
        (j["margin_k8"] - j["margin"]).sort_values(ascending=False).index, 8, 0.75)
    c100, c25 = cfg(q, 4, 1.0), cfg(q, 4, 0.25)
    j = c100[["margin", "binary_label"]].join(c25[["margin"]], rsuffix="_25").dropna()
    j = j[(j["binary_label"] == 1) & (j["margin"] > 0) & (j["margin_25"] < 0)]
    add("hurt by aggressive compression (k=4: 100% -> 25%)",
        (j["margin"] - j["margin_25"]).sort_values(ascending=False).index, 4, 0.25)
    return picks


def main() -> None:
    setup_logging()
    q = table()
    picks = select(q)
    runner = build_runner(InspectConfig())
    out = []
    (PLOTS_DIR / "edgeinspect").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(picks), 5, figsize=(12.5, 2.75 * len(picks)))
    for row, (story, sid, k, r) in enumerate(picks):
        rec = q[(q["sample_id"] == sid) & (q["reference_count"] == k) & (q["token_retention"] == r)].iloc[0]
        cat = rec["category"]
        refs = load_references(cat, k, 0)
        ref_imgs = [load_image(x) for x in refs]
        qimg = load_image(LocoImage(sid, cat, "test", rec["label"], rec["query_image"]))
        ex = None
        if rec["pred_cal"] == 1:
            torch.cuda.empty_cache()
            ex = explain(runner, cat, ref_imgs, qimg, build_compressor("visionzip", r))
        other = {}
        if story.startswith("improved"):
            other = q[(q["sample_id"] == sid) & (q["reference_count"] == 1) & (q["token_retention"] == 1.0)].iloc[0]
        if story.startswith("hurt"):
            other = q[(q["sample_id"] == sid) & (q["reference_count"] == 4) & (q["token_retention"] == 1.0)].iloc[0]
        item = {"story": story, "sample_id": sid, "category": cat, "label": rec["label"], "k": k, "retention": r,
                "references": [x.sample_id for x in refs], "score_llr": float(rec["llr"]),
                "threshold_llr": float(rec["threshold_llr"]), "predicted": "ANOMALOUS" if rec["pred_cal"] else "NORMAL",
                "raw_answer": rec["raw_output"], "ttft_ms": float(rec["ttft_ms"]),
                "latency_ms": float(rec["total_latency_ms"]), "peak_allocated_mb": float(rec["peak_allocated_mb"]),
                "peak_reserved_mb": float(rec["peak_reserved_mb"]), "explanation": ex}
        if len(other):
            item["comparison"] = {"k": int(other["reference_count"]), "retention": float(other["token_retention"]),
                                  "score_llr": float(other["llr"]), "threshold_llr": float(other["threshold_llr"]),
                                  "predicted": "ANOMALOUS" if other["pred_cal"] else "NORMAL"}
        out.append(item)
        show = ref_imgs[:4]
        for j in range(4):
            ax = axes[row][j]
            ax.set_axis_off()
            if j < len(show):
                ax.imshow(show[j])
                ax.set_title(f"reference {j + 1}" + (f" (of {k})" if j == 3 and k > 4 else ""), fontsize=7)
        ax = axes[row][4]
        ax.imshow(qimg)
        ax.set_axis_off()
        txt = (f"{story}\n{cat} | truth: {rec['label']}\nk={k}, {int(r * 100)}% tokens -> {item['predicted']}"
               f" (score {rec['llr']:+.2f}, thr {rec['threshold_llr']:+.2f})\n"
               f"TTFT {rec['ttft_ms']:.0f} ms, E2E {rec['total_latency_ms']:.0f} ms, {rec['peak_allocated_mb']:.0f} MB")
        if "comparison" in item:
            c = item["comparison"]
            txt += f"\nvs k={c['k']}, {int(c['retention'] * 100)}%: {c['predicted']} (score {c['score_llr']:+.2f}, thr {c['threshold_llr']:+.2f})"
        if ex and ex.get("issue"):
            txt += "\n" + "\n".join(textwrap.wrap(f"model: {ex.get('anomaly_type')} - {ex.get('issue')}", 60))
        ax.set_title(txt, fontsize=6.3, loc="left")
        logger.info("%s | %s -> %s | %s", story, sid, item["predicted"], (ex or {}).get("raw_explanation", "")[:150])
    fig.suptitle("EdgeInspect-VLM qualitative examples (real model outputs; decision = normal-only calibrated threshold)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "edgeinspect" / "examples.png", dpi=110, bbox_inches="tight")
    write_json(out, RESULTS_DIR / "edgeinspect" / "examples.json")
    logger.info("wrote %d examples", len(out))


if __name__ == "__main__":
    main()
