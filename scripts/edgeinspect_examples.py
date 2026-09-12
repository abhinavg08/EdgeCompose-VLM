"""EdgeInspect stage A10: explanation mode + sample visualizations for the README.

For a few test images per category (one logical, one structural, one normal), run Stage A
(classification + anomaly score) and, if classified ANOMALOUS, Stage B (JSON explanation).
Saves results/edgeinspect/examples.json and plots/edgeinspect/example_<category>.png
(references | query, with the real model outputs and measured runtime).

    python scripts/edgeinspect_examples.py --k 4 --retention 1.0
"""
from __future__ import annotations

import argparse
import logging
import textwrap

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from edgecompose.compression import build_compressor  # noqa: E402
from edgecompose.utils import PLOTS_DIR, RESULTS_DIR, setup_logging, write_json  # noqa: E402
from edgeinspect.datasets.mvtec_loco import CATEGORIES  # noqa: E402
from edgeinspect.inference.classify import classify  # noqa: E402
from edgeinspect.inference.explain import explain  # noqa: E402
from edgeinspect.references.sampler import load_queries, load_references  # noqa: E402
from edgeinspect.runner import InspectConfig, build_runner, load_image  # noqa: E402

logger = logging.getLogger("edgeinspect_examples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--retention", type=float, default=1.0)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = ap.parse_args()
    setup_logging()
    runner = build_runner(InspectConfig())
    comp = build_compressor("visionzip", args.retention)
    out = []
    (PLOTS_DIR / "edgeinspect").mkdir(parents=True, exist_ok=True)
    for cat in args.categories:
        refs = load_references(cat, args.k, 0)
        ref_imgs = [load_image(r) for r in refs]
        qs = load_queries(cat, "final")
        picks = [next(q for q in qs if q.label == lab) for lab in ("logical_anomaly", "structural_anomaly", "normal")]
        classify(runner, cat, ref_imgs, load_image(picks[0]), comp)  # warmup
        fig, axes = plt.subplots(len(picks), args.k + 1, figsize=(2.2 * (args.k + 1), 2.9 * len(picks)))
        for row, q in enumerate(picks):
            qi = load_image(q)
            res = classify(runner, cat, ref_imgs, qi, comp)
            ex = explain(runner, cat, ref_imgs, qi, comp) if res.get("prediction") == 1 else None
            rec = {"category": cat, "query": q.sample_id, "label": q.label, "k": args.k, "retention": args.retention,
                   "references": [r.sample_id for r in refs], "classification": res, "explanation": ex}
            out.append(rec)
            logger.info("%s %s (%s) -> %s p=%.3f | %s", cat, q.sample_id, q.label, res.get("prediction_label"),
                        res.get("anomaly_score") or float("nan"), (ex or {}).get("raw_explanation", "")[:160])
            for j, im in enumerate(ref_imgs):
                ax = axes[row][j]
                ax.imshow(im)
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(f"reference {j + 1}", fontsize=8)
            ax = axes[row][args.k]
            ax.imshow(qi)
            ax.set_axis_off()
            txt = (f"query: {q.label}\npred: {res.get('prediction_label')} (score {res.get('anomaly_score', 0):.2f})\n"
                   f"TTFT {res.get('ttft_ms', 0):.0f} ms, E2E {res.get('total_latency_ms', 0):.0f} ms")
            if ex:
                txt += "\n" + "\n".join(textwrap.wrap(f"{ex.get('anomaly_type')}: {ex.get('issue')}", 34))
            ax.set_title(txt, fontsize=6.5, loc="left")
        fig.suptitle(f"EdgeInspect-VLM example - {cat} (k={args.k}, {int(args.retention * 100)}% tokens; real model output)",
                     fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "edgeinspect" / f"example_{cat}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    write_json(out, RESULTS_DIR / "edgeinspect" / "examples.json")
    logger.info("wrote examples")


if __name__ == "__main__":
    main()
