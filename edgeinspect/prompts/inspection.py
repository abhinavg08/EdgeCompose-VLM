"""Inspection prompts (few-shot: k known-good references + 1 query).

Design choices (kept fixed across all configurations):
* Only the product *name* is given per category - no category-specific rules (e.g. the
  breakfast-box composition). The references are the only source of "normal".
* Every image is preceded by a text label ("Reference 1 (known-good):", "Query image:"),
  so the model can tell references from the query.
* Stage A asks for exactly one word so the benchmark decode is 2-4 tokens; the anomaly
  score is computed from class likelihoods, not from generated text.
"""
from __future__ import annotations

from typing import Dict, List

PRODUCT_NAMES = {
    "breakfast_box": "breakfast box",
    "juice_bottle": "juice bottle",
    "pushpins": "box of pushpins",
    "screw_bag": "bag of screws and washers",
    "splicing_connectors": "set of splicing connectors with cable",
}
CLASS_TOKENS = {"NORMAL": "NORMAL", "ANOMALOUS": "ANOMALOUS"}

_ANOMALY_DEFINITION = (
    "An anomaly may be structural (damage, deformation, contamination or another physical defect) "
    "or logical (a component that is missing, extra, misplaced, of the wrong type or wrong quantity, "
    "or incorrectly arranged)."
)


def _image_block(k: int) -> List[Dict]:
    content: List[Dict] = []
    for i in range(k):
        content += [{"type": "text", "text": f"Reference {i + 1} (known-good):"}, {"type": "image"}]
    content += [{"type": "text", "text": "Query image (the product to inspect):"}, {"type": "image"}]
    return content


def classification_content(category: str, k: int) -> List[Dict]:
    """Chat content for Stage A (k reference placeholders, then the query placeholder)."""
    product = PRODUCT_NAMES.get(category, category.replace("_", " "))
    intro = (f"You are performing industrial visual inspection of a {product}. "
             f"The first {k} image{'s show' if k > 1 else ' shows'} correctly manufactured product"
             f"{'s' if k > 1 else ''} (known-good references). The final image is the product to inspect.")
    ask = ("Compare the query image against the known-good references and decide whether the query product "
           f"is NORMAL or ANOMALOUS. {_ANOMALY_DEFINITION} "
           "Answer with exactly one word: NORMAL or ANOMALOUS.")
    return [{"type": "text", "text": intro}] + _image_block(k) + [{"type": "text", "text": ask}]


def explanation_content(category: str, k: int) -> List[Dict]:
    """Chat content for Stage B (only run for queries classified ANOMALOUS)."""
    product = PRODUCT_NAMES.get(category, category.replace("_", " "))
    intro = (f"You are performing industrial visual inspection of a {product}. "
             f"The first {k} image{'s show' if k > 1 else ' shows'} known-good references; the final image "
             "was flagged as ANOMALOUS.")
    ask = (f"{_ANOMALY_DEFINITION} Describe the anomaly in the query image. Respond with only a JSON object with "
           'keys "status" (always "ANOMALOUS"), "anomaly_type" ("structural" or "logical"), "issue" (at most 6 '
           'words) and "explanation" (one sentence comparing the query with the references).')
    return [{"type": "text", "text": intro}] + _image_block(k) + [{"type": "text", "text": ask}]


def render_text(content: List[Dict]) -> str:
    """Human-readable prompt (image placeholders shown as <image>) for logs/manifests."""
    return "\n".join("<image>" if c["type"] == "image" else c["text"] for c in content)
