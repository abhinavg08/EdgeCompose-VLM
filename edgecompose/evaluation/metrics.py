"""Benchmark metrics.

TextVQA: the standard VQA accuracy used by the official TextVQA/EvalAI evaluator (and by
lmms-eval): answers are normalised (punctuation, digits, articles, contractions), then

    acc(pred) = mean over the 10 leave-one-out subsets of 9 human answers of
                min(1, #matches(pred, subset) / 3)

POPE: binary yes/no classification with "yes" as the positive class; reports accuracy,
precision, recall, F1 and the yes-ratio (fraction of "yes" predictions; a value well
above the 0.5 base rate indicates a yes-bias, i.e. object hallucination).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

# --------------------------------------------------------------------------- TextVQA
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in _PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_COMMA_STRIP, in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _process_digit_article(in_text: str) -> str:
    out = []
    for word in in_text.lower().split():
        word = _NUMBER_MAP.setdefault(word, word)
        if word not in _ARTICLES:
            out.append(word)
    out = [_CONTRACTIONS.get(w, w) for w in out]
    return " ".join(out)


def normalize_vqa_answer(ans: str) -> str:
    """EvalAI answer processor (as in the official VQA/TextVQA evaluation)."""
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = _process_punctuation(ans)
    ans = _process_digit_article(ans)
    return ans


def vqa_accuracy(prediction: str, gt_answers: Sequence[str]) -> float:
    """Standard VQA accuracy of one prediction against (usually 10) human answers."""
    if not gt_answers:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    gts = [normalize_vqa_answer(a) for a in gt_answers]
    if len(gts) == 1:
        return float(pred == gts[0])
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        matches = sum(1 for a in others if a == pred)
        accs.append(min(1.0, matches / 3.0))
    return sum(accs) / len(accs)


# --------------------------------------------------------------------------- POPE
def parse_yes_no(prediction: str) -> str:
    """POPE answer parsing (as in the POPE reference evaluation / lmms-eval).

    Only the first sentence is kept, commas are removed, and the answer is "no" if the
    words contain 'no'/'No'/'not'; otherwise "yes" (the original POPE rule, case-folded).
    """
    text = prediction.strip().lower()
    first = re.split(r"[.\n]", text, maxsplit=1)[0].replace(",", "")
    words = re.findall(r"[a-z']+", first)
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def pope_metrics(preds: Iterable[str], labels: Iterable[str]) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 / yes-ratio for parsed yes/no predictions."""
    p = [x.lower() for x in preds]
    y = [x.lower() for x in labels]
    if len(p) != len(y):
        raise ValueError("preds and labels differ in length")
    n = len(p)
    if n == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "yes_ratio": float("nan"), "n": 0}
    tp = sum(1 for a, b in zip(p, y) if a == "yes" and b == "yes")
    fp = sum(1 for a, b in zip(p, y) if a == "yes" and b == "no")
    fn = sum(1 for a, b in zip(p, y) if a == "no" and b == "yes")
    tn = sum(1 for a, b in zip(p, y) if a == "no" and b == "no")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / n,
        "n": n,
    }


def score_sample(dataset: str, prediction: str, answers: List[str]) -> float:
    """Per-sample score used in the raw logs (VQA accuracy, or POPE correctness)."""
    if dataset == "textvqa":
        return vqa_accuracy(prediction, answers)
    if dataset == "pope":
        return float(parse_yes_no(prediction) == answers[0].strip().lower())
    raise ValueError(f"unknown dataset {dataset}")
