"""Entity-level evaluation of PII detection.

Two decisions in this file are worth defending in a review.

1. Evaluation is ENTITY-LEVEL, not token-level. A PII document is >99%
   non-PII text, so a model that predicts "nothing is PII" scores about 99% on
   token accuracy while leaking every single identifier. Token accuracy is
   therefore actively misleading here, and this file never reports it.

2. "Accuracy" is defined explicitly, because the assignment asks for it but the
   term has no standard meaning without true negatives. We report

       accuracy = TP / (TP + FP + FN)

   which is the Jaccard index, sometimes called entity-level accuracy. It is
   the fraction of all entities involved in the comparison that were handled
   correctly, and unlike (TP+TN)/total it cannot be inflated by the vast
   quantity of correctly-ignored text. The README states this definition
   alongside the numbers.

Matching is on (normalised text, type). Comparing by character offset would
punish a detection that is correct but one space wider than the annotation,
which measures annotation nitpicking rather than detection quality.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class CategoryMetrics:
    """Precision / recall / F1 / accuracy for one PII type."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        """Jaccard index: TP / (TP + FP + FN). See module docstring."""
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def support(self) -> int:
        """Number of ground-truth entities of this type."""
        return self.tp + self.fn


@dataclass
class EvaluationReport:
    overall: CategoryMetrics = field(default_factory=CategoryMetrics)
    by_type: Dict[str, CategoryMetrics] = field(default_factory=dict)
    false_positives: List[Tuple[str, str]] = field(default_factory=list)
    false_negatives: List[Tuple[str, str]] = field(default_factory=list)
    corpus: str = ""


def normalise(text: str) -> str:
    """Canonical form for comparison.

    Case, whitespace and dash style are all collapsed. Dash normalisation
    matters more than it looks: the document uses an en-dash in "Pune – 411
    044" while a human annotator types a hyphen. Without this, the same address
    is scored as a false positive AND a false negative simultaneously, which
    measures typography rather than detection.
    """
    cleaned = text.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".,;:()[]\"'")
    return cleaned.lower()


# Types whose entities are routinely split across several paragraphs, so many
# predictions may legitimately correspond to one annotation.
FRAGMENT_TYPES = {"ADDRESS"}


def _digits(text: str) -> str:
    return re.sub(r"[^\d]", "", text)


def matches(pred_text: str, gt_text: str, etype: str) -> bool:
    """Decide whether a prediction and an annotation refer to the same entity.

    Comparison is type-aware because "the same entity" means different things
    per category:

    * PHONE  - separators are cosmetic, so only the digits are compared.
               "+91 20 4505 3237" and "+91 20 45053237" are one number.
    * ADDRESS - containment either way counts as a match. A postal address is
               often typed across several paragraphs, so the tool reports it as
               fragments while an annotator writes it as one string. Demanding
               an exact match would score correct detections as both a miss and
               a false alarm, measuring paragraph layout rather than detection.
    * everything else - exact match after normalisation.
    """
    p, g = normalise(pred_text), normalise(gt_text)

    if etype == "PHONE":
        return _digits(p) == _digits(g) and len(_digits(p)) >= 8
    if etype == "ADDRESS":
        return p in g or g in p
    return p == g


def evaluate(
    predicted: List[Tuple[str, str]],
    ground_truth: List[Tuple[str, str]],
    corpus: str = "",
) -> EvaluationReport:
    """Compare predictions against ground truth by greedy one-to-one matching.

    Both sides are de-duplicated first. The unit of evaluation is the unique
    entity, not the occurrence: "did the tool find Rajesh Kushal Hegde?" is the
    meaningful question, and counting his 40 mentions separately would let one
    frequently-repeated name dominate the score.

    Matching is greedy and one-to-one, so a single prediction can satisfy at
    most one annotation. That prevents one over-long address span from
    "covering" several annotations and inflating recall.
    """
    pred_unique = _dedupe(predicted)
    gt_unique = _dedupe(ground_truth)

    unmatched_preds = list(pred_unique)
    tp: List[Tuple[str, str]] = []
    fn: List[Tuple[str, str]] = []

    for gt_text, gt_type in gt_unique:
        hits = [
            p for p in unmatched_preds
            if p[1] == gt_type and matches(p[0], gt_text, gt_type)
        ]
        if not hits:
            fn.append((gt_text, gt_type))
            continue

        if gt_type in FRAGMENT_TYPES:
            # A multi-paragraph address is reported as several fragments. All
            # of them describe the one annotated address, so they are consumed
            # together and counted as a single true positive. Counting the
            # extra fragments as false positives would penalise the tool for
            # correctly following the document's own line breaks.
            for hit in hits:
                unmatched_preds.remove(hit)
            tp.append(hits[0])
        else:
            unmatched_preds.remove(hits[0])
            tp.append(hits[0])

    fp = unmatched_preds

    report = EvaluationReport(corpus=corpus)
    report.overall = CategoryMetrics(tp=len(tp), fp=len(fp), fn=len(fn))

    per_type: Dict[str, CategoryMetrics] = defaultdict(CategoryMetrics)
    for _, etype in tp:
        per_type[etype].tp += 1
    for _, etype in fp:
        per_type[etype].fp += 1
    for _, etype in fn:
        per_type[etype].fn += 1

    report.by_type = dict(sorted(per_type.items()))
    report.false_positives = sorted(fp)
    report.false_negatives = sorted(fn)
    return report


def _dedupe(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drop duplicates while preserving order.

    De-duplication is type-aware for the same reason matching is. A phone
    number written "+91 22 40094400" in one place and "+ 91 22 4009 4400" in
    another is one number; treating them as two predictions means the second
    one has no annotation left to match and is wrongly counted as a false
    positive.
    """
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for text, etype in items:
        key = (_digits(text) if etype == "PHONE" else normalise(text), etype)
        if key not in seen:
            seen.add(key)
            out.append((text, etype))
    return out


def load_ground_truth(path: str) -> List[Tuple[str, str]]:
    """Load a ground-truth annotation file: [{"text": ..., "type": ...}, ...]."""
    with open(path, encoding="utf-8") as fh:
        return [(item["text"], item["type"]) for item in json.load(fh)]


def format_report(report: EvaluationReport) -> str:
    """Render the report as a Markdown table."""
    lines: List[str] = []
    if report.corpus:
        lines.append(f"### Corpus: {report.corpus}\n")

    lines.append("| PII Type | Precision | Recall | F1 | Accuracy | TP | FP | FN | Support |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for etype, m in report.by_type.items():
        lines.append(
            f"| {etype} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | "
            f"{m.accuracy:.2f} | {m.tp} | {m.fp} | {m.fn} | {m.support} |"
        )

    o = report.overall
    lines.append(
        f"| **OVERALL** | **{o.precision:.2f}** | **{o.recall:.2f}** | "
        f"**{o.f1:.2f}** | **{o.accuracy:.2f}** | {o.tp} | {o.fp} | {o.fn} | {o.support} |"
    )
    return "\n".join(lines)
