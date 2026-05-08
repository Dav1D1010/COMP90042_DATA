"""
Evaluation metrics for the Climatron fact-checking pipeline.

Design rationale
================

Why Evidence Retrieval F1 (per-claim, macro-averaged)?
    The standard FEVER evaluation protocol measures whether the system retrieves
    the correct evidence sentences for each claim.  Per-claim F1 is computed as:
        precision = |pred_ev ∩ gt_ev| / |pred_ev|
        recall    = |pred_ev ∩ gt_ev| / |gt_ev|
        F1        = 2 * P * R / (P + R)
    then averaged across all claims (macro-average).  This penalises both
    over-retrieval (low precision) and under-retrieval (low recall) and is the
    metric used by the assignment's eval.py script.

Why Harmonic Mean of retrieval F1 and classification accuracy?
    The harmonic mean H = 2 * (F1 * Acc) / (F1 + Acc) is a single-number
    summary that penalises systems which excel at one task but fail at the
    other.  A model that retrieves perfectly but classifies randomly scores
    H ≈ 2 * (1.0 * 0.25) / 1.25 = 0.40, while a balanced model with 0.60
    on both scores H = 0.60.  This follows the assignment's evaluation
    framework which combines retrieval and classification quality into one
    ranking metric.

Why per-class breakdowns?
    The four-class label distribution is imbalanced (SUPPORTS ~42%, DISPUTED
    ~10%).  Overall accuracy hides per-class failures — a model could score
    60% accuracy while completely missing the DISPUTED class.  Per-class
    accuracy and F1 expose this, enabling targeted mitigation (LDAM margins,
    CB loss re-weighting).

"""

from __future__ import annotations

from typing import Dict, List


def compute_metrics(
    predictions: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
) -> Dict:
    """Compute retrieval F1, classification accuracy, and harmonic mean.

    Parameters
    ----------
    predictions : dict[str, dict]
        {claim_id: {"claim_label": str, "evidences": [str, ...]}}.
        ``evidences`` is a list of evidence ID strings predicted by the
        retrieval pipeline.
    ground_truth : dict[str, dict]
        Same structure as *predictions*, containing the gold-standard labels
        and evidence sets.

    Returns
    -------
    dict
        Keys:
        - ``"retrieval_f1"`` (float): Macro-averaged per-claim evidence F1.
        - ``"classification_acc"`` (float): Overall claim classification accuracy.
        - ``"harmonic_mean"`` (float): Harmonic mean of the two scores above.
        - ``"per_class_accuracy"`` (dict[str, float]):
            For each label string, the fraction of ground-truth claims with
            that label that were correctly classified.
        - ``"per_class_f1"`` (dict[str, float]):
            For each label string, the average retrieval F1 over claims with
            that ground-truth label.
    """
    if not predictions or not ground_truth:
        raise ValueError("predictions and ground_truth must be non-empty dicts")

    # ── 1. Evidence Retrieval F1 ──────────────────────────────────────
    f1_scores: List[float] = []
    per_class_f1: Dict[str, List[float]] = {}  # label → list of per-claim F1s

    # ── 2. Claim Classification Accuracy ───────────────────────────────
    correct = 0
    total = 0
    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}

    for claim_id, gt in ground_truth.items():
        pred = predictions.get(claim_id)
        gt_label = gt.get("claim_label", "")
        gt_ev = set(gt.get("evidences", []))

        # --- Retrieval F1 ---
        if pred is not None and "evidences" in pred:
            pred_ev = set(pred["evidences"])
            if len(pred_ev) > 0 and len(gt_ev) > 0:
                precision = len(pred_ev & gt_ev) / len(pred_ev)
                recall = len(pred_ev & gt_ev) / len(gt_ev)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
        else:
            f1 = 0.0
        f1_scores.append(f1)

        # Track per-class F1
        per_class_f1.setdefault(gt_label, []).append(f1)

        # --- Classification ---
        pred_label = pred.get("claim_label", "") if pred else ""
        if pred_label == gt_label:
            correct += 1
            per_class_correct[gt_label] = per_class_correct.get(gt_label, 0) + 1
        per_class_total[gt_label] = per_class_total.get(gt_label, 0) + 1
        total += 1

    retrieval_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    classification_acc = correct / total if total > 0 else 0.0

    # Harmonic mean: 2 * (F1 * Acc) / (F1 + Acc), 0 if denominator is 0
    if retrieval_f1 + classification_acc > 0:
        harmonic_mean = (
            2 * retrieval_f1 * classification_acc / (retrieval_f1 + classification_acc)
        )
    else:
        harmonic_mean = 0.0

    # ── Per-class summaries ──────────────────────────────────────────
    per_class_accuracy = {
        label: per_class_correct.get(label, 0) / per_class_total[label]
        for label in per_class_total
    }

    per_class_f1_mean = {
        label: sum(vals) / len(vals) if vals else 0.0
        for label, vals in per_class_f1.items()
    }

    return {
        "retrieval_f1": retrieval_f1,
        "classification_acc": classification_acc,
        "harmonic_mean": harmonic_mean,
        "per_class_accuracy": per_class_accuracy,
        "per_class_f1": per_class_f1_mean,
    }
