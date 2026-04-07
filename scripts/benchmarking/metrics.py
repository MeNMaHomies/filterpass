"""
Metric computation for anti-spoofing benchmarks.

All functions are pure (no I/O), model-agnostic, and dataset-agnostic.
EER delegates to eval_metrics.compute_eer (official ASVspoof DET-curve method).
Dataset-specific metrics (e.g. min-tDCF) are handled by DatasetAdapter.extra_metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base.dataset_base import Trial
from .config import CHUNK_DURATION_S
from .eval_metrics import compute_eer

# ── Detection metrics ──────────────────────────────────────────────────────────


def compute_detection_metrics(
    all_scores: dict[str, float],
    trials: list[Trial],
) -> dict:
    """
    Compute EER, AUC-ROC, FAR/FRR, classification metrics,
    and per-condition EER.

    `trials` is dataset-agnostic — any DatasetAdapter.load_trials() output works.
    Dataset-specific metrics (min-tDCF) are merged in by __main__ via extra_metrics().
    """
    utt_to_label = {t.utt_id: t.label for t in trials}
    utt_to_condition = {t.utt_id: t.condition for t in trials}

    scored_ids = list(all_scores.keys())
    scores_arr = np.array([all_scores[u] for u in scored_ids])
    labels_arr = np.array([utt_to_label[u] for u in scored_ids])

    bona_scores = scores_arr[labels_arr == "bonafide"]
    spoof_scores = scores_arr[labels_arr == "spoof"]

    eer, eer_threshold = compute_eer(bona_scores, spoof_scores)

    binary_labels = (labels_arr == "bonafide").astype(int)
    preds = (scores_arr >= eer_threshold).astype(int)

    results = {
        "eer_pct": 100 * eer,
        "eer_threshold": eer_threshold,
        "auc_roc": roc_auc_score(binary_labels, scores_arr),
        "far": float(
            np.sum((preds == 1) & (binary_labels == 0))
            / max(np.sum(binary_labels == 0), 1)
        ),
        "frr": float(
            np.sum((preds == 0) & (binary_labels == 1))
            / max(np.sum(binary_labels == 1), 1)
        ),
        "accuracy": accuracy_score(binary_labels, preds),
        "precision": precision_score(binary_labels, preds, zero_division=0),
        "recall": recall_score(binary_labels, preds, zero_division=0),
        "f1": f1_score(binary_labels, preds, zero_division=0),
    }

    conditions_arr = np.array([utt_to_condition[u] for u in scored_ids])
    condition_eers = {}
    for cond in sorted(set(conditions_arr)):
        mask = conditions_arr == cond
        c_bona = scores_arr[mask & (labels_arr == "bonafide")]
        c_spoof = scores_arr[mask & (labels_arr == "spoof")]
        if len(c_bona) > 0 and len(c_spoof) > 0:
            c_eer, _ = compute_eer(c_bona, c_spoof)
            condition_eers[cond] = 100 * c_eer
    results["condition_eers"] = condition_eers

    return results


# ── RTF / latency metrics ──────────────────────────────────────────────────────


def compute_rtf_stats(rtf_values: list[float]) -> dict:
    arr = np.array(rtf_values)
    return {
        "rtf_mean": float(arr.mean()),
        "rtf_median": float(np.median(arr)),
        "rtf_p95": float(np.percentile(arr, 95)),
        "rtf_max": float(arr.max()),
        "latency_mean_ms": float(arr.mean() * CHUNK_DURATION_S * 1000),
        "latency_median_ms": float(np.median(arr) * CHUNK_DURATION_S * 1000),
        "latency_p95_ms": float(np.percentile(arr, 95) * CHUNK_DURATION_S * 1000),
    }
