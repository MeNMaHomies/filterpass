"""
ASVspoof tandem detection cost function (t-DCF) and supporting utilities.

Ported from the official ASVspoof 2019/2021 evaluation toolkit so the
benchmarking pipeline has no dependency on the XLSR-Mamba (or any other)
external repository.

Reference:
    T. Kinnunen et al., "Tandem Assessment of Spoofing Countermeasures and
    Automatic Speaker Verification: Fundamentals," IEEE/ACM TASLP.
"""

from __future__ import annotations

import sys

import numpy as np


def obtain_asv_error_rates(
    tar_asv: np.ndarray,
    non_asv: np.ndarray,
    spoof_asv: np.ndarray,
    asv_threshold: float,
) -> tuple[float, float, float | None, float | None]:
    Pfa_asv = np.sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = np.sum(tar_asv < asv_threshold) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[float, float]:
    """Returns (EER, threshold). EER is in [0, 1], not percentage."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


def compute_tDCF(
    bonafide_score_cm: np.ndarray,
    spoof_score_cm: np.ndarray,
    Pfa_asv: float,
    Pmiss_asv: float,
    Pfa_spoof_asv: float,
    cost_model: dict,
    print_cost: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cost_model["Cfa"] < 0 or cost_model["Cmiss"] < 0:
        print("WARNING: Usually the cost values should be positive!")

    if (
        cost_model["Ptar"] < 0
        or cost_model["Pnon"] < 0
        or cost_model["Pspoof"] < 0
        or abs(cost_model["Ptar"] + cost_model["Pnon"] + cost_model["Pspoof"] - 1)
        > 1e-10
    ):
        sys.exit("ERROR: Prior probabilities must be positive and sum to one.")

    if Pfa_spoof_asv is None:
        sys.exit("ERROR: Pfa_spoof_asv must be provided.")

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit("ERROR: Scores contain NaN or Inf.")

    if np.unique(combined_scores).size < 3:
        sys.exit("ERROR: Provide soft CM scores, not binary decisions.")

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    C0 = (
        cost_model["Ptar"] * cost_model["Cmiss"] * Pmiss_asv
        + cost_model["Pnon"] * cost_model["Cfa"] * Pfa_asv
    )
    C1 = cost_model["Ptar"] * cost_model["Cmiss"] - C0
    C2 = cost_model["Pspoof"] * cost_model["Cfa_spoof"] * Pfa_spoof_asv

    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit("ERROR: Negative tDCF weights — check ASV error rates.")

    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_default = C0 + np.minimum(C1, C2)
    tDCF_norm = tDCF / tDCF_default

    return tDCF_norm, CM_thresholds
