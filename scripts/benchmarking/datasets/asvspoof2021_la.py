"""
ASVspoof 2021 LA dataset adapter.

Reads the CM trial_metadata.txt key file and (optionally) the ASV scores
needed for min-tDCF computation.

Expected directory layout under `keys_dir`:
    CM/trial_metadata.txt
    ASV/trial_metadata.txt
    ASV/ASVTorch_Kaldi/score.txt    (optional — min-tDCF skipped if absent)

Column layout of trial_metadata.txt (space-separated, no header):
    0: speaker_id  1: utt_id  2: ?  3: ?  4: ?  5: label  6: condition  7: phase
"""

import os
import sys

import numpy as np
import pandas as pd

from ..dataset_base import DatasetAdapter, Trial


_CM_KEY  = os.path.join("CM", "trial_metadata.txt")
_ASV_KEY = os.path.join("ASV", "trial_metadata.txt")
_ASV_SCR = os.path.join("ASV", "ASVTorch_Kaldi", "score.txt")

_COL_UTT       = 1
_COL_LABEL     = 5
_COL_CONDITION = 6
_COL_PHASE     = 7
_COL_ASV_SCORE = 2


class ASVspoof2021LA(DatasetAdapter):
    def __init__(self, eval_dir: str, keys_dir: str, eval_metrics_module=None):
        """
        Args:
            eval_dir: Directory containing .flac evaluation audio files.
            keys_dir: Root of the keys directory (contains CM/ and ASV/).
            eval_metrics_module: The `eval_metrics` module from the ASVspoof/model repo,
                                 required only for min-tDCF. Pass None to skip tDCF.
        """
        self._eval_dir = eval_dir
        self._keys_dir = keys_dir
        self._em = eval_metrics_module

    @property
    def name(self) -> str:
        return "ASVspoof 2021 LA"

    def load_trials(self, phase: str) -> list[Trial]:
        cm_file = os.path.join(self._keys_dir, _CM_KEY)
        if not os.path.exists(cm_file):
            print(f"ERROR: CM keys not found: {cm_file}")
            sys.exit(1)

        meta = pd.read_csv(cm_file, sep=" ", header=None)
        phase_rows = meta[meta[_COL_PHASE] == phase]

        if len(phase_rows) == 0:
            print(f"WARNING: No entries for phase='{phase}'. Using all entries.")
            phase_rows = meta

        print(f"CM trials loaded ({phase}): {len(phase_rows)}")

        return [
            Trial(
                utt_id=str(row[_COL_UTT]),
                label=str(row[_COL_LABEL]),
                condition=str(row[_COL_CONDITION]),
            )
            for _, row in phase_rows.iterrows()
        ]

    def audio_path(self, utt_id: str) -> str:
        return os.path.join(self._eval_dir, utt_id + ".flac")

    def extra_metrics(
        self,
        scores: dict[str, float],
        trials: list[Trial],
        phase: str,
    ) -> dict:
        if self._em is None:
            return {"min_tDCF": None}

        asv_key_file = os.path.join(self._keys_dir, _ASV_KEY)
        asv_scr_file = os.path.join(self._keys_dir, _ASV_SCR)

        if not os.path.exists(asv_key_file) or not os.path.exists(asv_scr_file):
            print("WARNING: ASV keys/scores not found — skipping min-tDCF.")
            return {"min_tDCF": None}

        em = self._em
        asv_key = pd.read_csv(asv_key_file, sep=" ", header=None)
        asv_scr = pd.read_csv(asv_scr_file, sep=" ", header=None)

        phase_mask    = asv_key[_COL_PHASE] == phase
        asv_scr_phase = asv_scr[phase_mask]

        tar_asv   = asv_scr_phase[_COL_ASV_SCORE][asv_key[phase_mask][_COL_LABEL] == "target"].values
        non_asv   = asv_scr_phase[_COL_ASV_SCORE][asv_key[phase_mask][_COL_LABEL] == "nontarget"].values
        spoof_asv = asv_scr_phase[_COL_ASV_SCORE][asv_key[phase_mask][_COL_LABEL] == "spoof"].values

        _, asv_threshold = em.compute_eer(tar_asv, non_asv)
        Pfa_asv, Pmiss_asv, _, Pfa_spoof_asv = em.obtain_asv_error_rates(
            tar_asv, non_asv, spoof_asv, asv_threshold
        )

        utt_to_label = {t.utt_id: t.label for t in trials}
        scored_ids   = list(scores.keys())
        scores_arr   = np.array([scores[u] for u in scored_ids])
        labels_arr   = np.array([utt_to_label[u] for u in scored_ids])

        bona_scores  = scores_arr[labels_arr == "bonafide"]
        spoof_scores = scores_arr[labels_arr == "spoof"]

        Pspoof = 0.05
        cost_model = {
            "Pspoof":    Pspoof,
            "Ptar":      (1 - Pspoof) * 0.99,
            "Pnon":      (1 - Pspoof) * 0.01,
            "Cmiss":     1,
            "Cfa":       10,
            "Cfa_spoof": 10,
        }
        tDCF_curve, _ = em.compute_tDCF(
            bona_scores, spoof_scores,
            Pfa_asv, Pmiss_asv, Pfa_spoof_asv,
            cost_model, False,
        )
        return {"min_tDCF": float(np.min(tDCF_curve))}
