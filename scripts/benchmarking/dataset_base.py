"""
Abstract contract every dataset adapter must implement.

To add a new dataset:
  1. Create scripts/benchmarking/datasets/<your_dataset>.py
  2. Subclass DatasetAdapter and implement all abstract methods
  3. Register it in scripts/benchmarking/datasets/__init__.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Trial:
    utt_id: str
    label: str       # "bonafide" | "spoof"
    condition: str   # grouping label for per-condition EER (codec, attack type, etc.)


class DatasetAdapter(ABC):

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name used in summary reports."""

    # ── data access ───────────────────────────────────────────────────────────

    @abstractmethod
    def load_trials(self, phase: str) -> list[Trial]:
        """
        Return all evaluation trials for `phase`.
        Each Trial carries a utt_id, a binary label, and a condition tag.
        """

    @abstractmethod
    def audio_path(self, utt_id: str) -> str:
        """Return the absolute path to the audio file for a given utterance ID."""

    # ── optional dataset-specific metrics ─────────────────────────────────────

    def extra_metrics(
        self,
        scores: dict[str, float],
        trials: list[Trial],
        phase: str,
    ) -> dict:
        """
        Compute dataset-specific metrics (e.g. min-tDCF for ASVspoof).
        Return an empty dict if the dataset does not define additional metrics.
        Results are merged into the detection metrics dict before reporting.
        """
        return {}
