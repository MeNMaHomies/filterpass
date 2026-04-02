"""
Abstract contract every model adapter must implement.

To add a new model:
  1. Create scripts/benchmarking/models/<your_model>.py
  2. Subclass BenchmarkModel and implement all abstract methods
  3. Register it in scripts/benchmarking/models/__init__.py
"""

from abc import ABC, abstractmethod

import torch


class BenchmarkModel(ABC):
    """Stateless interface for a deepfake-detection model under benchmark."""

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in summary reports."""

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """
        Download weights (if needed) and move the model to `device`.
        Called once before inference begins.
        """

    # ── inference ─────────────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, chunks: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of audio chunks.

        Args:
            chunks: Float32 tensor of shape (N, CHUNK_SAMPLES) already on device.

        Returns:
            1-D float32 tensor of shape (N,) — higher score = more likely bonafide.
        """

    # ── optional metadata ─────────────────────────────────────────────────────

    def parameter_count(self) -> int:
        """Return total trainable parameter count (override if not a nn.Module)."""
        return 0
