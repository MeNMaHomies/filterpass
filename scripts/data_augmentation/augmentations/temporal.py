"""
Time-domain augmentations.

Simulate VoIP artefacts, playback speed variation, and recording level differences.
"""

import random

import numpy as np
import scipy.signal as sps

from ..base import Augmentation
from ..utils import match_length

_RTP_FRAME_DURATION_S = 0.020  # 20 ms standard RTP packet


class PacketLoss(Augmentation):
    """
    VoIP-style packet loss: split into 20 ms RTP frames and randomly drop a
    fraction, filling with silence, repeat of previous frame, or low-level noise.
    """

    def __init__(
        self,
        rate: float = 0.05,
        fill: str = "silence",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.rate = rate
        self.fill = fill  # "silence" | "repeat" | "noise"

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        frame_samples = int(_RTP_FRAME_DURATION_S * sr)
        out = audio.copy()

        for start in range(0, len(audio), frame_samples):
            if random.random() >= self.rate:
                continue
            end = min(start + frame_samples, len(audio))
            size = end - start
            if self.fill == "silence":
                out[start:end] = 0.0
            elif self.fill == "repeat" and start > 0:
                prev = out[max(0, start - frame_samples) : start]
                out[start:end] = match_length(prev, size)
            else:
                out[start:end] = np.random.randn(size).astype(np.float32) * 0.01

        return out


class TimeStretch(Augmentation):
    """Speed up or slow down via linear resampling (no pitch correction)."""

    def __init__(
        self,
        rate_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.rate_range = rate_range

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        rate = random.uniform(*self.rate_range)
        orig_len = len(audio)
        stretched = sps.resample(audio, int(orig_len / rate)).astype(np.float32)
        if len(stretched) > orig_len:
            return stretched[:orig_len]
        return np.pad(stretched, (0, orig_len - len(stretched)))


class RandomGain(Augmentation):
    """Apply a uniformly sampled dB gain shift."""

    def __init__(self, db_range: tuple[float, float] = (-6.0, 6.0), p: float = 0.5):
        super().__init__(p=p)
        self.db_range = db_range

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        gain_linear = 10 ** (random.uniform(*self.db_range) / 20.0)
        return np.clip(audio * gain_linear, -1.0, 1.0)
