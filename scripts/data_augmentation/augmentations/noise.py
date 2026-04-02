"""
Noise-based augmentations.

Each class is a standalone callable that mixes a noise signal into the audio
at a randomly sampled SNR.
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal as sps

from ..base import Augmentation
from ..utils import mix_at_snr, match_length


class WhiteNoise(Augmentation):
    def __init__(self, snr_db: tuple[float, float] = (10.0, 30.0), p: float = 0.5):
        super().__init__(p=p)
        self.snr_db = snr_db

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        noise = np.random.randn(len(audio)).astype(np.float32)
        return mix_at_snr(audio, noise, random.uniform(*self.snr_db))


class PinkNoise(Augmentation):
    """Voss-McCartney pink (1/f) noise approximation."""

    def __init__(self, snr_db: tuple[float, float] = (10.0, 25.0), p: float = 0.5):
        super().__init__(p=p)
        self.snr_db = snr_db

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        n = len(audio)
        cols = 16
        array = np.full((n, cols), np.nan)
        array[0] = np.random.randn(cols)
        for i in range(1, n):
            j = int(np.log2(i & -i)) if i & (i - 1) else 0
            j = min(j, cols - 1)
            array[i] = array[i - 1]
            array[i, j] = np.random.randn()
        pink = np.nansum(array, axis=1).astype(np.float32)
        pink = pink / (pink.std() + 1e-9)
        return mix_at_snr(audio, pink, random.uniform(*self.snr_db))


class BrownNoise(Augmentation):
    """Brown/red noise: cumulative sum of white noise (~1/f^2 spectrum)."""

    def __init__(self, snr_db: tuple[float, float] = (10.0, 20.0), p: float = 0.5):
        super().__init__(p=p)
        self.snr_db = snr_db

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        white = np.random.randn(len(audio)).astype(np.float32)
        brown = np.cumsum(white)
        brown -= brown.mean()
        brown = (brown / (brown.std() + 1e-9)).astype(np.float32)
        return mix_at_snr(audio, brown, random.uniform(*self.snr_db))


class BackgroundNoise(Augmentation):
    """Mix in a random real-world noise file from a directory."""

    def __init__(
        self,
        noise_dir: str,
        snr_db: tuple[float, float] = (5.0, 20.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.snr_db = snr_db
        self._noise_files = (
            list(Path(noise_dir).glob("**/*.flac"))
            + list(Path(noise_dir).glob("**/*.wav"))
        )

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not self._noise_files:
            return audio

        bg, bg_sr = sf.read(
            str(random.choice(self._noise_files)), dtype="float32", always_2d=False
        )
        if bg.ndim > 1:
            bg = bg.mean(axis=1)
        if bg_sr != sr:
            bg = sps.resample(bg, int(len(bg) * sr / bg_sr)).astype(np.float32)
        bg = match_length(bg, len(audio))
        return mix_at_snr(audio, bg, random.uniform(*self.snr_db))
