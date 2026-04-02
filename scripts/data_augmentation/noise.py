"""
Noise-based augmentation: white, pink, brown, and environmental background noise.
All variants are mixed into the signal at a randomly sampled SNR.
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal as sps

from .config import AugmentationConfig
from .utils import mix_at_snr, match_length


def _white(n: int) -> np.ndarray:
    return np.random.randn(n).astype(np.float32)


def _pink(n: int) -> np.ndarray:
    """Voss-McCartney pink noise approximation."""
    cols = 16
    array = np.full((n, cols), np.nan)
    array[0] = np.random.randn(cols)
    for i in range(1, n):
        j = int(np.log2(i & -i)) if i & (i - 1) else 0
        j = min(j, cols - 1)
        array[i] = array[i - 1]
        array[i, j] = np.random.randn()
    pink = np.nansum(array, axis=1).astype(np.float32)
    return pink / (pink.std() + 1e-9)


def _brown(n: int) -> np.ndarray:
    """Brown/red noise: cumulative sum of white noise."""
    white = np.random.randn(n).astype(np.float32)
    brown = np.cumsum(white)
    brown -= brown.mean()
    return (brown / (brown.std() + 1e-9)).astype(np.float32)


def augment(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    n = len(audio)

    if cfg.add_white_noise:
        audio = mix_at_snr(audio, _white(n), random.uniform(*cfg.white_noise_snr_db))

    if cfg.add_pink_noise:
        audio = mix_at_snr(audio, _pink(n), random.uniform(*cfg.pink_noise_snr_db))

    if cfg.add_brown_noise:
        audio = mix_at_snr(audio, _brown(n), random.uniform(*cfg.brown_noise_snr_db))

    if cfg.background_noise_dir:
        noise_files = (
            list(Path(cfg.background_noise_dir).glob("**/*.flac"))
            + list(Path(cfg.background_noise_dir).glob("**/*.wav"))
        )
        if noise_files:
            bg, bg_sr = sf.read(str(random.choice(noise_files)), dtype="float32", always_2d=False)
            if bg.ndim > 1:
                bg = bg.mean(axis=1)
            if bg_sr != sr:
                bg = sps.resample(bg, int(len(bg) * sr / bg_sr)).astype(np.float32)
            audio = mix_at_snr(audio, match_length(bg, n), random.uniform(*cfg.background_noise_snr_db))

    return audio
