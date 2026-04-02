"""
Random gain augmentation.
Applies a uniformly sampled dB gain shift to simulate level variation
across different recording environments and devices.
"""

import random

import numpy as np

from .config import AugmentationConfig


def augment(audio: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.apply_random_gain:
        return audio

    gain_linear = 10 ** (random.uniform(*cfg.gain_db_range) / 20.0)
    return np.clip(audio * gain_linear, -1.0, 1.0)
