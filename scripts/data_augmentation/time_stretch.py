"""
Time-stretch augmentation via linear resampling.
Speeds up or slows down audio without pitch correction, then trims or pads
back to the original length to keep a consistent chunk shape.
"""

import random

import numpy as np
import scipy.signal as sps

from .config import AugmentationConfig


def augment(audio: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.apply_time_stretch:
        return audio

    rate = random.uniform(*cfg.time_stretch_range)
    orig_len = len(audio)
    stretched = sps.resample(audio, int(orig_len / rate)).astype(np.float32)

    if len(stretched) > orig_len:
        return stretched[:orig_len]
    return np.pad(stretched, (0, orig_len - len(stretched)))
