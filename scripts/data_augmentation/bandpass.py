"""
Telephone / narrowband codec bandwidth simulation.
A 4th-order Butterworth bandpass filter limits the frequency range to 300–3400 Hz,
matching the response of G.711 and similar telephony codecs.
"""

import numpy as np
import scipy.signal as sps

from .config import AugmentationConfig


def augment(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.apply_bandpass:
        return audio

    nyq = sr / 2.0
    lo = np.clip(cfg.bandpass_low_hz / nyq, 1e-4, 0.9999)
    hi = np.clip(cfg.bandpass_high_hz / nyq, lo + 1e-4, 0.9999)
    b, a = sps.butter(4, [lo, hi], btype="band")
    return sps.lfilter(b, a, audio).astype(np.float32)
