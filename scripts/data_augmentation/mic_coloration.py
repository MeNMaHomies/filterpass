"""
Microphone coloration simulation.
Applies a low-shelf boost around a configurable frequency to mimic the resonance
peak common in cheap or budget microphone capsules.
"""

import numpy as np
import scipy.signal as sps

from .config import AugmentationConfig


def augment(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.apply_mic_coloration:
        return audio

    freq = np.clip(cfg.mic_coloration_freq_hz / (sr / 2.0), 1e-4, 0.9999)
    gain_linear = 10 ** (cfg.mic_coloration_gain_db / 20.0)
    b, a = sps.butter(2, freq, btype="low")
    low = sps.lfilter(b, a, audio).astype(np.float32)
    return np.clip(audio + (gain_linear - 1.0) * low, -1.0, 1.0)
