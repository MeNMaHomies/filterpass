"""
Codec saturation / compression artefact simulation.
Applies soft-clipping via a tanh curve above a configurable threshold,
approximating the non-linear distortion introduced by lossy codecs (MP3, AAC, G.711).
"""

import numpy as np

from .config import AugmentationConfig


def augment(audio: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.apply_codec_distortion:
        return audio

    thr = cfg.codec_clip_threshold
    audio = np.where(
        np.abs(audio) <= thr,
        audio,
        np.sign(audio) * (thr + (1.0 - thr) * np.tanh((np.abs(audio) - thr) / (1.0 - thr + 1e-9))),
    )
    return audio.astype(np.float32)
