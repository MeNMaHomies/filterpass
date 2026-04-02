"""
VoIP packet-loss simulation.
Splits audio into 20 ms frames (matching RTP packet duration) and randomly
drops a configurable fraction of frames, filling the gap with silence,
a repeat of the previous frame, or low-level noise.
"""

import random

import numpy as np

from .config import AugmentationConfig
from .utils import match_length

_FRAME_DURATION_S = 0.020  # 20 ms — standard RTP packet size


def augment(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.simulate_packet_loss:
        return audio

    frame_samples = int(_FRAME_DURATION_S * sr)
    out = audio.copy()

    for start in range(0, len(audio), frame_samples):
        if random.random() >= cfg.packet_loss_rate:
            continue
        end = min(start + frame_samples, len(audio))
        size = end - start
        if cfg.packet_loss_fill == "silence":
            out[start:end] = 0.0
        elif cfg.packet_loss_fill == "repeat" and start > 0:
            prev = out[max(0, start - frame_samples):start]
            out[start:end] = match_length(prev, size)
        else:
            out[start:end] = np.random.randn(size).astype(np.float32) * 0.01

    return out
