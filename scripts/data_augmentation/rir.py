"""
Room impulse response simulation via pyroomacoustics.
Simulates random shoe-box rooms to introduce realistic reverberation.
"""

import random

import numpy as np

from .config import AugmentationConfig

try:
    import pyroomacoustics as pra
    HAS_PRA = True
except ImportError:
    HAS_PRA = False
    print("[warn] pyroomacoustics not installed – RIR simulation disabled.")


def augment(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if not cfg.simulate_rir or not HAS_PRA:
        return audio

    dim = [random.uniform(*cfg.room_dim_range) for _ in range(3)]
    rt60 = random.uniform(*cfg.rt60_range)

    try:
        e_absorption, max_order = pra.inverse_sabine(rt60, dim)
    except Exception:
        return audio  # room geometry incompatible with requested RT60

    room = pra.ShoeBox(dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    src = [d * random.uniform(0.2, 0.8) for d in dim]
    mic = [d * random.uniform(0.2, 0.8) for d in dim]
    room.add_source(src, signal=audio)
    room.add_microphone(np.array(mic).reshape(3, 1))
    room.simulate()

    out = room.mic_array.signals[0].astype(np.float32)
    if len(out) > len(audio):
        out = out[: len(audio)]
    else:
        out = np.pad(out, (0, len(audio) - len(out)))

    return np.clip(out, -1.0, 1.0)
