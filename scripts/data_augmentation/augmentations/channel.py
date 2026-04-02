"""
Channel and device simulation augmentations.

Simulate the signal degradation introduced by rooms, telephone codecs,
cheap microphones, and lossy compression.
"""

import random

import numpy as np
import scipy.signal as sps

from ..base import Augmentation

try:
    import pyroomacoustics as pra

    _HAS_PRA = True
except ImportError:
    _HAS_PRA = False


class RoomImpulseResponse(Augmentation):
    """Simulate a random shoe-box room via pyroomacoustics."""

    def __init__(
        self,
        room_dim_range: tuple[float, float] = (3.0, 10.0),
        rt60_range: tuple[float, float] = (0.2, 0.8),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.room_dim_range = room_dim_range
        self.rt60_range = rt60_range

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not _HAS_PRA:
            return audio

        dim = [random.uniform(*self.room_dim_range) for _ in range(3)]
        rt60 = random.uniform(*self.rt60_range)

        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, dim)
        except Exception:
            return audio

        room = pra.ShoeBox(
            dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order
        )
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


class BandpassFilter(Augmentation):
    """Telephone / narrowband codec bandwidth limit (4th-order Butterworth)."""

    def __init__(
        self,
        low_hz: float = 300.0,
        high_hz: float = 3400.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.low_hz = low_hz
        self.high_hz = high_hz

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2.0
        lo = np.clip(self.low_hz / nyq, 1e-4, 0.9999)
        hi = np.clip(self.high_hz / nyq, lo + 1e-4, 0.9999)
        b, a = sps.butter(4, [lo, hi], btype="band")
        return sps.lfilter(b, a, audio).astype(np.float32)


class MicColoration(Augmentation):
    """Low-shelf boost to mimic cheap microphone capsule resonance."""

    def __init__(
        self,
        freq_hz: float = 1500.0,
        gain_db: float = 3.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.freq_hz = freq_hz
        self.gain_db = gain_db

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        freq = np.clip(self.freq_hz / (sr / 2.0), 1e-4, 0.9999)
        gain_linear = 10 ** (self.gain_db / 20.0)
        b, a = sps.butter(2, freq, btype="low")
        low = sps.lfilter(b, a, audio).astype(np.float32)
        return np.clip(audio + (gain_linear - 1.0) * low, -1.0, 1.0)


class CodecDistortion(Augmentation):
    """Soft-clip via tanh to emulate lossy codec saturation artefacts."""

    def __init__(self, threshold: float = 0.85, p: float = 0.5):
        super().__init__(p=p)
        self.threshold = threshold

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        thr = self.threshold
        audio = np.where(
            np.abs(audio) <= thr,
            audio,
            np.sign(audio)
            * (thr + (1.0 - thr) * np.tanh((np.abs(audio) - thr) / (1.0 - thr + 1e-9))),
        )
        return audio.astype(np.float32)
