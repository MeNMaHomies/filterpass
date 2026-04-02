"""
Abstract base class for all augmentations.

Every augmentation is a callable with a probability gate:
    output = SomeAugmentation(p=0.5)(audio, sr)

Subclasses implement `apply(audio, sr)` — the base handles the coin flip.
"""

import random
from abc import ABC, abstractmethod

import numpy as np


class Augmentation(ABC):
    def __init__(self, p: float = 1.0):
        self.p = p

    @abstractmethod
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Transform audio. Guaranteed to be called only when the coin flip passes."""

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < self.p:
            return self.apply(audio, sr)
        return audio

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"
