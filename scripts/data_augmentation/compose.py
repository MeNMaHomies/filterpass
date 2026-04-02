"""
Combinators for building augmentation pipelines.

    Compose  — apply all transforms in sequence (each gated by its own p)
    OneOf    — pick exactly one transform at random
    SomeOf   — pick N transforms at random (N can be a range)
"""

import random

import numpy as np

from .base import Augmentation


class Compose(Augmentation):
    """Apply every transform in order. Each transform's own `p` controls whether it fires."""

    def __init__(self, transforms: list[Augmentation], p: float = 1.0):
        super().__init__(p=p)
        self.transforms = transforms

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        for t in self.transforms:
            audio = t(audio, sr)
        return audio

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose(p={self.p}, [{inner}])"


class OneOf(Augmentation):
    """Pick exactly one transform at random and apply it."""

    def __init__(self, transforms: list[Augmentation], p: float = 1.0):
        super().__init__(p=p)
        self.transforms = transforms

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        t = random.choice(self.transforms)
        return t.apply(audio, sr)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"OneOf(p={self.p}, [{inner}])"


class SomeOf(Augmentation):
    """
    Pick N transforms at random (without replacement) and apply them in order.
    `n` can be a fixed int or a (min, max) tuple.
    """

    def __init__(self, transforms: list[Augmentation], n: int | tuple[int, int] = 1, p: float = 1.0):
        super().__init__(p=p)
        self.transforms = transforms
        self.n = n

    def _pick_count(self) -> int:
        if isinstance(self.n, tuple):
            return random.randint(self.n[0], min(self.n[1], len(self.transforms)))
        return min(self.n, len(self.transforms))

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        chosen = random.sample(self.transforms, self._pick_count())
        for t in chosen:
            audio = t.apply(audio, sr)
        return audio

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"SomeOf(n={self.n}, p={self.p}, [{inner}])"
