"""
In-memory augmented audio dataset for PyTorch.

Loads audio files lazily and applies augmentations on the fly — no disk writes.
Supports label inference from folder names or an explicit label map.

    dataset = AudioDataset("data/eval/", transform=telephony(), sr=16000)
    audio, label = dataset[0]
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .base import Augmentation

_SUPPORTED_EXTENSIONS = {".flac", ".wav"}


class AudioDataset(Dataset):
    """
    PyTorch Dataset that loads audio and applies augmentations in memory.

    Label resolution order:
        1. Explicit ``labels`` dict  (utt_id or filename -> int)
        2. Folder name mapping via ``label_map``  (folder -> int)
        3. No labels (returns -1)

    Parameters
    ----------
    audio_dir : str
        Root directory containing .flac / .wav files (searched recursively).
    transform : Augmentation, optional
        Augmentation pipeline to apply on each __getitem__ call.
    sr : int
        Target sample rate. Files at a different rate are resampled.
    labels : dict, optional
        Mapping of stem (e.g. "LA_E_9332881") or filename to integer label.
    label_map : dict, optional
        Mapping of parent folder name to integer label.
        E.g. {"bonafide": 0, "spoof": 1}.
    mono : bool
        If True, downmix stereo to mono.
    max_length : int, optional
        If set, pad or truncate audio to exactly this many samples.
    """

    def __init__(
        self,
        audio_dir: str,
        transform: Optional[Augmentation] = None,
        sr: int = 16_000,
        labels: Optional[dict[str, int]] = None,
        label_map: Optional[dict[str, int]] = None,
        mono: bool = True,
        max_length: Optional[int] = None,
    ):
        self.transform = transform
        self.sr = sr
        self.labels = labels or {}
        self.label_map = label_map or {}
        self.mono = mono
        self.max_length = max_length

        self.files = sorted(
            f
            for f in Path(audio_dir).rglob("*")
            if f.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not self.files:
            raise FileNotFoundError(
                f"No .flac or .wav files found in {audio_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.files[idx]
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)

        if audio.ndim > 1 and self.mono:
            audio = audio.mean(axis=1)

        if file_sr != self.sr:
            import scipy.signal as sps
            audio = sps.resample(
                audio, int(len(audio) * self.sr / file_sr)
            ).astype(np.float32)

        if self.transform is not None:
            audio = self.transform(audio, self.sr)

        if self.max_length is not None:
            if len(audio) > self.max_length:
                audio = audio[: self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)))

        label = self._resolve_label(path)
        return torch.from_numpy(audio), label

    def _resolve_label(self, path: Path) -> int:
        stem = path.stem
        if stem in self.labels:
            return self.labels[stem]

        name = path.name
        if name in self.labels:
            return self.labels[name]

        folder = path.parent.name
        if folder in self.label_map:
            return self.label_map[folder]

        return -1
