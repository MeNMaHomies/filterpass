"""
File I/O layer: applies an augmentation pipeline to files or directories.

Decoupled from any specific augmentation — accepts any Augmentation callable.
"""

import random
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
from .base import Augmentation

_FORMAT_MAP = {
    ".flac": ("FLAC", "PCM_16"),
    ".wav": ("WAV", "PCM_16"),
}


def augment_file(
    input_path: str,
    output_path: str,
    transform: Augmentation,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)

    stereo = audio.ndim == 2
    channels = [audio[:, ch] for ch in range(audio.shape[1])] if stereo else [audio]
    augmented = [transform(ch, sr) for ch in channels]

    out_audio = np.stack(augmented, axis=1) if stereo else augmented[0]

    # Determine output format based on file extension
    ext = Path(output_path).suffix.lower()
    # Default to FLAC if extension is unrecognized
    fmt, subtype = _FORMAT_MAP.get(ext, ("FLAC", "PCM_16"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out_audio, sr, format=fmt, subtype=subtype)
    print(f"  {input_path}  ->  {output_path}")


def augment_directory(
    input_dir: str,
    output_dir: str,
    transform: Augmentation,
    n_augmentations: int = 1,
) -> None:
    input_files = sorted(
        f for ext in ("*.flac", "*.wav") for f in Path(input_dir).rglob(ext)
    )
    if not input_files:
        print(f"[warn] No .flac or .wav files found in {input_dir}")
        return

    for audio_file in input_files:
        rel = audio_file.relative_to(input_dir)
        for i in range(n_augmentations):
            stem = rel.stem
            out_path = Path(output_dir) / rel.parent / (stem + rel.suffix)
            augment_file(str(audio_file), str(out_path), transform, seed=i)
