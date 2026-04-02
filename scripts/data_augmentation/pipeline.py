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


def augment_file(
    input_path: str,
    output_path: str,
    transform: Augmentation,
    seed: Optional[int] = None,
    output_subtype: str = "PCM_16",
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)

    stereo = audio.ndim == 2
    channels = [audio[:, ch] for ch in range(audio.shape[1])] if stereo else [audio]
    augmented = [transform(ch, sr) for ch in channels]

    out_audio = np.stack(augmented, axis=1) if stereo else augmented[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out_audio, sr, format="FLAC", subtype=output_subtype)
    print(f"  {input_path}  ->  {output_path}")


def augment_directory(
    input_dir: str,
    output_dir: str,
    transform: Augmentation,
    n_augmentations: int = 1,
    output_subtype: str = "PCM_16",
) -> None:
    input_files = sorted(Path(input_dir).rglob("*.flac"))
    if not input_files:
        print(f"[warn] No .flac files found in {input_dir}")
        return

    for flac_file in input_files:
        rel = flac_file.relative_to(input_dir)
        for i in range(n_augmentations):
            stem = rel.stem + (f"_aug{i + 1}" if n_augmentations > 1 else "_aug")
            out_path = Path(output_dir) / rel.parent / (stem + ".flac")
            augment_file(str(flac_file), str(out_path), transform, seed=i, output_subtype=output_subtype)
