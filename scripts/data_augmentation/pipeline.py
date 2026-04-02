"""
Full augmentation pipeline: orchestrates all augmentation modules over a file or directory.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .config import AugmentationConfig
from . import noise, rir, bandpass, mic_coloration, codec_distortion, packet_loss, time_stretch, gain


def _augment_channel(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    audio = noise.augment(audio, sr, cfg)
    audio = rir.augment(audio, sr, cfg)
    audio = bandpass.augment(audio, sr, cfg)
    audio = mic_coloration.augment(audio, sr, cfg)
    audio = codec_distortion.augment(audio, cfg)
    audio = packet_loss.augment(audio, sr, cfg)
    audio = time_stretch.augment(audio, cfg)
    audio = gain.augment(audio, cfg)
    return audio


def augment_file(
    input_path: str,
    output_path: str,
    cfg: AugmentationConfig,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)

    stereo = audio.ndim == 2
    channels = [audio[:, ch] for ch in range(audio.shape[1])] if stereo else [audio]
    augmented = [_augment_channel(ch, sr, cfg) for ch in channels]

    out_audio = np.stack(augmented, axis=1) if stereo else augmented[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out_audio, sr, format="FLAC", subtype=cfg.output_subtype)
    print(f"  ✓  {input_path}  →  {output_path}")


def augment_directory(
    input_dir: str,
    output_dir: str,
    cfg: AugmentationConfig,
    n_augmentations: int = 1,
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
            augment_file(str(flac_file), str(out_path), cfg, seed=i)
