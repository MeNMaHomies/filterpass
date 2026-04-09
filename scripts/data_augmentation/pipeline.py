"""
File I/O layer: applies an augmentation pipeline to files or directories.

augment_file / augment_directory — generic, accept any Augmentation callable.
augment_bonafide / augment_spoof  — preset-aware helpers used by the oversampling
                                    pipeline (guaranteed + stochastic selection).
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .base import Augmentation
from .compose import Compose

_FORMAT_MAP = {
    ".flac": ("FLAC", "PCM_16"),
    ".wav": ("WAV", "PCM_16"),
}

PresetList = list[tuple[str, type[Compose]]]


def _apply_preset(audio: np.ndarray, sr: int, preset_fn) -> np.ndarray:
    transform = preset_fn()
    if audio.ndim == 2:
        channels = [transform(audio[:, ch], sr) for ch in range(audio.shape[1])]
        return np.stack(channels, axis=1)
    return transform(audio, sr)


def _write(output_path: str, audio: np.ndarray, sr: int) -> None:
    ext = Path(output_path).suffix.lower()
    fmt, subtype = _FORMAT_MAP.get(ext, ("FLAC", "PCM_16"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr, format=fmt, subtype=subtype)


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
    if audio.ndim == 2:
        channels = [transform(audio[:, ch], sr) for ch in range(audio.shape[1])]
        out_audio = np.stack(channels, axis=1)
    else:
        out_audio = transform(audio, sr)
    _write(output_path, out_audio, sr)


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


def augment_bonafide(
    input_path: str,
    output_dir: str,
    audio_id: str,
    guaranteed: PresetList,
    stochastic_pool: PresetList,
    n_stochastic: int,
    extension: str = ".flac",
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Produces all augmented copies for a single bonafide file.

    Applies every preset in `guaranteed`, then draws `n_stochastic` presets
    with replacement from `stochastic_pool` (seeded on audio_id for resumability).

    Returns (results, errors) where results is a list of (aug_id, output_path).
    Skips files already present on disk.
    """
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    results: list[tuple[str, str]] = []
    errors: list[str] = []

    for preset_name, preset_fn in guaranteed:
        aug_id = f"{audio_id}_{preset_name}"
        out_path = str(Path(output_dir) / (aug_id + extension))
        if not Path(out_path).exists():
            try:
                _write(out_path, _apply_preset(audio, sr, preset_fn), sr)
            except Exception as e:
                errors.append(f"Failed {audio_id} with {preset_name}: {e}")
                continue
        results.append((aug_id, out_path))

    rng = random.Random(audio_id)
    picks = [rng.choice(stochastic_pool) for _ in range(n_stochastic)]
    for slot, (preset_name, preset_fn) in enumerate(picks):
        aug_id = f"{audio_id}_s{slot}_{preset_name}"
        out_path = str(Path(output_dir) / (aug_id + extension))
        if not Path(out_path).exists():
            try:
                _write(out_path, _apply_preset(audio, sr, preset_fn), sr)
            except Exception as e:
                errors.append(
                    f"Failed {audio_id} with {preset_name} (slot {slot}): {e}"
                )
                continue
        results.append((aug_id, out_path))

    return results, errors


def augment_spoof(
    input_path: str,
    output_path: str,
    preset_pool: PresetList,
) -> None:
    """Applies a single randomly selected preset from `preset_pool` to a spoof file."""
    _, preset_fn = random.choice(preset_pool)
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    _write(output_path, _apply_preset(audio, sr, preset_fn), sr)
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    _write(output_path, _apply_preset(audio, sr, preset_fn), sr)
