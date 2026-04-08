import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_augmentation.presets import (
    full,
    headset,
    light,
    meeting_room,
    noisy_mobile,
    pstn,
    telephony,
    voip,
)

_PRESETS = [
    ("telephony", telephony),
    ("pstn", pstn),
    ("voip", voip),
    ("light", light),
    ("full", full),
    ("meeting_room", meeting_room),
    ("noisy_mobile", noisy_mobile),
    ("headset", headset),
]

_FORMAT_MAP = {
    ".flac": ("FLAC", "PCM_16"),
    ".wav": ("WAV", "PCM_16"),
}


def _write_audio(path: str, audio: np.ndarray, sr: int, extension: str) -> None:
    fmt, subtype = _FORMAT_MAP.get(extension, ("FLAC", "PCM_16"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, format=fmt, subtype=subtype)


def _apply_transform(audio: np.ndarray, sr: int, preset_fn) -> np.ndarray:
    transform = preset_fn()
    stereo = audio.ndim == 2
    if stereo:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]
        augmented = [transform(ch, sr) for ch in channels]
        return np.stack(augmented, axis=1)
    return transform(audio, sr)


def _process_bonafide_file(args: tuple) -> tuple[list, list]:
    parts, audio_dir, output_dir, extension = args

    audio_id = parts[1]
    input_path = os.path.join(audio_dir, audio_id + extension)

    if not os.path.exists(input_path):
        return [], [f"Missing: {input_path}"]

    rows = []
    errors = []

    # Original clean copy
    orig_out = os.path.join(output_dir, audio_id + extension)
    if not os.path.exists(orig_out):
        shutil.copy2(input_path, orig_out)
    rows.append(parts)

    # Load audio once, apply all presets in memory
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)

    for preset_name, preset_fn in _PRESETS:
        aug_id = f"{audio_id}_{preset_name}"
        output_path = os.path.join(output_dir, aug_id + extension)
        aug_row = [parts[0], aug_id] + parts[2:]

        if os.path.exists(output_path):
            rows.append(aug_row)
            continue

        try:
            aug_audio = _apply_transform(audio, sr, preset_fn)
            _write_audio(output_path, aug_audio, sr, extension)
            rows.append(aug_row)
        except Exception as e:
            errors.append(f"Failed {audio_id} with {preset_name}: {e}")

    return rows, errors


def _process_spoof_file(args: tuple) -> tuple[list | None, str | None]:
    parts, audio_dir, output_dir, extension = args

    audio_id = parts[1]
    input_path = os.path.join(audio_dir, audio_id + extension)
    output_path = os.path.join(output_dir, audio_id + extension)

    if not os.path.exists(input_path):
        return None, f"Missing: {input_path}"

    if os.path.exists(output_path):
        return parts, None

    _, preset_fn = random.choice(_PRESETS)
    try:
        audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
        aug_audio = _apply_transform(audio, sr, preset_fn)
        _write_audio(output_path, aug_audio, sr, extension)
        return parts, None
    except Exception as e:
        return None, f"Failed {audio_id}: {e}"


def separate_data(input_file):
    """
    Reads the ASVspoof protocol file, separates rows into bonafide and spoof,
    and returns them as in-memory lists.
    """
    bonafide_data = []
    spoof_data = []

    try:
        with open(input_file, "r") as infile:
            for line in infile:
                parts = line.strip().split()
                if not parts:
                    continue

                label = parts[-1].lower()

                if label == "bonafide":
                    bonafide_data.append(parts)
                elif label == "spoof":
                    spoof_data.append(parts)

        print(f"Separation complete!")
        print(f"Total bonafide records: {len(bonafide_data)}")
        print(f"Total spoof records: {len(spoof_data)}\n")

        return bonafide_data, spoof_data

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return [], []


def oversample_bonafide(
    bonafide_list: list,
    audio_dir: str,
    output_dir: str,
    extension: str = ".flac",
    max_workers: int | None = None,
) -> list:
    """
    For each bonafide utterance:
      - Copies the original clean file to output_dir unchanged.
      - Generates one augmented copy per preset in _PRESETS (loaded once from disk).
    Skips files already present on disk for resumability.
    Returns protocol rows for all files (original + augmented).
    """
    os.makedirs(output_dir, exist_ok=True)

    valid = [p for p in bonafide_list if len(p) >= 2]
    args = [(p, audio_dir, output_dir, extension) for p in valid]

    new_rows = []
    workers = max_workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_bonafide_file, a): a for a in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Bonafide oversampling"):
            rows, errors = future.result()
            new_rows.extend(rows)
            for err in errors:
                tqdm.write(err)

    print(f"\nBonafide complete: {len(new_rows)} total files (original + augmented).")
    return new_rows


def augment_spoof(
    spoof_list: list,
    audio_dir: str,
    output_dir: str,
    extension: str = ".flac",
    max_workers: int | None = None,
) -> list:
    """
    For each spoof utterance, apply a randomly sampled preset in-place
    (no duplication — one augmented file replaces the original).
    Skips files already present on disk for resumability.
    Returns protocol rows with the same IDs.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid = [p for p in spoof_list if len(p) >= 2]
    args = [(p, audio_dir, output_dir, extension) for p in valid]

    rows = []
    workers = max_workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_spoof_file, a): a for a in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Spoof augmentation"):
            result, error = future.result()
            if result is not None:
                rows.append(result)
            if error:
                tqdm.write(error)

    print(f"Spoof complete: {len(rows)} files augmented in-place.")
    return rows


def write_augmented_protocol(
    bonafide_rows: list,
    spoof_rows: list,
    output_path: str,
) -> None:
    """Write a merged protocol file containing bonafide + spoof rows."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_rows = bonafide_rows + spoof_rows
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(" ".join(row) + "\n")
    print(f"Protocol written: {output_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    protocol_path = "./data_training/keys/ASVspoof2019.LA.cm.dev.trl.txt"
    audio_folder_path = "./data_training/ASVspoof2019_LA_dev/flac"
    augmented_audio_dir = "./data_training/augmented/dev/flac"
    augmented_protocol_path = (
        "./data_training/augmented/keys/ASVspoof2019.LA.cm.dev.augmented.txt"
    )

    bonafide_list, spoof_list = separate_data(protocol_path)

    bonafide_rows = oversample_bonafide(
        bonafide_list,
        audio_dir=audio_folder_path,
        output_dir=augmented_audio_dir,
    )

    spoof_rows = augment_spoof(
        spoof_list,
        audio_dir=audio_folder_path,
        output_dir=augmented_audio_dir,
    )

    write_augmented_protocol(bonafide_rows, spoof_rows, augmented_protocol_path)

    print("\n--- Augmented dataset summary ---")
    separate_data(augmented_protocol_path)
