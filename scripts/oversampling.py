import os
import random

from tqdm import tqdm

from scripts.data_augmentation.pipeline import augment_file
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


def oversample_bonafide(
    bonafide_list: list,
    audio_dir: str,
    output_dir: str,
    extension: str = ".flac",
) -> list:
    """
    For each bonafide utterance:
      - Copies the original clean file to output_dir unchanged.
      - Generates one augmented copy per preset in _PRESETS.
    Returns protocol rows for all files (original + augmented).
    """
    os.makedirs(output_dir, exist_ok=True)
    new_rows = []

    import shutil

    for parts in tqdm(bonafide_list, desc="Bonafide oversampling"):
        if len(parts) < 2:
            continue

        audio_id = parts[1]
        input_path = os.path.join(audio_dir, audio_id + extension)

        if not os.path.exists(input_path):
            tqdm.write(f"Missing: {input_path}")
            continue

        # Original clean copy
        shutil.copy2(input_path, os.path.join(output_dir, audio_id + extension))
        new_rows.append(parts)

        # One augmented copy per preset
        for preset_name, preset_fn in _PRESETS:
            aug_id = f"{audio_id}_{preset_name}"
            output_path = os.path.join(output_dir, aug_id + extension)
            try:
                augment_file(input_path, output_path, preset_fn())
                aug_row = parts[:-1] + [aug_id] + [parts[-1]]
                new_rows.append(aug_row)
            except Exception as e:
                tqdm.write(f"Failed {audio_id} with {preset_name}: {e}")

    print(f"\nBonafide complete: {len(new_rows)} total files (original + augmented).")
    return new_rows


def augment_spoof(
    spoof_list: list,
    audio_dir: str,
    output_dir: str,
    extension: str = ".flac",
) -> list:
    """
    For each spoof utterance, apply a randomly sampled preset in-place
    (no duplication — one augmented file replaces the original).
    Returns protocol rows with the same IDs (output dir changes, not filenames).
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for parts in tqdm(spoof_list, desc="Spoof augmentation"):
        if len(parts) < 2:
            continue

        audio_id = parts[1]
        input_path = os.path.join(audio_dir, audio_id + extension)
        output_path = os.path.join(output_dir, audio_id + extension)

        if not os.path.exists(input_path):
            tqdm.write(f"Missing: {input_path}")
            continue

        _, preset_fn = random.choice(_PRESETS)
        try:
            augment_file(input_path, output_path, preset_fn())
            rows.append(parts)
        except Exception as e:
            tqdm.write(f"Failed {audio_id}: {e}")

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
    protocol_path = "./data_training/keys/ASVspoof2019.LA.cm.train.trn.txt"
    audio_folder_path = "./data_training/ASVspoof2019_LA_train/flac"
    augmented_audio_dir = "./data_training/augmented/flac"
    augmented_protocol_path = (
        "./data_training/augmented/keys/ASVspoof2019.LA.cm.train.augmented.txt"
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
