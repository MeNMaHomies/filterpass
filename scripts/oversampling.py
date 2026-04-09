import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_augmentation.pipeline import augment_bonafide, augment_spoof
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

# Presets always applied to every bonafide file (telephony-domain coverage)
GUARANTEED_PRESETS = [
    ("telephony", telephony),
    ("pstn", pstn),
    ("voip", voip),
]

# Presets sampled stochastically — each file gets a different draw (with replacement)
STOCHASTIC_POOL = [
    ("light", light),
    ("full", full),
    ("meeting_room", meeting_room),
    ("noisy_mobile", noisy_mobile),
    ("headset", headset),
]

# 1 original + 3 guaranteed + N stochastic = 9 copies per bonafide file
N_STOCHASTIC = len(STOCHASTIC_POOL)


def _process_bonafide_file(args: tuple) -> tuple[list, list]:
    """
    Worker target for ProcessPoolExecutor — must be a module-level function to be picklable.

    Copies the original file verbatim, then delegates all augmentation to augment_bonafide.
    The original copy is always included so the model trains on clean speech as well.
    Protocol rows for every output file (original + augmented) are returned so the caller
    can build the augmented protocol without any shared state across workers.
    """
    parts, audio_dir, output_dir, extension = args

    audio_id = parts[1]
    input_path = os.path.join(audio_dir, audio_id + extension)

    if not os.path.exists(input_path):
        return [], [f"Missing: {input_path}"]

    rows = []
    errors = []

    orig_out = os.path.join(output_dir, audio_id + extension)
    if not os.path.exists(orig_out):
        shutil.copy2(input_path, orig_out)
    rows.append(parts)

    aug_results, aug_errors = augment_bonafide(
        input_path,
        output_dir,
        audio_id,
        GUARANTEED_PRESETS,
        STOCHASTIC_POOL,
        N_STOCHASTIC,
        extension,
    )
    errors.extend(aug_errors)
    for aug_id, _ in aug_results:
        # Preserve all protocol fields; only the utterance ID (index 1) changes.
        rows.append([parts[0], aug_id] + parts[2:])

    return rows, errors


def _process_spoof_file(args: tuple) -> tuple[list | None, str | None]:
    """
    Worker target for ProcessPoolExecutor — must be a module-level function to be picklable.

    Applies a single randomly drawn preset in-place (no duplication). Spoof files are not
    oversampled — augmentation here only ensures they go through the same channel distortion
    domain as the bonafide files, preventing a clean/degraded distribution mismatch.
    """
    parts, audio_dir, output_dir, extension = args

    audio_id = parts[1]
    input_path = os.path.join(audio_dir, audio_id + extension)
    output_path = os.path.join(output_dir, audio_id + extension)

    if not os.path.exists(input_path):
        return None, f"Missing: {input_path}"

    if os.path.exists(output_path):
        return parts, None

    try:
        augment_spoof(input_path, output_path, GUARANTEED_PRESETS + STOCHASTIC_POOL)
        return parts, None
    except Exception as e:
        return None, f"Failed {audio_id}: {e}"


def separate_data(input_file: str) -> tuple[list, list]:
    """
    Parses an ASVspoof protocol file and splits rows by label.

    Each row is returned as a list of whitespace-split tokens (the raw protocol columns),
    preserving the original field order for downstream protocol writing.
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
    Expands the bonafide set by generating augmented copies of every utterance in parallel.

    Each file produces 1 original + N augmented versions (see GUARANTEED_PRESETS,
    STOCHASTIC_POOL, N_STOCHASTIC). Existing files are skipped, so interrupted runs
    can be safely resumed without re-processing completed files.

    Returns a flat list of protocol rows covering all output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid = [p for p in bonafide_list if len(p) >= 2]
    args = [(p, audio_dir, output_dir, extension) for p in valid]

    new_rows = []
    workers = max_workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_bonafide_file, a): a for a in args}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Bonafide oversampling"
        ):
            rows, errors = future.result()
            new_rows.extend(rows)
            for err in errors:
                tqdm.write(err)

    print(f"\nBonafide complete: {len(new_rows)} total files (original + augmented).")
    return new_rows


def augment_spoof_batch(
    spoof_list: list,
    audio_dir: str,
    output_dir: str,
    extension: str = ".flac",
    max_workers: int | None = None,
) -> list:
    """
    Applies a random augmentation preset to every spoof utterance in parallel.

    One augmented file replaces the original — spoof files are not duplicated.
    Existing output files are skipped for resumability.

    Returns protocol rows with the same IDs as the input (no new entries are created).
    """
    os.makedirs(output_dir, exist_ok=True)

    valid = [p for p in spoof_list if len(p) >= 2]
    args = [(p, audio_dir, output_dir, extension) for p in valid]

    rows = []
    workers = max_workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_spoof_file, a): a for a in args}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Spoof augmentation"
        ):
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
    """Merges bonafide and spoof rows and writes them to a single protocol file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_rows = bonafide_rows + spoof_rows
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(" ".join(row) + "\n")
    print(f"Protocol written: {output_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    protocol_path = "./data_training/keys/ASVspoof2019.LA.cm.train.trn.txt"
    audio_folder_path = "./data_training/ASVspoof2019_LA_train/flac"
    augmented_audio_dir = "./data_training/augmented/train/flac"
    augmented_protocol_path = (
        "./data_training/augmented/keys/ASVspoof2019.LA.cm.train.augmented.txt"
    )

    bonafide_list, spoof_list = separate_data(protocol_path)

    bonafide_rows = oversample_bonafide(
        bonafide_list,
        audio_dir=audio_folder_path,
        output_dir=augmented_audio_dir,
    )

    spoof_rows = augment_spoof_batch(
        spoof_list,
        audio_dir=audio_folder_path,
        output_dir=augmented_audio_dir,
    )

    write_augmented_protocol(bonafide_rows, spoof_rows, augmented_protocol_path)

    print("\n--- Augmented dataset summary ---")
    separate_data(augmented_protocol_path)
