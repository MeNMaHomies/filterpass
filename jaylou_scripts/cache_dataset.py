"""
Pre-process all training/dev FLAC files to float32[64000] .npy arrays and
write them to {base_dir}/cache/{split}/.

Run once before training:
    python -m jaylou_scripts.cache_dataset

Subsequent training runs will load .npy files instead of decoding FLAC,
eliminating disk I/O overhead from every epoch.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaylou_scripts.dataset import MAX_SAMPLES, _SPLIT_MAP, _preprocess, _read_audio
from jaylou_scripts.config import CONFIG


def _cache_file(args: tuple) -> tuple[bool, str]:
    flac_path, cache_path = args
    if os.path.exists(cache_path):
        return True, ""
    try:
        waveform_np, sr = _read_audio(flac_path)
        waveform_np = _preprocess(waveform_np, sr)
        np.save(cache_path, waveform_np)
        return True, ""
    except Exception as e:
        return False, f"{flac_path}: {e}"


def build_cache(
    base_dir: str,
    splits: tuple[str, ...] = ("train", "dev"),
    max_workers: int = 8,
) -> None:
    for split in splits:
        audio_subdir, protocol_subpath = _SPLIT_MAP[split]
        audio_dir = os.path.join(base_dir, audio_subdir)
        protocol_path = os.path.join(base_dir, protocol_subpath)
        cache_dir = os.path.join(base_dir, "cache", split)
        os.makedirs(cache_dir, exist_ok=True)

        with open(protocol_path) as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]

        tasks = []
        for parts in lines:
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            tasks.append((
                os.path.join(audio_dir, f"{utt_id}.flac"),
                os.path.join(cache_dir, f"{utt_id}.npy"),
            ))

        already_cached = sum(1 for _, cp in tasks if os.path.exists(cp))
        remaining = len(tasks) - already_cached
        print(
            f"\n[{split}] {len(tasks)} utterances  |  "
            f"{already_cached} already cached  |  {remaining} to process"
        )

        if remaining == 0:
            print(f"[{split}] Nothing to do.")
            continue

        errors = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_cache_file, t): t for t in tasks}
            with tqdm(total=len(tasks), desc=f"Caching {split}", unit="file") as pbar:
                for future in as_completed(futures):
                    ok, msg = future.result()
                    if not ok:
                        errors += 1
                        tqdm.write(f"[ERROR] {msg}")
                    pbar.update(1)

        print(f"[{split}] Complete  |  {errors} errors")


if __name__ == "__main__":
    build_cache(CONFIG["base_dir"])
