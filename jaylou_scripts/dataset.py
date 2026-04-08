import os
import sys

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPLIT_MAP = {
    "train": (
        "augmented/train/flac",
        "augmented/keys/ASVspoof2019.LA.cm.train.augmented.txt",
    ),
    "dev": (
        "augmented/dev/flac",
        "augmented/keys/ASVspoof2019.LA.cm.dev.augmented.txt",
    ),
    "eval": ("ASVspoof2019_LA_eval/flac", "keys/ASVspoof2019.LA.cm.eval.trl.txt"),
}

_LABEL_MAP = {"bonafide": 0, "spoof": 1}

MAX_SAMPLES = 64_000  # 4s at 16 kHz — standard training length across SOTA


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio via torchaudio+ffmpeg, falling back to librosa on failure."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path, backend="ffmpeg")
        return waveform.squeeze(0).numpy(), sr
    except Exception:
        waveform_np, sr = librosa.load(path, sr=None, mono=False)
        return waveform_np, sr


def _preprocess(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Mono → resample to 16kHz → amplitude normalise → pad/truncate to MAX_SAMPLES."""
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    if sr != 16_000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16_000)

    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    if len(waveform) >= MAX_SAMPLES:
        waveform = waveform[:MAX_SAMPLES]
    else:
        waveform = np.pad(waveform, (0, MAX_SAMPLES - len(waveform)))

    return waveform.astype(np.float32)


class ASVspoofDataset(Dataset):
    """
    4-second fixed-length dataset for ASVspoof 2019 LA.

    Loads from a pre-built .npy cache (see jaylou_scripts/cache_dataset.py) when
    available — this eliminates FLAC decoding overhead on every epoch and is the
    recommended path for training. Falls back to live FLAC loading if no cache
    exists for a given utterance.

    Args:
        base_dir:    Root data_training directory.
        split:       "train", "dev", or "eval".
    """

    def __init__(self, base_dir: str, split: str = "train"):
        if split not in _SPLIT_MAP:
            raise ValueError(f"split must be one of {list(_SPLIT_MAP)}, got '{split}'")

        audio_subdir, protocol_subpath = _SPLIT_MAP[split]
        self.audio_dir = os.path.join(base_dir, audio_subdir)
        protocol_path = os.path.join(base_dir, protocol_subpath)

        cache_dir = os.path.join(base_dir, "cache", split)
        self.cache_dir = cache_dir if os.path.isdir(cache_dir) else None

        self.samples: list[tuple[str, int]] = []

        with open(protocol_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        skipped = 0
        for parts in tqdm(lines, desc=f"Indexing {split}", unit="file"):
            if len(parts) < 5:
                skipped += 1
                continue
            utt_id = parts[1]
            label = _LABEL_MAP.get(parts[4], -1)
            path = os.path.join(self.audio_dir, f"{utt_id}.flac")
            self.samples.append((path, label))

        cache_status = f"cache at {cache_dir}" if self.cache_dir else "no cache (run cache_dataset.py)"
        print(
            f"{split}: {len(self.samples)} utterances indexed  |  "
            f"{skipped} skipped  |  {cache_status}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        waveform_np = self._load(path)
        return {
            "input_values": torch.from_numpy(waveform_np),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _load(self, path: str) -> np.ndarray:
        if self.cache_dir is not None:
            utt_id = os.path.splitext(os.path.basename(path))[0]
            cache_path = os.path.join(self.cache_dir, f"{utt_id}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)

        try:
            waveform_np, sr = _read_audio(path)
            return _preprocess(waveform_np, sr)
        except Exception as e:
            tqdm.write(f"[ERROR] {os.path.basename(path)}: {e}")
            return np.zeros(MAX_SAMPLES, dtype=np.float32)
