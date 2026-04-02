"""
Audio loading and fixed-size chunking for benchmark inference.

Each utterance is loaded by a DataLoader worker, split into CHUNK_SAMPLES-sized
frames (zero-padded on the final chunk), and returned as a float32 tensor.
"""

import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CHUNK_SAMPLES, TARGET_SR


def _load_and_chunk(flac_path: str) -> tuple[str, torch.Tensor | None]:
    """
    Runs inside DataLoader worker processes.
    Returns (utt_id, chunks_tensor) where chunks_tensor is (N, CHUNK_SAMPLES),
    or (utt_id, None) on any failure.
    """
    utt_id = os.path.splitext(os.path.basename(flac_path))[0]
    try:
        try:
            audio, sr = sf.read(flac_path, dtype="float32", always_2d=False)
        except Exception as sf_err:
            print(f"[FALLBACK:TORCHAUDIO] {utt_id}: {sf_err}", flush=True)
            import torchaudio
            waveform, sr = torchaudio.load(flac_path)
            audio = waveform[0].numpy()

        if audio.ndim > 1:
            audio = audio[:, 0]

        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        if len(audio) == 0:
            return utt_id, None

        chunks = []
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i : i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            chunks.append(chunk)

        return utt_id, torch.from_numpy(np.stack(chunks))
    except Exception as e:
        print(f"[WORKER ERROR] {utt_id}: {type(e).__name__}: {e}", flush=True)
        return utt_id, None


class UtteranceDataset(Dataset):
    """One item per FLAC file path; workers perform loading + chunking."""

    def __init__(self, flac_paths: list[str]):
        self.paths = flac_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _load_and_chunk(self.paths[idx])


def _collate(batch):
    """Pass utterances through unchanged — chunk counts differ per file."""
    return batch


def build_loader(flac_paths: list[str], num_workers: int) -> DataLoader:
    return DataLoader(
        UtteranceDataset(flac_paths),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
