"""
Audio loading and chunking for benchmark inference.

Supports two modes:
  - Blind chunking (default): every sample goes through, no filtering.
  - VAD-filtered chunking (--vad): WebRTC VAD discards silence at the 30ms
    frame level, then the remaining voiced audio is chunked.

Both modes support an optional overlapping sliding window via overlap_pct.
"""

import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CHUNK_SAMPLES, TARGET_SR

_VAD_FRAME_MS = 30  # WebRTC VAD requires 10, 20, or 30 ms frames
_BYTES_PER_SAMPLE = 2  # 16-bit PCM


# ── VAD ───────────────────────────────────────────────────────────────────────


def _vad_filter(audio: np.ndarray, sr: int, vad_mode: int) -> np.ndarray:
    """
    Run WebRTC VAD on 30ms frames and return only the voiced samples
    as a contiguous float32 array. Silence frames are discarded entirely.
    """
    import webrtcvad

    vad = webrtcvad.Vad(vad_mode)

    # WebRTC VAD needs 16-bit PCM bytes
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_bytes = int(sr * (_VAD_FRAME_MS / 1000.0)) * _BYTES_PER_SAMPLE
    frame_samples = int(sr * (_VAD_FRAME_MS / 1000.0))

    voiced_samples = []
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame_pcm = pcm[offset : offset + frame_bytes]

        # If background noise, discard the entire frame; if speech, keep it.
        if vad.is_speech(frame_pcm, sr):
            sample_start = offset // _BYTES_PER_SAMPLE
            voiced_samples.append(audio[sample_start : sample_start + frame_samples])

    if not voiced_samples:
        return np.array([], dtype=np.float32)

    return np.concatenate(voiced_samples)


# ── Chunking ──────────────────────────────────────────────────────────────────


def _chunk_audio(audio: np.ndarray, hop_samples: int) -> list[np.ndarray]:
    """
    Split audio into CHUNK_SAMPLES-sized windows advancing by hop_samples.
    The final chunk is zero-padded if shorter than CHUNK_SAMPLES.
    """
    if len(audio) == 0:
        return []

    chunks = []

    # Full windows
    for start in range(0, max(len(audio) - CHUNK_SAMPLES + 1, 1), hop_samples):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)

    return chunks


# ── Worker function ───────────────────────────────────────────────────────────


def _load_and_chunk(
    flac_path: str,
    use_vad: bool,
    vad_mode: int,
    hop_samples: int,
) -> tuple[str, torch.Tensor | None]:
    """
    Runs inside DataLoader worker processes.
    Returns (utt_id, chunks_tensor) where chunks_tensor is (N, CHUNK_SAMPLES),
    or (utt_id, None) on any failure.
    """
    utt_id = os.path.splitext(os.path.basename(flac_path))[0]
    try:
        import librosa
        import torchaudio

        try:
            waveform, sr = torchaudio.load(flac_path, backend="ffmpeg")
            audio = waveform.mean(dim=0).numpy()
        except Exception:
            audio, sr = librosa.load(flac_path, sr=None, mono=True)

        # Force 16 kHz
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        if len(audio) == 0:
            return utt_id, None

        # VAD: discard silence at 30ms frame level, keep only voiced audio
        if use_vad:
            audio = _vad_filter(audio, TARGET_SR, vad_mode)
            if len(audio) == 0:
                return utt_id, None

        # Chunk with sliding window (hop_samples == CHUNK_SAMPLES = no overlap)
        chunks = _chunk_audio(audio, hop_samples)
        if not chunks:
            return utt_id, None

        return utt_id, torch.from_numpy(np.stack(chunks))
    except Exception as e:
        print(f"[WORKER ERROR] {utt_id}: {type(e).__name__}: {e}", flush=True)
        return utt_id, None


# ── Dataset + DataLoader ──────────────────────────────────────────────────────


class UtteranceDataset(Dataset):
    """One item per FLAC file path; workers perform loading + chunking."""

    def __init__(
        self,
        flac_paths: list[str],
        use_vad: bool = False,
        vad_mode: int = 2,
        hop_samples: int = CHUNK_SAMPLES,
    ):
        self.paths = flac_paths
        self.use_vad = use_vad
        self.vad_mode = vad_mode
        self.hop_samples = hop_samples

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _load_and_chunk(
            self.paths[idx],
            self.use_vad,
            self.vad_mode,
            self.hop_samples,
        )


def _collate(batch):
    """Pass utterances through unchanged — chunk counts differ per file."""
    return batch


def build_loader(
    flac_paths: list[str],
    num_workers: int,
    use_vad: bool = False,
    vad_mode: int = 2,
    overlap_pct: int = 0,
) -> DataLoader:
    hop_samples = max(1, int(CHUNK_SAMPLES * (1 - overlap_pct / 100)))

    return DataLoader(
        UtteranceDataset(flac_paths, use_vad, vad_mode, hop_samples),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
