import os

import librosa
import numpy as np
import soundfile as sf
import torch
from duy_scripts.preprocessing import _chunk_audio, _vad_filter
from torch.utils.data import Dataset
from tqdm import tqdm


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio via torchaudio+ffmpeg, falling back to librosa on failure."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path, backend="ffmpeg")
        return waveform.squeeze(0).numpy(), sr
    except Exception:
        waveform_np, sr = librosa.load(path, sr=None, mono=False)
        return waveform_np, sr


class ASVspoof2019LADataset(Dataset):
    def __init__(
        self,
        base_dir,
        split="train",
        use_vad=False,
        vad_mode=2,
        overlap_pct=0,
        chunk_samples=8000,
    ):
        """
        Args:
            base_dir (str):     Path to the root 'ASVspoof2019_LA' directory.
            split (str):        One of 'train', 'dev', or 'eval'.
            use_vad (bool):     Whether to filter silence via WebRTC VAD.
            vad_mode (int):     VAD aggressiveness (0=least, 3=most aggressive).
            overlap_pct (int):  Overlap between chunks as a percentage (0-99).
            chunk_samples (int): Number of samples per chunk.
        """
        self.base_dir = base_dir
        self.split = split
        self.use_vad = use_vad
        self.vad_mode = vad_mode
        self.chunk_samples = chunk_samples
        self.hop_samples = max(1, int(chunk_samples * (1 - overlap_pct / 100)))

        # split_map = {
        #     "train": ("train", "ASVspoof2019.LA.cm.train.trn.txt"),
        #     "dev":   ("dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
        #     "eval":  ("eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
        # }

        split_map = {
            "train": ("augmented/train/flac", "ASVspoof2019.LA.cm.train.augmented.txt"),
            "dev":   ("augmented/dev/flac",   "ASVspoof2019.LA.cm.dev.augmented.txt"),
            "eval":  ("ASVspoof2019_LA_eval/flac", "ASVspoof2019.LA.cm.eval.trl.txt"),
        }

        audio_dir_name, protocol_name = split_map[split]
        # self.audio_dir = os.path.join("../output/", audio_dir_name)
        self.audio_dir = os.path.join(base_dir, audio_dir_name)
        protocol_path = os.path.join(base_dir, "augmented/keys", protocol_name)
        self.label_map = {"bonafide": 0, "spoof": 1}

        # Parse protocol file
        raw_data = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    raw_data.append(
                        {
                            "path": os.path.join(self.audio_dir, f"{parts[1]}.flac"),
                            "label": self.label_map[parts[4]],
                        }
                    )

        # Pre-load all audio into memory during __init__ so __getitem__ never
        # touches disk. Stored as a dict: path -> list[np.ndarray] | None.
        # This also makes the dataset safe with num_workers > 0 because the
        # cache is fully populated before any worker is forked.
        print(f"Pre-loading {len(raw_data)} files into chunks...")
        self._chunk_cache: dict[str, list[np.ndarray] | None] = {}
        self.samples: list[tuple[str, int, int]] = []

        for item in tqdm(raw_data, desc=f"Loading {split}", unit="file"):
            path = item["path"]
            utt_id, chunks = self._load_chunks(path)
            # _load_chunks writes to _chunk_cache internally (success + failure)
            if chunks is None:
                tqdm.write(f"[SKIPPED] {utt_id}: failed to load or produced no chunks")
                continue
            for i in range(len(chunks)):
                self.samples.append((path, item["label"], i))

        print(f"Total chunks: {len(self.samples)} load for ", self.audio_dir)

    def _load_chunks(self, path: str) -> tuple[str, list[np.ndarray] | None]:
        """
        Load a FLAC file, preprocess, and split into chunks.
        Results (including failures) are written to self._chunk_cache so that
        __getitem__ never re-reads from disk.

        Returns:
            (utt_id, chunks) where chunks is a list of np.ndarray,
            or (utt_id, None) on any failure or if no chunks were produced.
        """
        utt_id = os.path.splitext(os.path.basename(path))[0]

        # Return cached result if already processed
        if path in self._chunk_cache:
            return utt_id, self._chunk_cache[path]

        try:
            waveform_np, sr = _read_audio(path)

            # 1. Force mono
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)

            # 2. Resample to 16 kHz if needed
            if sr != 16000:
                waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=16000)

            if len(waveform_np) == 0:
                self._chunk_cache[path] = None
                return utt_id, None

            # 3. Amplitude normalisation to [-1, 1]
            max_val = np.max(np.abs(waveform_np))
            if max_val > 0:
                waveform_np = waveform_np / max_val

            # 4. Optional VAD filtering
            if self.use_vad:
                waveform_np = _vad_filter(waveform_np, 16000, self.vad_mode)
                if len(waveform_np) == 0:
                    self._chunk_cache[path] = None
                    return utt_id, None

            # 5. Sliding-window chunking — passes chunk_samples so the
            #    dataset parameter is actually respected
            chunks = _chunk_audio(waveform_np, self.hop_samples, self.chunk_samples)
            if not chunks:
                self._chunk_cache[path] = None
                return utt_id, None

            self._chunk_cache[path] = chunks
            return utt_id, chunks

        except Exception as e:
            tqdm.write(f"[ERROR] {utt_id}: {type(e).__name__}: {e}")
            self._chunk_cache[path] = None
            return utt_id, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label, chunk_idx = self.samples[idx]

        # Cache is always populated by __init__, so no disk I/O here
        chunks = self._chunk_cache.get(path)

        if chunks is None:
            # Defensive fallback — should not be reachable in normal usage
            waveform = torch.zeros(self.chunk_samples, dtype=torch.float32)
        else:
            waveform = torch.tensor(chunks[chunk_idx], dtype=torch.float32)

        return {
            "input_values": waveform,
            "labels": torch.tensor(label, dtype=torch.long),
        }
