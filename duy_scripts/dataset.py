import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from torch.utils.data import Dataset

from preprocessing import _vad_filter, _chunk_audio


class ASVspoof2019LADataset(Dataset):
    def __init__(self, base_dir, split="train", use_vad=False, vad_mode=2, overlap_pct=0, chunk_samples = 8000):
        """
        Args:
            base_dir (str): Path to the root 'ASVspoof2019_LA' directory.
            split (str): One of 'train', 'dev', or 'eval'.
            use_vad (bool): Whether to filter silence via WebRTC VAD.
            vad_mode (int): VAD aggressiveness (0=least, 3=most aggressive).
            overlap_pct (int): Overlap between chunks as a percentage (0-99).
        """
        self.base_dir = base_dir
        self.split = split
        self.use_vad = use_vad
        self.vad_mode = vad_mode
        self.hop_samples = max(1, int(chunk_samples * (1 - overlap_pct / 100)))
        self.chunk_samples = chunk_samples

        split_map = {
            "train": ("train", "ASVspoof2019.LA.cm.train.trn.txt"),
            "dev":   ("dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
            "eval":  ("eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
        }
        audio_dir_name, protocol_name = split_map[split]
        self.audio_dir = os.path.join("../output/", audio_dir_name)
        protocol_path = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols", protocol_name)
        self.label_map = {"bonafide": 0, "spoof": 1}
        self._chunks_cache = {}

        # Parse protocol file
        raw_data = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    raw_data.append({
                        "path": os.path.join(self.audio_dir, f"{parts[1]}.flac"),
                        "label": self.label_map[parts[4]],
                    })

        # Expand each file into (path, label, chunk_index) entries
        # Files that fail to load are skipped entirely
        print(f"Expanding {len(raw_data)} files into chunks...")
        self.samples = []
        for item in raw_data:
            utt_id, chunks = self._load_chunks(item["path"])
            self._chunk_cache[item["path"]] = chunks
            if chunks is None:
                print(f"[SKIPPED] {utt_id}: failed to load or produced no chunks")
                continue
            for i in range(len(chunks)):
                self.samples.append((item["path"], item["label"], i))

        print(f"Total chunks: {len(self.samples)}")

    def _load_chunks(self, path: str) -> tuple[str, list | None]:
        """
        Load a FLAC file, preprocess, and split into chunks.

        Returns:
            (utt_id, chunks) where chunks is a list of np.ndarray,
            or (utt_id, None) on any failure or if no chunks were produced.
        """
        utt_id = os.path.splitext(os.path.basename(path))[0]

        if path in self._chunks_cache:
            return utt_id, self._chunks_cache[path]

        try:
            waveform_np, sr = sf.read(path)

            # 1. Force mono
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)

            # 2. Resample to 16kHz
            if sr != 16000:
                waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=16000)

            if len(waveform_np) == 0:
                return utt_id, None

            # 3. Amplitude normalization [-1, 1]
            max_val = np.max(np.abs(waveform_np))
            if max_val > 0:
                waveform_np = waveform_np / max_val

            # 4. Optional VAD filtering
            if self.use_vad:
                waveform_np = _vad_filter(waveform_np, 16000, self.vad_mode)
                if len(waveform_np) == 0:
                    return utt_id, None

            # 5. Chunk with sliding window
            chunks = _chunk_audio(waveform_np, self.hop_samples)
            if not chunks:
                return utt_id, None

            self._chunks_cache[path] = chunks
            return utt_id, chunks

        except Exception as e:
            print(f"[ERROR] {utt_id}: {type(e).__name__}: {e}", flush=True)
            return utt_id, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label, chunk_idx = self.samples[idx]
        utt_id, chunks = self._load_chunks(path)

        if chunks is None:
            # Shouldn't happen since failed files are filtered in __init__,
            # but handles the edge case of a cache eviction or file disappearing
            waveform = torch.zeros(self.chunk_samples, dtype=torch.float32)
        else:
            waveform = torch.tensor(chunks[chunk_idx], dtype=torch.float32)

        return {
            "input_values": waveform,
            "labels": torch.tensor(label, dtype=torch.long),
        }