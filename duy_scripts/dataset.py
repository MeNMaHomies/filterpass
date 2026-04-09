import os
import math
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset
import torch.nn.functional as F

# from preprocessing import _vad_filter

class ASVspoof2019LADataset(Dataset):
    def __init__(self, base_dir, split="train", max_seconds=4.0):
        """
        Args:
            base_dir (str): Path to the root 'ASVspoof2019_LA' directory.
            split (str): One of 'train', 'dev', or 'eval'.
            max_seconds (float): Maximum length of audio in seconds.
        """
        self.base_dir = base_dir
        self.split = split
        self.max_length = int(16000 * max_seconds)  # Wav2Vec2 strictly expects 16kHz
        self.is_train = (split == "train")

        split_map = {
            "train": ("train/flac", "train/keys/ASVspoof2019.LA.cm.train.augmented.txt"),
            "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
            "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"),
        }

        audio_dir_name, protocol_name = split_map[split]
        self.audio_dir = os.path.join(base_dir, audio_dir_name)
        protocol_path = os.path.join(base_dir, protocol_name)

        print(f"Reading folder: {self.audio_dir}")

        self.data = []
        self.label_map = {"bonafide": 0, "spoof": 1}

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    filename = parts[1]
                    label_str = parts[4]
                    self.data.append({
                        "path": os.path.join(self.audio_dir, f"{filename}.flac"),
                        "label": self.label_map[label_str]
                    })

    def _random_crop_or_pad(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For sequences longer than max_length:
          - Training: random crop (data augmentation, exposes more signal)
          - Dev/Eval: centre crop (deterministic, reproducible)
        For shorter sequences: right-pad with zeros and return an attention mask.

        Returns:
            waveform     (T,)  — fixed-length waveform
            attn_mask    (T,)  — 1 where real signal, 0 where padding
        """
        seq_len = waveform.shape[0]

        if seq_len >= self.max_length:
            if self.is_train:
                start = torch.randint(0, seq_len - self.max_length + 1, (1,)).item()
            else:
                start = (seq_len - self.max_length) // 2  # centre crop
            waveform = waveform[start: start + self.max_length]
            attn_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            pad_amount = self.max_length - seq_len
            waveform = F.pad(waveform, (0, pad_amount))
            attn_mask = torch.zeros(self.max_length, dtype=torch.long)
            attn_mask[:seq_len] = 1

        return waveform, attn_mask

    @staticmethod
    def _normalize(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        RMS normalization — more robust than peak normalization because it
        targets perceived loudness rather than a single outlier sample.
        Falls back to peak norm if RMS is near zero (silence / near-silence).
        """
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > eps:
            waveform = waveform / (rms + eps)
            # Soft-clip to [-1, 1] so occasional loud peaks don't saturate
            waveform = torch.clamp(waveform, -1.0, 1.0)
        else:
            # Near-silent clip — fall back to peak norm
            peak = torch.max(torch.abs(waveform))
            if peak > eps:
                waveform = waveform / peak
        return waveform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 0. Load audio
        waveform_np, sample_rate = sf.read(item["path"])
        label = torch.tensor(item["label"], dtype=torch.long)

        # 1. Convert to mono (before resampling — cheaper)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)

        # 2. Resample to 16 kHz if needed
        if sample_rate != 16000:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)

        # 3. Numpy → Tensor
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        # 4. RMS normalise
        waveform = self._normalize(waveform)

        # 5. Crop / pad + build attention mask
        waveform, attention_mask = self._random_crop_or_pad(waveform)

        return {
            "input_values":    waveform,         # (T,)
            "attention_mask":  attention_mask,   # (T,)  ← pass to Wav2Vec2
            "labels":          label,            # scalar
        }

# class ASVspoof2019LADataset(Dataset):
#     def __init__(
#         self,
#         base_dir,
#         split="train",
#         use_vad=False,
#         vad_mode=2,
#         overlap_pct=0,
#         chunk_samples=32000,
#         pad_mode="constant",
#         drop_last_chunk=False,
#     ):
#         """
#         Chunk-indexed, memory-efficient dataset.

#         Key idea:
#         - Build a lightweight index of chunks in __init__
#         - Do NOT store waveform chunks in RAM
#         - Load and process the file inside __getitem__
#         - Return exactly one indexed chunk per sample

#         Args:
#             base_dir (str): Root path to ASVspoof2019_LA
#             split (str): 'train', 'dev', or 'eval'
#             use_vad (bool): Apply VAD before chunking
#             vad_mode (int): VAD aggressiveness (0-3)
#             overlap_pct (int): Overlap percentage between chunks (0-99)
#             chunk_samples (int): Chunk size in samples
#             pad_mode (str): np.pad mode, usually "constant"
#             drop_last_chunk (bool): If True, drop incomplete last chunk instead of padding
#         """
#         self.base_dir = base_dir
#         self.split = split
#         self.use_vad = use_vad
#         self.vad_mode = vad_mode
#         self.chunk_samples = chunk_samples
#         self.pad_mode = pad_mode
#         self.drop_last_chunk = drop_last_chunk

#         if not (0 <= overlap_pct < 100):
#             raise ValueError("overlap_pct must be between 0 and 99")

#         self.hop_samples = max(1, int(chunk_samples * (1 - overlap_pct / 100)))

#         split_map = {
#             "train": ("train/flac", "train/keys/ASVspoof2019.LA.cm.train.augmented.txt"),
#             "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
#             "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019.LA.cm.eval.trl.txt"),
#         }

#         # split_map = {
#         #     "train": ("ASVspoof2019_LA_train/flac", "ASVspoof2019.LA.cm.train.trn.txt"),
#         #     "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019.LA.cm.dev.trl.txt"),
#         #     "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019.LA.cm.eval.trl.txt"),
#         # }

#         if split not in split_map:
#             raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_map.keys())}")

#         audio_dir_name, protocol_name = split_map[split]
#         self.audio_dir = os.path.join(base_dir, audio_dir_name)
#         protocol_path = os.path.join(base_dir, protocol_name)
#         self.label_map = {"bonafide": 0, "spoof": 1}


#         # file-level metadata
#         self.files = []
#         # chunk-level index: each entry points to one chunk in one file
#         self.samples = []

#         raw_data = []
#         with open(protocol_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 5:
#                     raw_data.append(
#                         {
#                             "utt_id": parts[1],
#                             "path": os.path.join(self.audio_dir, f"{parts[1]}.flac"),
#                             "label": self.label_map[parts[4]],
#                         }
#                     )

#         print(f"Building chunk index for {len(raw_data)} files from {self.audio_dir}...")

#         skipped = 0

#         for file_idx, item in enumerate(raw_data):
#             path = item["path"]
#             utt_id = item["utt_id"]
#             label = item["label"]

#             try:
#                 info = sf.info(path)
#                 num_frames = int(info.frames)
#                 sr = int(info.samplerate)

#                 if num_frames <= 0 or sr <= 0:
#                     skipped += 1
#                     print(f"[SKIPPED] {utt_id}: invalid audio metadata")
#                     continue

#                 # Estimate length after resampling to 16k
#                 if sr != 16000:
#                     est_len_16k = int(round(num_frames * 16000 / sr))
#                 else:
#                     est_len_16k = num_frames

#                 # We store only metadata, not audio
#                 file_record = {
#                     "utt_id": utt_id,
#                     "path": path,
#                     "label": label,
#                     "orig_sr": sr,
#                     "orig_num_frames": num_frames,
#                     "est_len_16k": est_len_16k,
#                 }
#                 self.files.append(file_record)
#                 current_file_index = len(self.files) - 1

#                 num_chunks = self._estimate_num_chunks(est_len_16k)

#                 if num_chunks == 0:
#                     skipped += 1
#                     print(f"[SKIPPED] {utt_id}: no chunks could be indexed")
#                     continue

#                 for chunk_idx in range(num_chunks):
#                     start = chunk_idx * self.hop_samples
#                     self.samples.append((current_file_index, start))

#             except Exception as e:
#                 skipped += 1
#                 print(f"[SKIPPED] {utt_id}: {type(e).__name__}: {e}")

#         print(f"Indexed {len(self.samples)} chunks across {len(self.files)} files")
#         print(f"Skipped files: {skipped}")

#     def _estimate_num_chunks(self, signal_len: int) -> int:
#         """
#         Estimate number of chunks from waveform length after resampling.
#         This is only for building the index; actual slicing happens in __getitem__.
#         """
#         if signal_len <= 0:
#             return 0

#         if signal_len < self.chunk_samples:
#             return 0 if self.drop_last_chunk else 1

#         if self.drop_last_chunk:
#             return 1 + (signal_len - self.chunk_samples) // self.hop_samples

#         # include tail chunk that may need padding
#         return 1 + math.ceil((signal_len - self.chunk_samples) / self.hop_samples)

#     def __len__(self):
#         return len(self.samples)

#     def _load_audio(self, path: str) -> np.ndarray:
#         """
#         Load and preprocess one file.
#         """
#         waveform_np, sr = sf.read(path)

#         # force mono
#         if waveform_np.ndim > 1:
#             waveform_np = waveform_np.mean(axis=1)

#         # resample to 16k
#         if sr != 16000:
#             waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=16000)

#         waveform_np = waveform_np.astype(np.float32, copy=False)

#         if len(waveform_np) == 0:
#             return np.zeros(self.chunk_samples, dtype=np.float32)

#         # normalize to [-1, 1]
#         max_val = np.max(np.abs(waveform_np))
#         if max_val > 0:
#             waveform_np = waveform_np / max_val

#         if self.use_vad:
#             waveform_np = _vad_filter(waveform_np, 16000, self.vad_mode)
#             if len(waveform_np) == 0:
#                 return np.zeros(self.chunk_samples, dtype=np.float32)

#         return waveform_np

#     def _slice_chunk(self, waveform_np: np.ndarray, start: int) -> np.ndarray:
#         """
#         Slice one chunk at the indexed start position.
#         Pads tail chunks if needed, unless drop_last_chunk=True.
#         """
#         end = start + self.chunk_samples
#         n = len(waveform_np)

#         if start >= n:
#             # defensive fallback in case VAD changed length a lot
#             return np.zeros(self.chunk_samples, dtype=np.float32)

#         chunk = waveform_np[start:end]

#         if len(chunk) == self.chunk_samples:
#             return chunk.astype(np.float32, copy=False)

#         if self.drop_last_chunk:
#             return np.zeros(self.chunk_samples, dtype=np.float32)

#         pad_amount = self.chunk_samples - len(chunk)
#         chunk = np.pad(chunk, (0, pad_amount), mode=self.pad_mode)
#         return chunk.astype(np.float32, copy=False)

#     def __getitem__(self, idx):
#         file_index, start = self.samples[idx]
#         file_meta = self.files[file_index]

#         path = file_meta["path"]
#         label = file_meta["label"]
#         utt_id = file_meta["utt_id"]

#         try:
#             waveform_np = self._load_audio(path)
#             chunk_np = self._slice_chunk(waveform_np, start)
#         except Exception as e:
#             print(f"[ERROR] {utt_id}: {type(e).__name__}: {e}", flush=True)
#             chunk_np = np.zeros(self.chunk_samples, dtype=np.float32)

#         waveform = torch.from_numpy(chunk_np)

#         return {
#             "input_values": waveform,
#             "labels": torch.tensor(label, dtype=torch.long),
#             "utt_id": utt_id,
#         }