import os
import warnings
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


# ── shared helpers ────────────────────────────────────────────────────────────

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file, convert to mono float32, resample to target_sr.

    Uses torchaudio as primary loader (wider codec support), soundfile as
    fallback. Returns silence on total failure so one bad file doesn't crash
    a 458k-sample eval run.
    """
    sr = None
    waveform_np = None

    try:
        waveform, sr = torchaudio.load(path)          # returns (C, T) tensor
        waveform_np = waveform.mean(dim=0).numpy()    # mono float32
    except Exception:
        pass

    if waveform_np is None:
        try:
            waveform_np, sr = sf.read(path, dtype="float32", always_2d=False)
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)
        except Exception:
            warnings.warn(f"Failed to load {path} — substituting silence.")
            return np.zeros(target_sr, dtype=np.float32)

    if sr != target_sr:
        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=target_sr)
    return waveform_np


def _rms_normalize(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > eps:
        return torch.clamp(waveform / (rms + eps), -1.0, 1.0)
    peak = torch.max(torch.abs(waveform))
    if peak > eps:
        return waveform / peak
    return waveform


def _trim_silence(
    waveform_np: np.ndarray,
    sr: int = 16000,
    top_db: float = 30.0,
    pad_ms: int = 100,
    frame_ms: int = 20,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Energy-based leading/trailing silence trim. Keeps frames within `top_db`
    of the peak frame RMS; re-pads `pad_ms` per side to preserve onsets.
    Returns the input unchanged if the whole clip is below threshold.
    """
    if waveform_np.size == 0:
        return waveform_np
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms   / 1000)
    if waveform_np.size < frame_len:
        return waveform_np
    n_frames = 1 + (waveform_np.size - frame_len) // hop_len
    if n_frames <= 0:
        return waveform_np
    idx = np.arange(n_frames) * hop_len
    frames = np.stack([waveform_np[i:i + frame_len] for i in idx])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    peak = rms.max()
    if peak <= 0:
        return waveform_np
    thresh = peak * (10.0 ** (-top_db / 20.0))
    active = np.where(rms >= thresh)[0]
    if active.size == 0:
        return waveform_np
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[active[0]]) - pad)
    end   = min(waveform_np.size, int(idx[active[-1]]) + frame_len + pad)
    return waveform_np[start:end]


def _crop_or_pad(
    waveform: torch.Tensor, max_length: int, is_train: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = waveform.shape[0]
    if seq_len >= max_length:
        start = (
            torch.randint(0, seq_len - max_length + 1, (1,)).item()
            if is_train
            else (seq_len - max_length) // 2
        )
        waveform = waveform[start : start + max_length]
        attn_mask = torch.ones(max_length, dtype=torch.long)
    else:
        pad = max_length - seq_len
        waveform = F.pad(waveform, (0, pad))
        attn_mask = torch.zeros(max_length, dtype=torch.long)
        attn_mask[:seq_len] = 1
    return waveform, attn_mask


def _make_item(
    waveform_np: np.ndarray,
    label: int,
    transform: Optional[Callable],
    max_length: int,
    is_train: bool,
) -> dict:
    if transform is not None:
        waveform_np = transform(waveform_np, 16000)
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    waveform = _rms_normalize(waveform)
    waveform, attn_mask = _crop_or_pad(waveform, max_length, is_train)
    return {
        "input_values":   waveform,
        "attention_mask": attn_mask,
        "labels":         torch.tensor(label, dtype=torch.long),
    }


# ── ASVspoof 2021 DF ──────────────────────────────────────────────────────────

class ASVspoof2021DFDataset(Dataset):
    """
    ASVspoof 2021 DeepFake eval set.

    Audio is spread across three part folders:
        asvspoof2021_DF/
          ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/
          ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac/
          ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac/
          trial_metadata.txt

    Protocol (trial_metadata.txt) format — space-separated:
        col 1 : utterance ID  → filename is {col1}.flac
        col 5 : label         → 'bonafide' or 'spoof'
    """

    _LABEL_MAP = {"bonafide": 0, "spoof": 1}
    _PART_DIRS = [
        "ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac",
        "ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac",
        "ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac",
    ]

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.max_length = int(16000 * max_seconds)
        self.transform = transform

        protocol_path = os.path.join(base_dir, "trial_metadata.txt")

        # Build a lookup: utterance_id → audio path
        audio_index: dict[str, str] = {}
        for part in self._PART_DIRS:
            flac_dir = os.path.join(base_dir, part)
            if not os.path.isdir(flac_dir):
                continue
            for fname in os.listdir(flac_dir):
                if fname.endswith(".flac"):
                    utt_id = fname[: -len(".flac")]
                    audio_index[utt_id] = os.path.join(flac_dir, fname)

        self.data = []
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                utt_id = parts[1]
                label_str = parts[5]
                if utt_id in audio_index and label_str in self._LABEL_MAP:
                    self.data.append({
                        "path":  audio_index[utt_id],
                        "label": self._LABEL_MAP[label_str],
                    })

        print(f"ASVspoof2021DF: {len(self.data)} utterances loaded "
              f"({sum(1 for d in self.data if d['label']==0)} bonafide, "
              f"{sum(1 for d in self.data if d['label']==1)} spoof)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        waveform_np = _load_audio(item["path"])
        return _make_item(waveform_np, item["label"], self.transform, self.max_length, is_train=False)


# ── WaveFake ──────────────────────────────────────────────────────────────────

class WaveFakeDataset(Dataset):
    """
    WaveFake — generated (spoof) audio only.

    All .wav files under generated_audio/ are spoof (label=1).
    No bonafide included; pair with ASVspoof2019LADataset bonafide via ConcatDataset.

    Structure:
        WaveFake/
          generated_audio/
            ljspeech_hifiGAN/          *.wav
            ljspeech_melgan/           *.wav
            ljspeech_waveglow/         *.wav
            ...                        (9 vocoder subfolders)
    """

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.max_length = int(16000 * max_seconds)
        self.transform = transform

        generated_dir = os.path.join(base_dir, "generated_audio")
        self.files = sorted(Path(generated_dir).rglob("*.wav"))

        if not self.files:
            raise FileNotFoundError(f"No .wav files found under {generated_dir}")

        print(f"WaveFake: {len(self.files)} spoof utterances loaded")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        waveform_np = _load_audio(str(self.files[idx]))
        return _make_item(waveform_np, 1, self.transform, self.max_length, is_train=True)


# ── MUSAN Speech ─────────────────────────────────────────────────────────────

class MUSANSpeechDataset(Dataset):
    """
    MUSAN speech subset — bonafide (label=0) only.

    Structure:
        musan/speech/
          librivox/   ANNOTATIONS  speech-librivox-NNNN.wav  ...
          us-gov/                  speech-us-gov-NNNN.wav    ...

    librivox ANNOTATIONS format (space-separated):
        filename  gender  language

    Only English librivox clips are included; all us-gov clips are English.
    """

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.max_length = int(16000 * max_seconds)
        self.transform = transform
        self.files: list[str] = []

        speech_dir = os.path.join(base_dir, "speech")

        # librivox — English-only via ANNOTATIONS
        librivox_dir = os.path.join(speech_dir, "librivox")
        annotations  = os.path.join(librivox_dir, "ANNOTATIONS")
        if os.path.isfile(annotations):
            with open(annotations) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[2].lower() == "english":
                        wav = os.path.join(librivox_dir, parts[0] + ".wav")
                        if os.path.isfile(wav):
                            self.files.append(wav)

        # us-gov — all files are English
        usgov_dir = os.path.join(speech_dir, "us-gov")
        if os.path.isdir(usgov_dir):
            for fname in sorted(os.listdir(usgov_dir)):
                if fname.endswith(".wav"):
                    self.files.append(os.path.join(usgov_dir, fname))

        if not self.files:
            raise FileNotFoundError(f"No MUSAN speech files found under {speech_dir}")

        print(f"MUSANSpeech: {len(self.files)} bonafide utterances loaded "
              f"({sum(1 for f in self.files if 'librivox' in f)} librivox-en, "
              f"{sum(1 for f in self.files if 'us-gov' in f)} us-gov)")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        waveform_np = _load_audio(self.files[idx])
        return _make_item(waveform_np, 0, self.transform, self.max_length, is_train=True)


# ── VoxCeleb1 ────────────────────────────────────────────────────────────────

class VoxCelebDataset(Dataset):
    """
    VoxCeleb1 dev — bonafide (label=0) only. Real-world YouTube recordings.

    Expected structure (after the official zip is extracted):
        voxceleb1/wav/wav/idXXXXX/<youtube_id>/NNNNN.wav

    Files are 16 kHz mono WAV already, so no resample is needed at load time.
    """

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.max_length = int(16000 * max_seconds)
        self.transform = transform

        # The official zip nests a redundant `wav/` inside the extract dir.
        # Accept either base_dir/wav/idXXXXX/... or base_dir/idXXXXX/...
        candidates = [
            os.path.join(base_dir, "wav", "wav"),
            os.path.join(base_dir, "wav"),
            base_dir,
        ]
        wav_root = next((c for c in candidates if os.path.isdir(c)
                         and any(d.startswith("id") for d in os.listdir(c))),
                        None)
        if wav_root is None:
            raise FileNotFoundError(
                f"No idXXXXX speaker dirs found under any of: {candidates}"
            )

        all_files = sorted(str(p) for p in Path(wav_root).rglob("*.wav"))
        if not all_files:
            raise FileNotFoundError(f"No .wav files found under {wav_root}")

        if max_samples is not None and max_samples < len(all_files):
            # Stratify by speaker so the subsample stays diverse — pick files
            # round-robin across speakers using a seeded RNG for reproducibility.
            rng = np.random.default_rng(42)
            by_speaker: dict[str, list[str]] = {}
            for p in all_files:
                spk = Path(p).parent.parent.name
                by_speaker.setdefault(spk, []).append(p)
            for spk in by_speaker:
                rng.shuffle(by_speaker[spk])
            picked: list[str] = []
            speakers = list(by_speaker.keys())
            i = 0
            while len(picked) < max_samples:
                spk = speakers[i % len(speakers)]
                if by_speaker[spk]:
                    picked.append(by_speaker[spk].pop())
                i += 1
            self.files = picked
        else:
            self.files = all_files

        n_speakers = len({Path(p).parent.parent.name for p in self.files})
        print(f"VoxCeleb1: {len(self.files)} bonafide utterances "
              f"across {n_speakers} speakers"
              + (f" (subsampled from {len(all_files)})" if max_samples and max_samples < len(all_files) else ""))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        waveform_np = _load_audio(self.files[idx])
        return _make_item(waveform_np, 0, self.transform, self.max_length, is_train=True)


# ── CommonVoice (English) ────────────────────────────────────────────────────

class CommonVoiceDataset(Dataset):
    """
    Mozilla CommonVoice English — bonafide (label=0) only. Consumer-mic
    recordings (laptop, phone, headset), broad speaker coverage.

    Expected structure (produced by tools/download_commonvoice.py):
        commonvoice/
          manifest.tsv                            (path, speaker_id, sentence, duration_s, source_shard)
          clips/<speaker_id>/<clip_id>.wav        (16 kHz mono PCM_16)

    Files are already 16 kHz mono so no resample at load time.
    """

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.max_length = int(16000 * max_seconds)
        self.transform = transform

        manifest_path = os.path.join(base_dir, "manifest.tsv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"CommonVoice manifest not found: {manifest_path}")

        rows: list[tuple[str, str]] = []  # (abs_path, speaker_id)
        with open(manifest_path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            path_i = header.index("path")
            spk_i  = header.index("speaker_id")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(path_i, spk_i):
                    continue
                abs_path = os.path.join(base_dir, parts[path_i])
                rows.append((abs_path, parts[spk_i]))

        if not rows:
            raise FileNotFoundError(f"No rows in CommonVoice manifest: {manifest_path}")

        if max_samples is not None and max_samples < len(rows):
            # Stratify by speaker — round-robin across speakers with a seeded RNG.
            rng = np.random.default_rng(42)
            by_speaker: dict[str, list[str]] = {}
            for p, spk in rows:
                by_speaker.setdefault(spk, []).append(p)
            for spk in by_speaker:
                rng.shuffle(by_speaker[spk])
            picked: list[str] = []
            speakers = list(by_speaker.keys())
            i = 0
            while len(picked) < max_samples and any(by_speaker.values()):
                spk = speakers[i % len(speakers)]
                if by_speaker[spk]:
                    picked.append(by_speaker[spk].pop())
                i += 1
            self.files = picked
        else:
            self.files = [p for p, _ in rows]

        n_speakers_used = len({Path(p).parent.name for p in self.files})
        print(f"CommonVoice: {len(self.files)} bonafide utterances "
              f"across {n_speakers_used} speakers"
              + (f" (subsampled from {len(rows)})" if max_samples and max_samples < len(rows) else ""))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        waveform_np = _load_audio(self.files[idx])
        # Trim leading/trailing silence — CV clips often have several seconds
        # of room tone before speech, which makes random crops land on silence
        # and creates a "leading silence → bonafide" shortcut vs WaveFake.
        waveform_np = _trim_silence(waveform_np)
        return _make_item(waveform_np, 0, self.transform, self.max_length, is_train=True)


# ── In-the-Wild ───────────────────────────────────────────────────────────────

class InTheWildDataset(Dataset):
    """
    In-the-Wild deepfake detection dataset.

    Label is inferred from subfolder name:
        in_the_wild/real/   → bonafide (0)
        in_the_wild/fake/   → spoof    (1)

    Usage: eval only — never train or tune thresholds on this.
    """

    _LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(
        self,
        base_dir: str,
        max_seconds: float = 4.0,
    ):
        self.max_length = int(16000 * max_seconds)

        self.data = []
        for folder, label in self._LABEL_MAP.items():
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Expected subfolder not found: {folder_path}")
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".wav") or fname.endswith(".flac"):
                    self.data.append({
                        "path":  os.path.join(folder_path, fname),
                        "label": label,
                    })

        bonafide = sum(1 for d in self.data if d["label"] == 0)
        spoof    = sum(1 for d in self.data if d["label"] == 1)
        print(f"InTheWild: {len(self.data)} utterances "
              f"({bonafide} real, {spoof} fake)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        waveform_np = _load_audio(item["path"])
        # No transform — eval set stays clean
        return _make_item(waveform_np, item["label"], None, self.max_length, is_train=False)
