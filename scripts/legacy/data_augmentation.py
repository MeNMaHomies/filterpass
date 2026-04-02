"""
Audio Augmentation Pipeline
============================
Techniques:
  1. Noise-based augmentation  — white, pink, brown, environmental background noise
  2. Channel / device simulation — room impulse responses (RIR), codec distortion, mic coloration
  3. Time-domain augmentation   — packet-loss / dropout simulation, clipping, time-stretch

Input / output: FLAC files (16-bit, any sample-rate).

Dependencies:
    pip install soundfile numpy scipy pyroomacoustics audiomentations
"""

import os
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf
import scipy.signal as sps

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import pyroomacoustics as pra
    HAS_PRA = True
except ImportError:
    HAS_PRA = False
    print("[warn] pyroomacoustics not installed – RIR simulation disabled.")

try:
    from audiomentations import (
        Compose, AddGaussianNoise, AddBackgroundNoise,
        TimeStretch, PitchShift, Gain,
    )
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("[warn] audiomentations not installed – some augmentations use numpy fallbacks.")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    # ── noise ──────────────────────────────────────────────────────────────
    add_white_noise: bool = True
    white_noise_snr_db: tuple[float, float] = (10.0, 30.0)   # (min, max) SNR in dB

    add_pink_noise: bool = True
    pink_noise_snr_db: tuple[float, float] = (10.0, 25.0)

    add_brown_noise: bool = False
    brown_noise_snr_db: tuple[float, float] = (10.0, 20.0)

    background_noise_dir: Optional[str] = None               # folder of .wav/.flac noise files
    background_noise_snr_db: tuple[float, float] = (5.0, 20.0)

    # ── channel / device ───────────────────────────────────────────────────
    simulate_rir: bool = True                                 # room impulse response via pyroomacoustics
    room_dim_range: tuple[float, float] = (3.0, 10.0)        # room side length in metres
    rt60_range: tuple[float, float] = (0.2, 0.8)             # reverberation time in seconds

    apply_bandpass: bool = True                              # telephone / codec bandwidth limit
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3400.0

    apply_mic_coloration: bool = True                        # mild EQ bump to simulate cheap mic
    mic_coloration_gain_db: float = 3.0
    mic_coloration_freq_hz: float = 1500.0

    apply_codec_distortion: bool = True                      # mild soft-clip (mp3/codec artefact feel)
    codec_clip_threshold: float = 0.85                       # 0–1

    # ── time-domain / packet-loss ──────────────────────────────────────────
    simulate_packet_loss: bool = True
    packet_loss_rate: float = 0.05                           # fraction of 20-ms packets dropped
    packet_loss_fill: str = "silence"                        # "silence" | "repeat" | "noise"

    apply_time_stretch: bool = False                         # slow down / speed up
    time_stretch_range: tuple[float, float] = (0.9, 1.1)

    apply_random_gain: bool = True
    gain_db_range: tuple[float, float] = (-6.0, 6.0)

    # ── output ─────────────────────────────────────────────────────────────
    output_subtype: str = "PCM_16"                           # FLAC subtype


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)) + 1e-9)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so the signal-to-noise ratio equals `snr_db`, then add."""
    target_noise_rms = rms(signal) / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / rms(noise))
    return np.clip(signal + noise_scaled, -1.0, 1.0)


def match_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Tile or trim noise to exactly `target_len` samples."""
    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    return noise[:target_len]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Noise-based augmentation
# ─────────────────────────────────────────────────────────────────────────────

def generate_white_noise(n: int) -> np.ndarray:
    return np.random.randn(n).astype(np.float32)


def generate_pink_noise(n: int) -> np.ndarray:
    """Voss-McCartney pink noise approximation."""
    cols = 16
    array = np.full((n, cols), np.nan)
    array[0] = np.random.randn(cols)
    for i in range(1, n):
        j = int(np.log2(i & -i)) if i & (i - 1) else 0
        j = min(j, cols - 1)
        array[i] = array[i - 1]
        array[i, j] = np.random.randn()
    pink = np.nansum(array, axis=1).astype(np.float32)
    return pink / (pink.std() + 1e-9)


def generate_brown_noise(n: int) -> np.ndarray:
    """Brown/red noise: cumulative sum of white noise."""
    white = np.random.randn(n).astype(np.float32)
    brown = np.cumsum(white)
    brown -= brown.mean()
    return (brown / (brown.std() + 1e-9)).astype(np.float32)


def augment_noise(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    n = len(audio)

    if cfg.add_white_noise:
        snr = random.uniform(*cfg.white_noise_snr_db)
        audio = mix_at_snr(audio, generate_white_noise(n), snr)

    if cfg.add_pink_noise:
        snr = random.uniform(*cfg.pink_noise_snr_db)
        audio = mix_at_snr(audio, generate_pink_noise(n), snr)

    if cfg.add_brown_noise:
        snr = random.uniform(*cfg.brown_noise_snr_db)
        audio = mix_at_snr(audio, generate_brown_noise(n), snr)

    if cfg.background_noise_dir:
        noise_files = list(Path(cfg.background_noise_dir).glob("**/*.flac")) + \
                      list(Path(cfg.background_noise_dir).glob("**/*.wav"))
        if noise_files:
            nf = random.choice(noise_files)
            bg, bg_sr = sf.read(str(nf), dtype="float32", always_2d=False)
            if bg.ndim > 1:
                bg = bg.mean(axis=1)
            if bg_sr != sr:
                num = int(len(bg) * sr / bg_sr)
                bg = sps.resample(bg, num).astype(np.float32)
            bg = match_length(bg, n)
            snr = random.uniform(*cfg.background_noise_snr_db)
            audio = mix_at_snr(audio, bg, snr)

    return audio


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel & device simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_rir(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    """Simulate a random shoe-box room using pyroomacoustics."""
    if not HAS_PRA:
        return audio

    dim = [random.uniform(*cfg.room_dim_range) for _ in range(3)]
    rt60 = random.uniform(*cfg.rt60_range)

    try:
        e_absorption, max_order = pra.inverse_sabine(rt60, dim)
    except Exception:
        return audio  # room too large / small for given rt60

    materials = pra.Material(e_absorption)
    room = pra.ShoeBox(dim, fs=sr, materials=materials, max_order=max_order)

    # Place source and mic away from walls
    src = [d * random.uniform(0.2, 0.8) for d in dim]
    mic = [d * random.uniform(0.2, 0.8) for d in dim]
    room.add_source(src, signal=audio)
    mic_array = np.array(mic).reshape(3, 1)
    room.add_microphone(mic_array)
    room.simulate()

    out = room.mic_array.signals[0].astype(np.float32)
    # Trim or pad to original length
    if len(out) > len(audio):
        out = out[: len(audio)]
    else:
        out = np.pad(out, (0, len(audio) - len(out)))
    return np.clip(out, -1.0, 1.0)


def apply_bandpass_filter(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    """Simulate telephone / narrowband codec frequency response."""
    lo = cfg.bandpass_low_hz / (sr / 2.0)
    hi = cfg.bandpass_high_hz / (sr / 2.0)
    lo = np.clip(lo, 1e-4, 0.9999)
    hi = np.clip(hi, lo + 1e-4, 0.9999)
    b, a = sps.butter(4, [lo, hi], btype="band")
    return sps.lfilter(b, a, audio).astype(np.float32)


def apply_mic_coloration(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    """Boost a mid-frequency band to mimic cheap microphone resonance."""
    freq = cfg.mic_coloration_freq_hz / (sr / 2.0)
    freq = np.clip(freq, 1e-4, 0.9999)
    gain_linear = 10 ** (cfg.mic_coloration_gain_db / 20.0)
    b, a = sps.butter(2, freq, btype="low")
    low = sps.lfilter(b, a, audio).astype(np.float32)
    return np.clip(audio + (gain_linear - 1.0) * low, -1.0, 1.0)


def apply_codec_distortion(audio: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    """Soft-clip to emulate codec saturation / compression artefacts."""
    thr = cfg.codec_clip_threshold
    audio = np.where(
        np.abs(audio) <= thr,
        audio,
        np.sign(audio) * (thr + (1.0 - thr) * np.tanh((np.abs(audio) - thr) / (1.0 - thr + 1e-9))),
    )
    return audio.astype(np.float32)


def augment_channel(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if cfg.simulate_rir:
        audio = simulate_rir(audio, sr, cfg)
    if cfg.apply_bandpass:
        audio = apply_bandpass_filter(audio, sr, cfg)
    if cfg.apply_mic_coloration:
        audio = apply_mic_coloration(audio, sr, cfg)
    if cfg.apply_codec_distortion:
        audio = apply_codec_distortion(audio, cfg)
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Time-domain augmentation (packet loss, time stretch, gain)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_packet_loss(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    """
    VoIP-style packet loss: split into 20-ms frames; randomly drop `packet_loss_rate`
    fraction of frames and fill them with silence, the previous frame (repeat), or noise.
    """
    frame_samples = int(0.020 * sr)  # 20 ms
    n = len(audio)
    out = audio.copy()

    for start in range(0, n, frame_samples):
        if random.random() < cfg.packet_loss_rate:
            end = min(start + frame_samples, n)
            size = end - start
            if cfg.packet_loss_fill == "silence":
                out[start:end] = 0.0
            elif cfg.packet_loss_fill == "repeat" and start > 0:
                prev_start = max(0, start - frame_samples)
                prev = out[prev_start:start]
                out[start:end] = match_length(prev, size)
            else:  # noise
                out[start:end] = np.random.randn(size).astype(np.float32) * 0.01

    return out


def apply_time_stretch_numpy(audio: np.ndarray, rate: float) -> np.ndarray:
    """Simple time-stretch via linear resampling (no pitch correction)."""
    orig_len = len(audio)
    new_len = int(orig_len / rate)
    stretched = sps.resample(audio, new_len).astype(np.float32)
    # Trim or pad to original length to keep consistent shape
    if len(stretched) > orig_len:
        return stretched[:orig_len]
    return np.pad(stretched, (0, orig_len - len(stretched)))


def apply_random_gain_fn(audio: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    gain_db = random.uniform(*cfg.gain_db_range)
    gain_linear = 10 ** (gain_db / 20.0)
    return np.clip(audio * gain_linear, -1.0, 1.0)


def augment_time_domain(audio: np.ndarray, sr: int, cfg: AugmentationConfig) -> np.ndarray:
    if cfg.simulate_packet_loss:
        audio = simulate_packet_loss(audio, sr, cfg)
    if cfg.apply_time_stretch:
        rate = random.uniform(*cfg.time_stretch_range)
        audio = apply_time_stretch_numpy(audio, rate)
    if cfg.apply_random_gain:
        audio = apply_random_gain_fn(audio, cfg)
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment_file(
    input_path: str,
    output_path: str,
    cfg: AugmentationConfig,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)

    # Handle stereo: work on mono internally, restore stereo at end
    stereo = audio.ndim == 2
    if stereo:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]
    else:
        channels = [audio]

    augmented_channels = []
    for ch_audio in channels:
        ch_audio = augment_noise(ch_audio, sr, cfg)
        ch_audio = augment_channel(ch_audio, sr, cfg)
        ch_audio = augment_time_domain(ch_audio, sr, cfg)
        augmented_channels.append(ch_audio)

    if stereo:
        out_audio = np.stack(augmented_channels, axis=1)
    else:
        out_audio = augmented_channels[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out_audio, sr, format="FLAC", subtype=cfg.output_subtype)
    print(f"  ✓  {input_path}  →  {output_path}")


def augment_directory(
    input_dir: str,
    output_dir: str,
    cfg: AugmentationConfig,
    n_augmentations: int = 1,
) -> None:
    input_files = sorted(Path(input_dir).rglob("*.flac"))
    if not input_files:
        print(f"[warn] No .flac files found in {input_dir}")
        return

    for flac_file in input_files:
        rel = flac_file.relative_to(input_dir)
        for i in range(n_augmentations):
            stem = rel.stem + (f"_aug{i+1}" if n_augmentations > 1 else "_aug")
            out_path = Path(output_dir) / rel.parent / (stem + ".flac")
            augment_file(str(flac_file), str(out_path), cfg, seed=i)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FLAC audio augmentation: noise, RIR, packet-loss simulation."
    )
    p.add_argument("input",  help="Input .flac file OR directory of .flac files")
    p.add_argument("output", help="Output .flac file OR output directory")
    p.add_argument("--n-aug", type=int, default=1,
                   help="Number of augmented copies per file (directory mode only)")
    p.add_argument("--bg-noise-dir", default=None,
                   help="Directory of background noise files (.wav / .flac)")

    # Toggle flags
    p.add_argument("--no-white-noise",    action="store_true")
    p.add_argument("--no-pink-noise",     action="store_true")
    p.add_argument("--add-brown-noise",   action="store_true")
    p.add_argument("--no-rir",            action="store_true")
    p.add_argument("--no-bandpass",       action="store_true")
    p.add_argument("--no-mic-color",      action="store_true")
    p.add_argument("--no-codec-dist",     action="store_true")
    p.add_argument("--no-packet-loss",    action="store_true")
    p.add_argument("--packet-fill",       choices=["silence", "repeat", "noise"],
                   default="silence")
    p.add_argument("--packet-loss-rate",  type=float, default=0.05)
    p.add_argument("--add-time-stretch",  action="store_true")
    p.add_argument("--no-gain",           action="store_true")
    p.add_argument("--seed",              type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = AugmentationConfig(
        add_white_noise       = not args.no_white_noise,
        add_pink_noise        = not args.no_pink_noise,
        add_brown_noise       = args.add_brown_noise,
        background_noise_dir  = args.bg_noise_dir,
        simulate_rir          = not args.no_rir,
        apply_bandpass        = not args.no_bandpass,
        apply_mic_coloration  = not args.no_mic_color,
        apply_codec_distortion= not args.no_codec_dist,
        simulate_packet_loss  = not args.no_packet_loss,
        packet_loss_fill      = args.packet_fill,
        packet_loss_rate      = args.packet_loss_rate,
        apply_time_stretch    = args.add_time_stretch,
        apply_random_gain     = not args.no_gain,
    )

    inp = Path(args.input)
    out = Path(args.output)

    if inp.is_dir():
        augment_directory(str(inp), str(out), cfg, n_augmentations=args.n_aug)
    elif inp.is_file() and inp.suffix.lower() == ".flac":
        augment_file(str(inp), str(out), cfg, seed=args.seed)
    else:
        print(f"[error] '{inp}' is not a .flac file or directory.")


if __name__ == "__main__":
    main()