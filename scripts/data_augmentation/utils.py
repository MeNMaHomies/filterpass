import numpy as np


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)) + 1e-9)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so signal-to-noise ratio equals `snr_db`, then mix."""
    target_noise_rms = rms(signal) / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / rms(noise))
    return np.clip(signal + noise_scaled, -1.0, 1.0)


def match_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Tile or trim `noise` to exactly `target_len` samples."""
    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    return noise[:target_len]
