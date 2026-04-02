from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AugmentationConfig:
    # ── noise ──────────────────────────────────────────────────────────────
    add_white_noise: bool = True
    white_noise_snr_db: tuple[float, float] = (10.0, 30.0)

    add_pink_noise: bool = True
    pink_noise_snr_db: tuple[float, float] = (10.0, 25.0)

    add_brown_noise: bool = False
    brown_noise_snr_db: tuple[float, float] = (10.0, 20.0)

    background_noise_dir: Optional[str] = None
    background_noise_snr_db: tuple[float, float] = (5.0, 20.0)

    # ── channel / device ───────────────────────────────────────────────────
    simulate_rir: bool = True
    room_dim_range: tuple[float, float] = (3.0, 10.0)
    rt60_range: tuple[float, float] = (0.2, 0.8)

    apply_bandpass: bool = True
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3400.0

    apply_mic_coloration: bool = True
    mic_coloration_gain_db: float = 3.0
    mic_coloration_freq_hz: float = 1500.0

    apply_codec_distortion: bool = True
    codec_clip_threshold: float = 0.85

    # ── time-domain / packet-loss ──────────────────────────────────────────
    simulate_packet_loss: bool = True
    packet_loss_rate: float = 0.05
    packet_loss_fill: str = "silence"   # "silence" | "repeat" | "noise"

    apply_time_stretch: bool = False
    time_stretch_range: tuple[float, float] = (0.9, 1.1)

    apply_random_gain: bool = True
    gain_db_range: tuple[float, float] = (-6.0, 6.0)

    # ── output ─────────────────────────────────────────────────────────────
    output_subtype: str = "PCM_16"
