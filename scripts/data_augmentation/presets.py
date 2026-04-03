"""
Factory functions that return ready-to-use augmentation pipelines.

Each preset returns a Compose that can be called directly:
    transform = telephony()
    augmented = transform(audio, sr)
"""

from .augmentations import (
    BandpassFilter,
    BrownNoise,
    CodecDistortion,
    MicColoration,
    PacketLoss,
    PinkNoise,
    RandomGain,
    RoomImpulseResponse,
    TimeStretch,
    WhiteNoise,
)
from .compose import Compose, OneOf, SomeOf

# ── ASVspoof 2021 codec conditions ──────────────────────────────────────────
# C2: a-law / VoIP        → voip()
# C3: u-law / PSTN        → pstn()


def telephony() -> Compose:
    """
    Simulates a phone call: narrowband codec, packet loss, cheap mic, and noise.
    Targets the C2 (VoIP/a-law) and C3 (PSTN/u-law) conditions from ASVspoof 2021.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(15, 30)), PinkNoise(snr_db=(15, 25))], p=0.7),
            BandpassFilter(low_hz=300, high_hz=3400, p=1.0),
            MicColoration(freq_hz=1500, gain_db=3.0, p=0.4),
            CodecDistortion(threshold=0.85, p=0.5),
            PacketLoss(rate=0.05, fill="silence", p=0.3),
            RandomGain(db_range=(-6, 6), p=0.5),
        ]
    )


def full() -> Compose:
    """All augmentations enabled with moderate probability — general-purpose training."""
    return Compose(
        [
            OneOf(
                [
                    WhiteNoise(snr_db=(10, 30)),
                    PinkNoise(snr_db=(10, 25)),
                    BrownNoise(snr_db=(10, 20)),
                ],
                p=0.7,
            ),
            RoomImpulseResponse(room_dim_range=(3, 10), rt60_range=(0.2, 0.8), p=0.4),
            BandpassFilter(low_hz=300, high_hz=3400, p=0.5),
            MicColoration(freq_hz=1500, gain_db=3.0, p=0.3),
            CodecDistortion(threshold=0.85, p=0.4),
            PacketLoss(rate=0.05, fill="silence", p=0.3),
            TimeStretch(rate_range=(0.9, 1.1), p=0.2),
            RandomGain(db_range=(-6, 6), p=0.5),
        ]
    )


def light() -> Compose:
    """Minimal augmentation — noise and gain only."""
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(20, 35)), PinkNoise(snr_db=(20, 30))], p=0.5),
            RandomGain(db_range=(-3, 3), p=0.5),
        ]
    )


def voip() -> Compose:
    """
    VoIP / video call simulation (Zoom, Teams, Discord).
    Wideband codec (Opus-like), heavier packet loss with jitter-buffer repeat fill.
    Maps to ASVspoof 2021 C2 (a-law / VoIP) condition.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(20, 35)), PinkNoise(snr_db=(20, 30))], p=0.5),
            BandpassFilter(low_hz=50, high_hz=7000, p=1.0),
            CodecDistortion(threshold=0.75, p=0.6),
            PacketLoss(rate=0.08, fill="repeat", p=0.5),
            RandomGain(db_range=(-8, 4), p=0.5),
        ]
    )


def pstn() -> Compose:
    """
    Landline / PSTN narrowband simulation.
    Strict G.711 band, no packet loss (circuit-switched), cheap handset coloration.
    Maps to ASVspoof 2021 C3 (u-law / PSTN) condition.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(20, 35)), PinkNoise(snr_db=(18, 28))], p=0.6),
            BandpassFilter(low_hz=300, high_hz=3400, p=1.0),
            MicColoration(freq_hz=1200, gain_db=4.0, p=0.6),
            CodecDistortion(threshold=0.80, p=0.5),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def meeting_room() -> Compose:
    """
    In-person meeting / conference room.
    Reverb is the dominant degradation — large room RIR with background noise.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(25, 40)), PinkNoise(snr_db=(20, 35))], p=0.6),
            RoomImpulseResponse(room_dim_range=(5, 15), rt60_range=(0.4, 1.2), p=0.8),
            MicColoration(freq_hz=2000, gain_db=2.0, p=0.3),
            RandomGain(db_range=(-6, 6), p=0.5),
        ]
    )


def noisy_mobile() -> Compose:
    """
    Mobile phone in a loud environment (street, cafe).
    Heavy background noise, AMR-WB-ish bandwidth, light packet loss.
    """
    return Compose(
        [
            SomeOf(
                [
                    WhiteNoise(snr_db=(5, 15)),
                    PinkNoise(snr_db=(5, 15)),
                    BrownNoise(snr_db=(8, 18)),
                ],
                n=(1, 2),
                p=0.9,
            ),
            BandpassFilter(low_hz=200, high_hz=5000, p=0.8),
            CodecDistortion(threshold=0.80, p=0.4),
            PacketLoss(rate=0.03, fill="silence", p=0.3),
            RandomGain(db_range=(-8, 6), p=0.5),
        ]
    )


def adversarial() -> Compose:
    """
    Stress test — stacks multiple degradations aggressively.
    Useful for training robustness on worst-case inputs.
    """
    return Compose(
        [
            SomeOf(
                [
                    WhiteNoise(snr_db=(5, 15)),
                    PinkNoise(snr_db=(5, 12)),
                    BrownNoise(snr_db=(5, 15)),
                ],
                n=(1, 3),
                p=1.0,
            ),
            RoomImpulseResponse(room_dim_range=(3, 12), rt60_range=(0.3, 1.0), p=0.6),
            BandpassFilter(low_hz=300, high_hz=3400, p=0.7),
            CodecDistortion(threshold=0.70, p=0.7),
            PacketLoss(rate=0.12, fill="silence", p=0.5),
            TimeStretch(rate_range=(0.9, 1.1), p=0.3),
            RandomGain(db_range=(-10, 6), p=0.7),
        ]
    )
