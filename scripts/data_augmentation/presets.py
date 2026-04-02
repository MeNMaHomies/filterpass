"""
Factory functions that return ready-to-use augmentation pipelines.

Each preset returns a Compose that can be called directly:
    transform = telephony()
    augmented = transform(audio, sr)
"""

from .compose import Compose, OneOf, SomeOf
from .augmentations import (
    WhiteNoise, PinkNoise, BrownNoise,
    RoomImpulseResponse, BandpassFilter, MicColoration, CodecDistortion,
    PacketLoss, TimeStretch, RandomGain,
)


def telephony() -> Compose:
    """
    Simulates a phone call: narrowband codec, packet loss, cheap mic, and noise.
    Targets the C2 (VoIP/a-law) and C3 (PSTN/u-law) conditions from ASVspoof 2021.
    """
    return Compose([
        OneOf([WhiteNoise(snr_db=(15, 30)), PinkNoise(snr_db=(15, 25))], p=0.7),
        BandpassFilter(low_hz=300, high_hz=3400, p=1.0),
        MicColoration(freq_hz=1500, gain_db=3.0, p=0.4),
        CodecDistortion(threshold=0.85, p=0.5),
        PacketLoss(rate=0.05, fill="silence", p=0.3),
        RandomGain(db_range=(-6, 6), p=0.5),
    ])


def full() -> Compose:
    """All augmentations enabled with moderate probability — general-purpose training."""
    return Compose([
        OneOf([
            WhiteNoise(snr_db=(10, 30)),
            PinkNoise(snr_db=(10, 25)),
            BrownNoise(snr_db=(10, 20)),
        ], p=0.7),
        RoomImpulseResponse(room_dim_range=(3, 10), rt60_range=(0.2, 0.8), p=0.4),
        BandpassFilter(low_hz=300, high_hz=3400, p=0.5),
        MicColoration(freq_hz=1500, gain_db=3.0, p=0.3),
        CodecDistortion(threshold=0.85, p=0.4),
        PacketLoss(rate=0.05, fill="silence", p=0.3),
        TimeStretch(rate_range=(0.9, 1.1), p=0.2),
        RandomGain(db_range=(-6, 6), p=0.5),
    ])


def light() -> Compose:
    """Minimal augmentation — noise and gain only."""
    return Compose([
        OneOf([WhiteNoise(snr_db=(20, 35)), PinkNoise(snr_db=(20, 30))], p=0.5),
        RandomGain(db_range=(-3, 3), p=0.5),
    ])
