from .augmentations import (
    BackgroundNoise,
    BandpassFilter,
    BrownNoise,
    CodecDistortion,
    HardClip,
    MicColoration,
    PacketLoss,
    PinkNoise,
    RandomGain,
    RoomImpulseResponse,
    TimeStretch,
    WhiteNoise,
)
from .base import Augmentation
from .compose import Compose, OneOf, SomeOf
from .dataset import AudioDataset
from .pipeline import augment_directory, augment_file
from .presets import (
    adversarial,
    cellular_weak,
    clipping,
    codec_chain,
    dropout,
    full,
    hd_voice,
    headset,
    hybrid_meeting,
    laptop_mic,
    light,
    meeting_room,
    near_silence,
    noisy_mobile,
    outdoor_windy,
    pstn,
    speakerphone,
    telephony,
    voip,
)

__all__ = [
    # Base + combinators
    "Augmentation",
    "Compose",
    "OneOf",
    "SomeOf",
    # Augmentations
    "WhiteNoise",
    "PinkNoise",
    "BrownNoise",
    "BackgroundNoise",
    "RoomImpulseResponse",
    "BandpassFilter",
    "MicColoration",
    "CodecDistortion",
    "HardClip",
    "PacketLoss",
    "TimeStretch",
    "RandomGain",
    # Presets — phone calls
    "telephony",
    "voip",
    "pstn",
    "cellular_weak",
    "hd_voice",
    "speakerphone",
    "codec_chain",
    # Presets — live meetings
    "meeting_room",
    "laptop_mic",
    "headset",
    "hybrid_meeting",
    "outdoor_windy",
    "noisy_mobile",
    # Presets — edge cases
    "near_silence",
    "dropout",
    "clipping",
    "adversarial",
    # Presets — general
    "full",
    "light",
    # File I/O
    "augment_file",
    "augment_directory",
    # Dataset
    "AudioDataset",
]
