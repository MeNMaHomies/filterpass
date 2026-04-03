from .augmentations import (
    BackgroundNoise,
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
from .base import Augmentation
from .compose import Compose, OneOf, SomeOf
from .pipeline import augment_directory, augment_file
from .presets import (
    adversarial,
    full,
    light,
    meeting_room,
    noisy_mobile,
    pstn,
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
    "PacketLoss",
    "TimeStretch",
    "RandomGain",
    # Presets
    "telephony",
    "full",
    "light",
    "voip",
    "pstn",
    "meeting_room",
    "noisy_mobile",
    "adversarial",
    # File I/O
    "augment_file",
    "augment_directory",
]
