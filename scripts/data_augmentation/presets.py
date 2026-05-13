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
    CodecRoundTrip,
    HardClip,
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


def cellular_weak() -> Compose:
    """
    2G / 3G weak signal.
    Extreme narrowband (AMR-NB: 200–3400 Hz), heavy codec distortion from low
    bitrate quantization, high packet loss, and strong noise floor.
    """
    return Compose(
        [
            OneOf(
                [
                    WhiteNoise(snr_db=(5, 15)),
                    PinkNoise(snr_db=(5, 12)),
                    BrownNoise(snr_db=(8, 15)),
                ],
                p=0.9,
            ),
            BandpassFilter(low_hz=200, high_hz=3400, p=1.0),
            CodecDistortion(threshold=0.60, p=0.9),
            PacketLoss(rate=0.15, fill="silence", p=0.7),
            RandomGain(db_range=(-6, 3), p=0.5),
        ]
    )


def hd_voice() -> Compose:
    """
    Modern HD Voice / EVS (4G / 5G calls).
    Wide bandwidth (50–14000 Hz), minimal codec distortion, no packet loss.
    The near-clean end of the phone-call spectrum.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(25, 40)), PinkNoise(snr_db=(25, 38))], p=0.4),
            BandpassFilter(low_hz=50, high_hz=14000, p=1.0),
            CodecDistortion(threshold=0.92, p=0.3),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def speakerphone() -> Compose:
    """
    Phone on loudspeaker (speaker → room → mic path).
    Small-to-medium room reverb, distant mic coloration, lower SNR.
    No packet loss — degradation comes from the acoustic path.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(15, 25)), PinkNoise(snr_db=(15, 22))], p=0.6),
            RoomImpulseResponse(room_dim_range=(3, 8), rt60_range=(0.2, 0.5), p=0.9),
            MicColoration(freq_hz=2500, gain_db=3.0, p=0.6),
            BandpassFilter(low_hz=150, high_hz=6000, p=0.7),
            RandomGain(db_range=(-8, 2), p=0.5),
        ]
    )


def codec_chain() -> Compose:
    """
    Multiple codec hops (PSTN ↔ VoIP transcoding).
    Applies two passes of BandpassFilter + CodecDistortion to simulate
    the artefact accumulation of international calls and conference bridges.
    """
    return Compose(
        [
            BandpassFilter(low_hz=300, high_hz=3400, p=1.0),
            CodecDistortion(threshold=0.80, p=1.0),
            BandpassFilter(low_hz=300, high_hz=3400, p=1.0),
            CodecDistortion(threshold=0.75, p=1.0),
            OneOf([WhiteNoise(snr_db=(20, 30)), PinkNoise(snr_db=(20, 28))], p=0.5),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def laptop_mic() -> Compose:
    """
    Built-in laptop microphone.
    Mid-range coloration from laptop body resonance (2–4 kHz boost),
    light HVAC background noise (brown noise, high SNR). Very common in
    remote work and online meetings.
    """
    return Compose(
        [
            BrownNoise(snr_db=(25, 40), p=0.7),
            MicColoration(freq_hz=3000, gain_db=4.0, p=0.8),
            BandpassFilter(low_hz=100, high_hz=8000, p=0.6),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def headset() -> Compose:
    """
    USB / Bluetooth headset.
    Slight high-frequency rolloff (SBC codec bandwidth), mild compression,
    low noise. The 'clean but slightly processed' meeting scenario.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(30, 45)), PinkNoise(snr_db=(28, 42))], p=0.3),
            BandpassFilter(low_hz=100, high_hz=8000, p=0.8),
            CodecDistortion(threshold=0.90, p=0.4),
            RandomGain(db_range=(-3, 3), p=0.4),
        ]
    )


def hybrid_meeting() -> Compose:
    """
    Hybrid meeting — remote participant piped through VoIP into a conference room.
    Chains room reverb (in-room acoustics) with VoIP-style codec degradation
    and packet loss on the remote leg.
    """
    return Compose(
        [
            OneOf([WhiteNoise(snr_db=(20, 35)), PinkNoise(snr_db=(18, 30))], p=0.5),
            RoomImpulseResponse(room_dim_range=(5, 12), rt60_range=(0.3, 0.8), p=0.7),
            BandpassFilter(low_hz=50, high_hz=7000, p=0.8),
            CodecDistortion(threshold=0.78, p=0.6),
            PacketLoss(rate=0.06, fill="repeat", p=0.4),
            RandomGain(db_range=(-6, 4), p=0.5),
        ]
    )


def outdoor_windy() -> Compose:
    """
    Outdoor call or interview in wind.
    Wind noise is spectrally brown at low frequencies and low SNR.
    Bandwidth is intact — degradation is purely environmental noise.
    """
    return Compose(
        [
            BrownNoise(snr_db=(3, 12), p=1.0),
            PinkNoise(snr_db=(15, 25), p=0.4),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def deepfake_targeted() -> Compose:
    return OneOf(
        [
            pstn(),
            voip(),
        ],
        p=0.5,
    )


def near_silence() -> Compose:
    """
    Edge case: very quiet / distant recording.
    Simulates a muted or far-field mic — barely audible signal.
    Tests model robustness at very low input levels.
    """
    return Compose(
        [
            RandomGain(db_range=(-20, -12), p=1.0),
            OneOf([WhiteNoise(snr_db=(10, 20)), PinkNoise(snr_db=(10, 18))], p=0.5),
        ]
    )


def dropout() -> Compose:
    """
    Edge case: severe / extended call dropouts.
    High packet loss rate (20–40 %) with long silence bursts — the
    'call keeps cutting out' experience rather than occasional glitches.
    """
    return Compose(
        [
            PacketLoss(rate=0.30, fill="silence", p=1.0),
            OneOf([WhiteNoise(snr_db=(15, 25)), PinkNoise(snr_db=(15, 22))], p=0.4),
            RandomGain(db_range=(-4, 4), p=0.4),
        ]
    )


def clipping() -> Compose:
    """
    Edge case: microphone overload / saturation.
    Hard-clips at a random low threshold (0.3–0.6) — happens when a
    speaker gets too close to a mic or shouts into a cheap headset.
    """
    return Compose(
        [
            HardClip(threshold_range=(0.3, 0.6), p=1.0),
            OneOf([WhiteNoise(snr_db=(20, 30)), PinkNoise(snr_db=(20, 28))], p=0.4),
            RandomGain(db_range=(-3, 3), p=0.4),
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
