import numpy as np

TARGET_SR = 16000
CHUNK_DURATION_S = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_S)  # 8000 — default only
_VAD_FRAME_MS = 30  # WebRTC VAD requires 10, 20, or 30 ms frames
_BYTES_PER_SAMPLE = 2  # 16-bit PCM

# ── VAD ───────────────────────────────────────────────────────────────────────


def _vad_filter(audio: np.ndarray, sr: int, vad_mode: int) -> np.ndarray:
    """
    Run WebRTC VAD on 30ms frames and return only the voiced samples
    as a contiguous float32 array. Silence frames are discarded entirely.
    """
    import webrtcvad

    vad = webrtcvad.Vad(vad_mode)

    # WebRTC VAD needs 16-bit PCM bytes
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_bytes = int(sr * (_VAD_FRAME_MS / 1000.0)) * _BYTES_PER_SAMPLE
    frame_samples = int(sr * (_VAD_FRAME_MS / 1000.0))

    voiced_samples = []
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame_pcm = pcm[offset : offset + frame_bytes]

        # If background noise, discard the entire frame; if speech, keep it.
        if vad.is_speech(frame_pcm, sr):
            sample_start = offset // _BYTES_PER_SAMPLE
            voiced_samples.append(audio[sample_start : sample_start + frame_samples])

    if not voiced_samples:
        return np.array([], dtype=np.float32)

    return np.concatenate(voiced_samples)


# ── Chunking ──────────────────────────────────────────────────────────────────


def _chunk_audio(
    audio: np.ndarray,
    hop_samples: int,
    chunk_samples: int = CHUNK_SAMPLES,
) -> list[np.ndarray]:
    """
    Split audio into chunk_samples-sized windows advancing by hop_samples.
    The final chunk is zero-padded if shorter than chunk_samples.

    Args:
        audio:         1-D float32 waveform.
        hop_samples:   Step size between windows.
        chunk_samples: Window size in samples (default: module-level CHUNK_SAMPLES).
    """
    if len(audio) == 0:
        return []

    chunks = []

    for start in range(0, max(len(audio) - chunk_samples + 1, 1), hop_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)

    return chunks