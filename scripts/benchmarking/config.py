from dataclasses import dataclass


TARGET_SR = 16000
CHUNK_DURATION_S = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_S)  # 8000


@dataclass
class BenchmarkConfig:
    """Inference execution config — no dataset or model paths."""

    out_dir: str
    phase: str | None = None  # None = all phases
    batch_size: int = 32
    num_workers: int = 8

    # ── chunking ──────────────────────────────────────────────────────────────
    vad: bool = False  # filter silence via WebRTC VAD before chunking
    vad_mode: int = 2  # VAD aggressiveness 0-3 (3 = most aggressive)
    overlap_pct: int = 0  # sliding window overlap percentage (0 = no overlap, 80 = 80% overlap)
