from __future__ import annotations

from dataclasses import dataclass, field

TARGET_SR = 16000

# Default chunk duration — kept as a module-level constant for import compatibility.
# Use BenchmarkConfig.chunk_samples at runtime so the CLI can override it.
CHUNK_DURATION_MS = 500
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_MS / 1000)  # 8000


@dataclass
class BenchmarkConfig:
    """Inference execution config — no dataset or model paths."""

    out_dir: str
    phase: str | None = None  # None = all phases
    batch_size: int = 32
    num_workers: int = 8

    # ── chunking ──────────────────────────────────────────────────────────────
    static: bool = False     # full-utterance mode — no chunking, one forward pass per file
    chunk_ms: int = CHUNK_DURATION_MS  # chunk duration in milliseconds (ignored when static=True)
    vad: bool = False        # filter silence via WebRTC VAD before chunking
    vad_mode: int = 2        # VAD aggressiveness 0-3 (3 = most aggressive)
    overlap_pct: int = 0     # sliding window overlap percentage (0 = no overlap, 80 = 80%)

    @property
    def chunk_samples(self) -> int:
        return int(TARGET_SR * self.chunk_ms / 1000)
