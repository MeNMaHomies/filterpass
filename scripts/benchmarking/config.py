from dataclasses import dataclass


TARGET_SR = 16000
CHUNK_DURATION_S = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_S)  # 8000


@dataclass
class BenchmarkConfig:
    """Inference execution config — no dataset or model paths."""
    out_dir: str
    phase: str = "eval"
    batch_size: int = 32
    num_workers: int = 8
