"""
Model-agnostic batched inference loop.

Accepts any BenchmarkModel adapter and a DataLoader built by audio_loader,
returns per-utterance scores and per-batch RTF measurements.

Batching strategy: chunks from multiple utterances are buffered into a single
GPU batch of exactly `batch_size` chunks. This keeps the GPU at full
utilization even when individual utterances are short (e.g. 3-4 seconds →
6-8 chunks, well below a typical batch_size of 32).
"""

import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .base.model_base import BenchmarkModel
from .config import CHUNK_DURATION_S


def _flush(
    model: BenchmarkModel,
    device: torch.device,
    chunk_buffer: list[torch.Tensor],
    owner_buffer: list[str],
    chunk_scores: dict[str, list[float]],
    all_rtf: list[float],
) -> None:
    """Run one GPU forward pass on whatever is currently in the buffer."""
    if not chunk_buffer:
        return

    batch = torch.stack(chunk_buffer).to(device)
    n = len(batch)
    t0 = time.perf_counter()
    scores = model.predict(batch)
    elapsed = time.perf_counter() - t0
    all_rtf.append(elapsed / (n * CHUNK_DURATION_S))

    for utt_id, score in zip(owner_buffer, scores.cpu().numpy().tolist()):
        chunk_scores[utt_id].append(score)

    chunk_buffer.clear()
    owner_buffer.clear()


def run_inference(
    model: BenchmarkModel,
    loader,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], list[float], list[str]]:
    """
    Iterate over `loader`, buffer chunks across utterances into full
    `batch_size` GPU batches, then aggregate per-utterance scores as the
    mean bonafide score across all chunks.

    Returns:
        all_scores  — {utt_id: mean_score}
        all_rtf     — list of per-batch RTF values
        skipped_ids — list of utt_ids that could not be loaded
    """
    all_rtf: list[float] = []
    skipped_ids: list[str] = []
    chunk_scores: dict[str, list[float]] = defaultdict(list)

    # GPU warmup: trigger CUDA kernel compilation before timed inference begins
    # so the first real batch doesn't inflate rtf_mean/rtf_max.
    if device.type == "cuda":
        print("Warming up GPU...", flush=True)
        dummy = torch.zeros(batch_size, 8000, device=device)
        with torch.no_grad():
            for _ in range(2):
                model.predict(dummy)
        torch.cuda.synchronize(device)
        print("Warmup done.", flush=True)

    # Cross-utterance chunk buffer: accumulate until we have a full batch
    chunk_buffer: list[torch.Tensor] = []
    owner_buffer: list[str] = []  # which utt_id each buffered chunk belongs to

    total = len(loader.dataset)
    with torch.no_grad():
        with tqdm(total=total, unit="utt") as pbar:
            for batch in loader:
                utt_id, chunks_tensor = batch[0]

                if chunks_tensor is None:
                    skipped_ids.append(utt_id)
                    tqdm.write(f"[SKIPPED] {utt_id}")
                    pbar.update(1)
                    continue

                for chunk in chunks_tensor:
                    chunk_buffer.append(chunk)
                    owner_buffer.append(utt_id)

                    if len(chunk_buffer) == batch_size:
                        _flush(
                            model,
                            device,
                            chunk_buffer,
                            owner_buffer,
                            chunk_scores,
                            all_rtf,
                        )

                pbar.update(1)

            # Flush any remaining chunks that didn't fill a complete batch
            _flush(model, device, chunk_buffer, owner_buffer, chunk_scores, all_rtf)

    all_scores = {
        utt_id: float(np.mean(scores)) for utt_id, scores in chunk_scores.items()
    }
    return all_scores, all_rtf, skipped_ids
