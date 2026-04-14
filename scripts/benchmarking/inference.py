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
from .config import TARGET_SR


def _flush(
    model: BenchmarkModel,
    device: torch.device,
    chunk_buffer: list[torch.Tensor],
    owner_buffer: list[str],
    chunk_scores: dict[str, list[float]],
    all_rtf: list[float],
    chunk_duration_s: float,
) -> None:
    """Run one GPU forward pass on whatever is currently in the buffer."""
    if not chunk_buffer:
        return

    batch = torch.stack(chunk_buffer).to(device)
    n = len(batch)
    t0 = time.perf_counter()
    scores = model.predict(batch)
    elapsed = time.perf_counter() - t0
    all_rtf.append(elapsed / (n * chunk_duration_s))

    for utt_id, score in zip(owner_buffer, scores.cpu().numpy().tolist()):
        chunk_scores[utt_id].append(score)

    chunk_buffer.clear()
    owner_buffer.clear()


def run_inference(
    model: BenchmarkModel,
    loader,
    device: torch.device,
    batch_size: int,
    chunk_ms: int = 500,
    static: bool = False,
) -> tuple[dict[str, float], list[float], list[str]]:
    """
    Chunked mode: buffers chunks across utterances into batch_size GPU batches,
    aggregates per-utterance scores via mean pooling across chunks.

    Static mode: one forward pass per file on the full audio (variable length,
    no batching across utterances). RTF is measured per utterance duration.

    Returns:
        all_scores  — {utt_id: mean_score}
        all_rtf     — list of RTF values (per batch in chunked, per utterance in static)
        skipped_ids — list of utt_ids that could not be loaded
    """
    all_rtf: list[float] = []
    skipped_ids: list[str] = []
    all_scores: dict[str, float] = {}

    if device.type == "cuda":
        print("Warming up GPU...", flush=True)
        warmup_samples = 64_000 if static else int(TARGET_SR * chunk_ms / 1000)
        dummy = torch.zeros(1 if static else batch_size, warmup_samples, device=device)
        with torch.no_grad():
            for _ in range(2):
                model.predict(dummy)
        torch.cuda.synchronize(device)
        print("Warmup done.", flush=True)

    total = len(loader.dataset)

    if static:
        # ── Static path: one utterance at a time, full audio, variable length ──
        with torch.no_grad():
            with tqdm(total=total, unit="utt") as pbar:
                for batch in loader:
                    utt_id, audio = batch[0]

                    if audio is None:
                        skipped_ids.append(utt_id)
                        tqdm.write(f"[SKIPPED] {utt_id}")
                        pbar.update(1)
                        continue

                    audio_duration_s = audio.shape[0] / TARGET_SR
                    inp = audio.unsqueeze(0).to(device)  # (1, full_samples)

                    t0 = time.perf_counter()
                    scores = model.predict(inp)
                    elapsed = time.perf_counter() - t0

                    all_rtf.append(elapsed / audio_duration_s)
                    all_scores[utt_id] = float(scores[0].cpu())
                    pbar.update(1)

    else:
        # ── Chunked path: cross-utterance batching, mean pooling ──────────────
        chunk_duration_s = chunk_ms / 1000.0
        chunk_scores: dict[str, list[float]] = defaultdict(list)
        chunk_buffer: list[torch.Tensor] = []
        owner_buffer: list[str] = []

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
                            _flush(model, device, chunk_buffer, owner_buffer,
                                   chunk_scores, all_rtf, chunk_duration_s)

                    pbar.update(1)

                _flush(model, device, chunk_buffer, owner_buffer,
                       chunk_scores, all_rtf, chunk_duration_s)

        all_scores = {
            utt_id: float(np.mean(scores)) for utt_id, scores in chunk_scores.items()
        }

    return all_scores, all_rtf, skipped_ids
