"""
Model-agnostic batched inference loop.

Accepts any BenchmarkModel adapter and a DataLoader built by audio_loader,
returns per-utterance scores and per-batch RTF measurements.
"""

import time

import numpy as np
import torch
from tqdm import tqdm

from .config import CHUNK_DURATION_S
from .model_base import BenchmarkModel


def run_inference(
    model: BenchmarkModel,
    loader,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], list[float], int]:
    """
    Iterate over `loader`, run `model.predict` in mini-batches of `batch_size` chunks,
    and aggregate per-utterance scores as the mean bonafide score across all chunks.

    Returns:
        all_scores  — {utt_id: mean_score}
        all_rtf     — list of per-mini-batch RTF values
        skipped     — number of utterances that could not be loaded
    """
    all_scores: dict[str, float] = {}
    all_rtf: list[float] = []
    skipped = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader.dataset)):
            utt_id, chunks_tensor = batch[0]

            if chunks_tensor is None:
                skipped += 1
                continue

            chunks_tensor = chunks_tensor.to(device)
            chunk_scores: list[float] = []

            for i in range(0, len(chunks_tensor), batch_size):
                mini = chunks_tensor[i : i + batch_size]
                n = len(mini)
                t0 = time.perf_counter()
                scores = model.predict(mini)
                elapsed = time.perf_counter() - t0
                chunk_scores.extend(scores.cpu().numpy().tolist())
                all_rtf.append(elapsed / (n * CHUNK_DURATION_S))

            all_scores[utt_id] = float(np.mean(chunk_scores))

    return all_scores, all_rtf, skipped
