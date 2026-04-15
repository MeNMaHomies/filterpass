"""
SSL-Anti-Spoofing adapter (Tak et al., Speaker Odyssey 2022).

Architecture: wav2vec 2.0 XLSR (300M) frontend + AASIST backend.
Paper: https://arxiv.org/abs/2202.12233
Repo:  https://github.com/TakHemlata/SSL_Anti-spoofing

Requires two external files not bundled in the repo:
  - xlsr2_300m.pt     : pre-trained XLSR feature extractor (fairseq format)
  - best_SSL_model_LA.pth : fine-tuned LA checkpoint from Google Drive

The upstream model.py hardcodes `cp_path = 'xlsr2_300m.pt'` (relative CWD lookup),
so we temporarily chdir to `xlsr_dir` during construction.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from ..base.model_base import BenchmarkModel


class Wav2Vec2Aasist(BenchmarkModel):
    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        xlsr_dir: str | None = None,
    ):
        self._repo_path = repo_path
        self._weights_path = weights_path
        self._xlsr_dir = xlsr_dir or repo_path
        self._model = None

    @property
    def name(self) -> str:
        return "Wav2Vec2-AASIST (Tak et al. 2022)"

    def load(self, device: torch.device) -> None:
        if self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        from model import Model  # type: ignore # noqa: PLC0415 — SSL_Anti-spoofing repo

        prev_dir = os.getcwd()
        os.chdir(self._xlsr_dir)
        try:
            model = Model(argparse.Namespace(), device)
        finally:
            os.chdir(prev_dir)

        state = torch.load(self._weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        self._model = model

    def predict(self, chunks: torch.Tensor) -> torch.Tensor:
        out = self._model(chunks)  # (N, 2)
        return out[:, 1]  # bonafide logit

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
