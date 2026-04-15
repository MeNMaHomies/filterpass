"""
TCM adapter (Truong et al., INTERSPEECH 2024).

Architecture: XLS-R 300M (fairseq) frontend + Conformer encoder + CLS token head.
Paper: Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection
Repo:  https://github.com/ductuantruong/tcm_add
License: MIT

Same XLS-R frontend as wav2vec2-aasist — model.py hardcodes `cp_path = 'xlsr2_300m.pt'`
relative to CWD, so we temporarily chdir to `xlsr_dir` during construction.

Model returns a tuple (logits, attn_weights). Bonafide score = logits[:, 1].
Pooling via CLS token — fully adaptive, no fixed sequence-length dependency.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from ..base.model_base import BenchmarkModel


class TCM(BenchmarkModel):
    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        xlsr_dir: str | None = None,
        emb_size: int = 144,
        heads: int = 4,
        kernel_size: int = 31,
        num_encoders: int = 4,
    ):
        self._repo_path = repo_path
        self._weights_path = weights_path
        self._xlsr_dir = xlsr_dir or repo_path
        self._emb_size = emb_size
        self._heads = heads
        self._kernel_size = kernel_size
        self._num_encoders = num_encoders
        self._model = None

    @property
    def name(self) -> str:
        return "TCM (Truong et al. 2024)"

    def load(self, device: torch.device) -> None:
        if self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        from model import Model  # noqa: PLC0415 — tcm_add repo

        args = argparse.Namespace(
            emb_size=self._emb_size,
            heads=self._heads,
            kernel_size=self._kernel_size,
            num_encoders=self._num_encoders,
        )

        prev_dir = os.getcwd()
        os.chdir(self._xlsr_dir)
        try:
            model = Model(args, device)
        finally:
            os.chdir(prev_dir)

        state = torch.load(self._weights_path, map_location=device)
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        self._model = model

    def predict(self, chunks: torch.Tensor) -> torch.Tensor:
        out, _ = self._model(chunks)  # (N, 2), attn weights discarded
        return out[:, 1]              # bonafide score

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
