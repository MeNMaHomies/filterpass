"""
Nes2Net adapter (Liu et al., IEEE TIFS 2025).

Architecture: XLS-R 300M (fairseq) frontend + Nested Res2Net TDNN backend.
Repo:  https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW
Paper: https://arxiv.org/abs/2504.05657

Same XLS-R frontend as wav2vec2-aasist — model_scripts/wav2vec2_Nes2Net_X.py
hardcodes `cp_path = 'xlsr2_300m.pt'` relative to CWD, so we temporarily
chdir to `xlsr_dir` during construction.

Pooling is mean or ASTP over the time dimension — fully adaptive, no fixed
sequence-length dependency.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from ..base.model_base import BenchmarkModel


class Nes2Net(BenchmarkModel):
    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        xlsr_dir: str | None = None,
        pool_func: str = "mean",
        nes_ratio: list[int] | None = None,
        se_ratio: list[int] | None = None,
        dilation: int = 2,
    ):
        self._repo_path = repo_path
        self._weights_path = weights_path
        self._xlsr_dir = xlsr_dir or repo_path
        self._pool_func = pool_func
        self._nes_ratio = nes_ratio or [8, 8]
        self._se_ratio = se_ratio or [1]
        self._dilation = dilation
        self._model = None

    @property
    def name(self) -> str:
        return "Nes2Net (Liu et al. 2025)"

    def load(self, device: torch.device) -> None:
        if self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        from model_scripts.wav2vec2_Nes2Net_X import (  # noqa: PLC0415
            wav2vec2_Nes2Net_no_Res_w_allT as Model,
        )

        args = argparse.Namespace(
            n_output_logits=2,
            dilation=self._dilation,
            pool_func=self._pool_func,
            Nes_ratio=self._nes_ratio,
            SE_ratio=self._se_ratio,
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
        out = self._model(chunks)  # (N, 2) logits
        return out[:, 1]           # bonafide score

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
