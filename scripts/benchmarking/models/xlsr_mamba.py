"""
XLSR-Mamba adapter.

Weights are pulled automatically from HuggingFace (AustinXiao/XLSR-Mamba-LA).
Requires mamba-ssm, which only builds on Linux/WSL2 with CUDA.
The XLSR-Mamba repo must be on sys.path before this module is imported —
the __main__.py handles that via --repo_path.
"""

import argparse
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..base.model_base import BenchmarkModel


_HF_REPO = "AustinXiao/XLSR-Mamba-LA"
_HF_FILE = "model.safetensors"

_DEFAULT_EMB_SIZE = 144
_DEFAULT_NUM_ENCODERS = 12


class XLSRMamba(BenchmarkModel):
    def __init__(
        self,
        repo_path: str,
        emb_size: int = _DEFAULT_EMB_SIZE,
        num_encoders: int = _DEFAULT_NUM_ENCODERS,
    ):
        self._repo_path = repo_path
        self._emb_size = emb_size
        self._num_encoders = num_encoders
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return "XLSR-Mamba (AustinXiao/XLSR-Mamba-LA)"

    def load(self, device: torch.device) -> None:
        if self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        from model import Model  # noqa: PLC0415 — XLSR-Mamba repo import

        print("Locating weights from Hugging Face...")
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        print(f"Weights at: {weights_path}")

        model_args = argparse.Namespace(
            emb_size=self._emb_size, num_encoders=self._num_encoders
        )
        model = Model(model_args, device)
        model.load_state_dict(load_file(weights_path))
        model.to(device)
        model.eval()

        self._model = model
        self._device = device

    def predict(self, chunks: torch.Tensor) -> torch.Tensor:
        out = self._model(chunks)  # (N, 2)
        return out[:, 1]  # bonafide logit

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
