"""
Filterpass SAP v5 adapter.

Wav2Vec2-base fine-tuned with a Self-Attention Pooling classification head.
Weights loaded from HuggingFace Hub (state_dict only).
"""

from __future__ import annotations

import torch

from ..base.model_base import BenchmarkModel

_HF_REPO = "Menmahomies/SAP_Classifierv5"
_HF_FILE = "best_model_SAP_v5.pt"

# Index 0 = bonafide, index 1 = spoof
_BONAFIDE_IDX = 0


class FilterpassSAPv5(BenchmarkModel):
    def __init__(
        self, model_name: str = "facebook/wav2vec2-base", freeze_extractor: bool = True
    ):
        self._model_name = model_name
        self._freeze_extractor = freeze_extractor
        self._model = None

    @property
    def name(self) -> str:
        return "Filterpass-SAP v5 (Wav2Vec2-base + Self-Attention Pooling)"

    def load(self, device: torch.device) -> None:
        from huggingface_hub import hf_hub_download

        from duy_scripts.classifiers.model_SAP import SAPClassifier

        print("Locating weights from Hugging Face...")
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        print(f"Weights at: {weights_path}")

        model = SAPClassifier(self._model_name, self._freeze_extractor)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        self._model = model

    def predict(self, chunks: torch.Tensor) -> torch.Tensor:
        logits = self._model(chunks)  # (N, 2)
        return logits[:, _BONAFIDE_IDX]

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
