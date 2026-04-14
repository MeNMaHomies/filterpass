"""
FatigueSense SAP adapter.

Wav2Vec2-base fine-tuned with a Self-Attention Pooling classification head.
Weights are loaded from a local .pt state_dict file (state_dict only, not full model).

Fill in _HF_REPO and _HF_FILE before running.
"""

from __future__ import annotations

import torch

from ..base.model_base import BenchmarkModel

_HF_REPO = "Menmahomies/SAP_Classifier"
_HF_FILE = "best_model_SAP_v4.pt"

# Index 0 = bonafide, index 1 = spoof — verify against training label convention
_BONAFIDE_IDX = 0


# ── Adapter ───────────────────────────────────────────────────────────────────
class FilterpassSAP(BenchmarkModel):
    def __init__(
        self, model_name: str = "facebook/wav2vec2-base", freeze_extractor: bool = True
    ):
        self._model_name = model_name
        self._freeze_extractor = freeze_extractor
        self._model: SAPClassifier | None = None

    @property
    def name(self) -> str:
        return "Filterpass-SAP (Wav2Vec2-base + Self-Attention Pooling)"

    def load(self, device: torch.device) -> None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from duy_scripts.classifiers.model_SAP import SAPClassifier  # noqa: PLC0415

        print("Locating weights from Hugging Face...")
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        print(f"Weights at: {weights_path}")

        model = SAPClassifier(self._model_name, self._freeze_extractor)
        model.load_state_dict(torch.load(weights_path, map_location=device))
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
