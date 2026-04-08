"""
Legacy config — chunked inference approach (0.5s chunks during training).
Superseded by jaylou_scripts/config.py which uses 4s fixed-length utterances.
"""
import torch

from duy_scripts.classifiers import SAPClassifier

SEED = 42

CONFIG = {
    # Paths
    "base_dir":        "./data_training",
    "checkpoint_dir":  "./checkpoints",
    "checkpoint_name": "best_model_SAP.pt",
    "model":           SAPClassifier,

    # Training
    "batch_size":  16,
    "max_epochs":  15,
    "patience":    16,

    # Optimiser
    "lr_encoder":    1e-6,
    "lr_classifier": 1e-4,
    "weight_decay":  0.01,
    "warmup_ratio":  0.1,
    "max_grad_norm": 0.5,

    # Loss
    "class_weights": [9.0, 1.0],

    # DataLoader
    "num_workers": 0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
