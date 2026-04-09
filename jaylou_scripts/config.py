import torch

from duy_scripts.classifiers import SAPClassifier

SEED = 42

CONFIG = {
    # Paths
    "base_dir": "./data_training",
    "checkpoint_dir": "./checkpoints",
    "checkpoint_name": "best_model_SAP.pt",
    "model": SAPClassifier,
    # Training
    "batch_size": 16,
    "max_epochs": 25,
    "patience": 5,
    # Minimum EER drop (absolute) that counts as a meaningful improvement.
    # Gains smaller than this increment the patience counter even if EER technically improved.
    "min_delta": 0.001,  # 0.1 percentage points
    # Gradient accumulation — effective batch = batch_size * grad_accum_steps.
    # Set to 1 to disable. Useful when batch_size is limited by GPU memory.
    "grad_accum_steps": 1,
    # Optimiser
    "lr_encoder": 1e-6,
    "lr_classifier": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 0.5,
    # Loss — [bonafide_weight, spoof_weight]
    # Dataset is ~balanced after 8x bonafide oversampling, so weights are
    # symmetric. Revert to [9.0, 1.0] if training without the augmented set.
    "class_weights": [1.0, 1.0],
    # DataLoader
    # num_workers=0 outperforms multi-worker loading on Windows because .npy cache
    # loads are fast enough for the main process to keep the GPU fed, and Windows
    # spawn overhead (full interpreter per worker) costs more than the parallelism saves.
    "num_workers": 0,
    "prefetch_factor": 4,  # only active when num_workers > 0
    # Encoder layer freezing — freeze the bottom N transformer blocks.
    # wav2vec2-base has 12 transformer layers.
    # 8 = freeze layers 0–7, train only top 4 — faster backward, less adaptation.
    # 6 = freeze layers 0–5, train layers 6–11 — more adaptation, slower.
    # Set to 0 to train all layers (requires gradient_checkpointing=True to fit in 6 GB).
    "freeze_encoder_layers": 8,
    # Gradient checkpointing — recompute activations on backward instead of storing them.
    # Cuts activation VRAM ~40% at the cost of a ~15% slower backward pass.
    # Required to fit batch_size=32 through unfrozen Wav2Vec2 layers on 6 GB.
    "gradient_checkpointing": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
