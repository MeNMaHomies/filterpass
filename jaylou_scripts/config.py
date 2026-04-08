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
    "batch_size": 32,  # increase if GPU OOMs, decrease to 16
    "max_epochs": 15,
    "patience": 16,
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
    # 14 physical / 20 logical cores. Workers are I/O-bound (.npy cache loads),
    # but Windows spawn overhead caps the benefit — 10 is the sweet spot.
    "num_workers": 10,
    "prefetch_factor": 4,  # batches queued per worker (10×4 = 40 batches ahead)
    # Encoder layer freezing — freeze the bottom N transformer blocks.
    # wav2vec2-base has 12 transformer layers.
    # RTX 4050 has 6 GB VRAM: freeze 8 to keep only the top 4 trainable.
    # This cuts ~65% of backward-pass cost and reduces activation VRAM substantially.
    # Set to 0 to train all layers (requires gradient_checkpointing=True to fit in 6 GB).
    "freeze_encoder_layers": 6,
    # Gradient checkpointing — recompute activations on backward instead of storing them.
    # Cuts activation VRAM ~40% at the cost of a ~15% slower backward pass.
    # Required to fit batch_size=32 through unfrozen Wav2Vec2 layers on 6 GB.
    "gradient_checkpointing": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
