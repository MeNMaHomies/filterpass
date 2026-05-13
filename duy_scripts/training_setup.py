"""
Training setup utilities — dataset construction, weighted sampler, DataLoader.

Usage in train.ipynb:
    from training_setup import build_loaders, build_eval_loaders
    train_loader, dev_loader = build_loaders(CONFIG, transform=transform)
    eval_loaders = build_eval_loaders(CONFIG)
"""

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from dataset import ASVspoof2019LADataset
from datasets import (
    ASVspoof2021DFDataset,
    CommonVoiceDataset,
    InTheWildDataset,
    MUSANSpeechDataset,
    VoxCelebDataset,
    WaveFakeDataset,
)


def _get_labels(dataset) -> list[int]:
    """Extract flat label list from ASVspoof2019LADataset or WaveFakeDataset."""
    if isinstance(dataset, ASVspoof2019LADataset):
        return [item["label"] for item in dataset.data]
    if isinstance(dataset, WaveFakeDataset):
        return [1] * len(dataset)           # WaveFake is spoof-only
    if isinstance(dataset, MUSANSpeechDataset):
        return [0] * len(dataset)           # MUSAN speech is bonafide-only
    if isinstance(dataset, VoxCelebDataset):
        return [0] * len(dataset)           # VoxCeleb1 is bonafide-only
    if isinstance(dataset, CommonVoiceDataset):
        return [0] * len(dataset)           # CommonVoice is bonafide-only
    if isinstance(dataset, ConcatDataset):
        labels = []
        for ds in dataset.datasets:
            labels.extend(_get_labels(ds))
        return labels
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def build_weighted_sampler(
    dataset, max_samples: Optional[int] = None
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that yields ~50/50 bonafide/spoof batches.

    Each sample is assigned a weight = 1 / count_of_its_class.
    num_samples defaults to 2 * min(n_bonafide, n_spoof) so the minority class
    is seen fully each epoch. If `max_samples` is set, the epoch is capped at
    that count instead — useful when one class explodes (e.g. VoxCeleb pushes
    bonafide to 150k+ and the natural epoch becomes ~280k).
    """
    labels = np.array(_get_labels(dataset))

    n_bonafide = int((labels == 0).sum())
    n_spoof    = int((labels == 1).sum())

    if n_bonafide == 0 or n_spoof == 0:
        raise ValueError(
            f"Both classes must be present. Got {n_bonafide} bonafide, {n_spoof} spoof."
        )

    ratio = max(n_bonafide, n_spoof) // max(min(n_bonafide, n_spoof), 1)
    print(f"Sampler — bonafide: {n_bonafide}, spoof: {n_spoof}, "
          f"ratio: 1:{ratio}")

    weight_per_class = {0: 1.0 / n_bonafide, 1: 1.0 / n_spoof}
    sample_weights = torch.tensor(
        [weight_per_class[int(l)] for l in labels], dtype=torch.float32
    )

    natural = 2 * min(n_bonafide, n_spoof)
    num_samples = min(natural, max_samples) if max_samples else natural
    if max_samples and max_samples < natural:
        print(f"Sampler — capping epoch at {num_samples} "
              f"(natural would be {natural})")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def build_loaders(
    config: dict,
    transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and dev DataLoaders.

    config keys used:
        base_dir        : root of asvspoof2019_LA
        wavefake_dir    : root of WaveFake (set None to skip WaveFake)
        batch_size      : int
        num_workers     : int

    Returns:
        train_loader, dev_loader
    """
    base_dir        = config["base_dir"]
    wavefake_dir    = config.get("wavefake_dir")
    musan_dir       = config.get("musan_dir")
    voxceleb_dir    = config.get("voxceleb_dir")
    commonvoice_dir = config.get("commonvoice_dir")
    batch_size      = config["batch_size"]
    num_workers  = config["num_workers"]
    max_seconds  = config.get("max_seconds", 4.0)

    # ── training set ─────────────────────────────────────────────────────────
    asvspoof_train = ASVspoof2019LADataset(
        base_dir, split="train", max_seconds=max_seconds, transform=transform
    )

    parts = [asvspoof_train]
    if wavefake_dir is not None:
        parts.append(WaveFakeDataset(
            wavefake_dir, max_seconds=max_seconds, transform=transform
        ))
    if musan_dir is not None:
        parts.append(MUSANSpeechDataset(
            musan_dir, max_seconds=max_seconds, transform=transform
        ))
    if voxceleb_dir is not None:
        parts.append(VoxCelebDataset(
            voxceleb_dir,
            max_seconds=max_seconds,
            transform=transform,
            max_samples=config.get("voxceleb_max_samples"),
        ))
    if commonvoice_dir is not None:
        parts.append(CommonVoiceDataset(
            commonvoice_dir,
            max_seconds=max_seconds,
            transform=transform,
            max_samples=config.get("commonvoice_max_samples"),
        ))
    train_dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]

    sampler = build_weighted_sampler(
        train_dataset, max_samples=config.get("max_train_samples")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,           # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── dev set ──────────────────────────────────────────────────────────────
    dev_dataset = ASVspoof2019LADataset(
        base_dir, split="dev", max_seconds=max_seconds, transform=None
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"\nTrain batches/epoch: {len(train_loader)} "
          f"(sampled {sampler.num_samples} from {len(train_dataset)} total)")
    print(f"Dev   batches: {len(dev_loader)}")

    return train_loader, dev_loader


def build_eval_loaders(config: dict) -> dict[str, DataLoader]:
    """
    Build eval DataLoaders for all three test sets.

    config keys used:
        base_dir          : root of asvspoof2019_LA
        asvspoof2021_dir  : root of asvspoof2021_DF
        in_the_wild_dir   : root of in_the_wild
        batch_size        : int
        num_workers       : int

    Returns:
        dict with keys: "asvspoof19_eval", "asvspoof21_df", "in_the_wild"
        All loaders use clean audio (no transform).
    """
    batch_size      = config["batch_size"]
    num_workers     = config["num_workers"]
    max_seconds     = config.get("max_seconds", 4.0)

    def _loader(dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    loaders = {}

    # ASVspoof 2019 LA eval (in-domain)
    loaders["asvspoof19_eval"] = _loader(
        ASVspoof2019LADataset(
            config["base_dir"], split="eval", max_seconds=max_seconds, transform=None
        )
    )

    # ASVspoof 2021 DF eval (cross-attack + codec)
    if config.get("asvspoof2021_dir"):
        loaders["asvspoof21_df"] = _loader(
            ASVspoof2021DFDataset(config["asvspoof2021_dir"], max_seconds=max_seconds)
        )

    # In-the-Wild (held-out generalization — run last)
    if config.get("in_the_wild_dir"):
        loaders["in_the_wild"] = _loader(
            InTheWildDataset(config["in_the_wild_dir"], max_seconds=max_seconds)
        )

    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)} samples, {len(loader)} batches")

    return loaders
