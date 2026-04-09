"""
Legacy training script — chunked inference approach (0.5s chunks during training).
Superseded by jaylou_scripts/train.py which trains on 4s fixed-length utterances.

Reason deprecated: 0.5s chunks produce only ~25 Wav2Vec2 feature vectors per forward
pass, insufficient temporal context for the model to learn discriminative artifacts.
The model underfit — unable to distinguish bonafide from spoof at that resolution.
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duy_scripts.dataset import ASVspoof2019LADataset
from jaylou_scripts.legacy.config import CONFIG, DEVICE, SEED
from jaylou_scripts.evaluate import compute_eer, plot_results, run_inference
from sklearn.metrics import accuracy_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed}")


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = ASVspoof2019LADataset(
        base_dir=CONFIG["base_dir"],
        split="train",
        use_vad=False,
        overlap_pct=0,
    )
    dev_dataset = ASVspoof2019LADataset(
        base_dir=CONFIG["base_dir"],
        split="dev",
        use_vad=False,
        overlap_pct=0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    return train_loader, dev_loader


def build_model_and_optimiser(
    train_loader: DataLoader,
) -> tuple[
    nn.Module, torch.optim.Optimizer, object, nn.Module, torch.cuda.amp.GradScaler
]:
    model = CONFIG["model"]().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CONFIG["lr_encoder"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["lr_classifier"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps = len(train_loader) * CONFIG["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    class_weights = torch.tensor(CONFIG["class_weights"]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda")

    return model, optimizer, scheduler, loss_fn, scaler


def train(
    model, train_loader, dev_loader, optimizer, scheduler, loss_fn, scaler
) -> dict:
    from tqdm import tqdm

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_name"])

    best_eer = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_eer": [],
    }

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['max_epochs']}")
        for batch in pbar:
            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        labels_dev, scores_dev, preds_dev, val_loss = run_inference(
            model, dev_loader, DEVICE, loss_fn=loss_fn
        )
        history["val_loss"].append(val_loss)

        dev_eer = compute_eer(labels_dev, scores_dev)[3]
        val_acc = accuracy_score(labels_dev, preds_dev)
        history["val_acc"].append(val_acc)
        history["val_eer"].append(dev_eer)

        print(
            f"Epoch {epoch + 1} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Dev EER: {dev_eer * 100:.2f}%"
        )

        if dev_eer < best_eer:
            best_eer = dev_eer
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best EER — model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{CONFIG['patience']}"
            )
            if patience_counter >= CONFIG["patience"]:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    return history
