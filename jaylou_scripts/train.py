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

from jaylou_scripts.dataset import ASVspoofDataset
from jaylou_scripts.config import CONFIG, DEVICE, SEED
from jaylou_scripts.evaluate import compute_eer, plot_results, run_inference
from sklearn.metrics import accuracy_score


def configure_cuda() -> None:
    """
    Ada Lovelace (RTX 40xx) TF32 — uses 10-bit mantissa instead of 23-bit for
    matmul/conv ops. Negligible accuracy loss, ~1.7x throughput on tensor cores.
    Both flags are True by default in PyTorch ≥ 1.12, but set explicitly to be safe.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark=True finds the fastest cuDNN conv algorithm for our fixed input
    # shape (64 000 samples). Trade-off: non-deterministic results.
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True (set in configure_cuda) takes precedence — deterministic
    # is left False so cuDNN can use the fastest algorithm.
    torch.backends.cudnn.deterministic = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed}")


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = ASVspoofDataset(base_dir=CONFIG["base_dir"], split="train")
    dev_dataset   = ASVspoofDataset(base_dir=CONFIG["base_dir"], split="dev")

    nw = CONFIG["num_workers"]
    pf = CONFIG["prefetch_factor"] if nw > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=pf,
        persistent_workers=nw > 0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=pf,
        persistent_workers=nw > 0,
    )
    return train_loader, dev_loader


def _freeze_encoder_layers(model: nn.Module, n_layers: int) -> None:
    """Freeze the bottom n_layers transformer blocks in the Wav2Vec2 encoder."""
    if n_layers <= 0:
        return
    transformer_layers = model.encoder.encoder.layers
    for layer in transformer_layers[:n_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Frozen Wav2Vec2 transformer layers 0–{n_layers - 1} (of {len(transformer_layers)})")


def build_model_and_optimiser(
    train_loader: DataLoader,
) -> tuple[nn.Module, torch.optim.Optimizer, object, nn.Module, torch.cuda.amp.GradScaler]:
    model = CONFIG["model"](
        model_name=CONFIG["model_name"],
        revision=CONFIG["model_revision"]
    ).to(DEVICE)
    _freeze_encoder_layers(model, CONFIG["freeze_encoder_layers"])

    if CONFIG["gradient_checkpointing"]:
        # Recomputes activations during backward instead of caching them.
        # Reduces activation VRAM ~40% at the cost of ~15% slower backward.
        model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, model.encoder.parameters()), "lr": CONFIG["lr_encoder"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["lr_classifier"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )

    steps_per_epoch = len(train_loader) // CONFIG["grad_accum_steps"]
    total_steps = steps_per_epoch * CONFIG["max_epochs"]
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
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
) -> dict:
    from tqdm import tqdm

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    checkpoint_path  = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_name"])
    history_plot_path = os.path.join(CONFIG["checkpoint_dir"], "training_history.png")

    best_eer = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_eer":    [],
    }

    accum_steps = CONFIG["grad_accum_steps"]

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['max_epochs']}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values)
                loss   = loss_fn(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        avg_loss  = total_loss / len(train_loader)
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

        improvement = best_eer - dev_eer
        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best EER — model saved to {checkpoint_path}")

        if improvement >= CONFIG["min_delta"]:
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  Marginal gain ({improvement * 100:.3f}pp). "
                f"Patience: {patience_counter}/{CONFIG['patience']}"
            )
            if patience_counter >= CONFIG["patience"]:
                print(f"  Early stopping at epoch {epoch + 1} — gains below min_delta.")
                plot_training_history(history, save_path=history_plot_path)
                break

        plot_training_history(history, save_path=history_plot_path)

    return history


def plot_training_history(history: dict, save_path: str | None = None) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Train vs Val Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Train vs Val Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [e * 100 for e in history["val_eer"]], label="Val EER (%)", color="red")
    axes[2].set(xlabel="Epoch", ylabel="EER (%)", title="Validation EER")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    configure_cuda()
    print(f"Training on: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)

    train_loader, dev_loader = build_dataloaders()
    model, optimizer, scheduler, loss_fn, scaler = build_model_and_optimiser(train_loader)

    history = train(model, train_loader, dev_loader, optimizer, scheduler, loss_fn, scaler)

    # --- Evaluation on eval set ---
    # eval_dataset = ASVspoofDataset(CONFIG["base_dir"], split="eval")
    # eval_loader  = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    # checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_name"])
    # model = CONFIG["model"]().to(DEVICE)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    # labels, scores, preds, _ = run_inference(model, eval_loader, DEVICE)
    # plot_results(labels, scores, preds, save_path=os.path.join(CONFIG["checkpoint_dir"], "eval_results.png"))
