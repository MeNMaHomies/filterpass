import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm


def compute_eer(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Returns (fpr, tpr, eer_idx, eer)."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return fpr, tpr, idx, eer


def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, preds: np.ndarray
) -> dict:
    _, _, _, eer = compute_eer(labels, scores)
    return {
        "eer":       eer,
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall":    recall_score(labels, preds, average="weighted"),
    }


def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """
    Runs inference over a DataLoader.
    Returns (labels, scores, preds, avg_loss).
    avg_loss is None when loss_fn is not provided.
    """
    model.eval()
    all_labels, all_scores, all_preds = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values)
                if loss_fn is not None:
                    total_loss += loss_fn(logits, labels).item()

            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader) if loss_fn is not None else None
    return (
        np.array(all_labels),
        np.array(all_scores),
        np.array(all_preds),
        avg_loss,
    )


def plot_results(
    labels: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Prints metrics and plots ROC curve + confusion matrix."""
    metrics = compute_metrics(labels, scores, preds)
    fpr, tpr, eer_idx, eer = compute_eer(labels, scores)
    roc_auc = auc(fpr, tpr)

    print("\n--- Final Evaluation Results ---")
    print(f"EER:       {metrics['eer']       * 100:.2f}%")
    print(f"Accuracy:  {metrics['accuracy']  * 100:.2f}%")
    print(f"F1:        {metrics['f1']              :.4f}")
    print(f"Precision: {metrics['precision']       :.4f}")
    print(f"Recall:    {metrics['recall']          :.4f}")
    print(f"AUC:       {roc_auc                    :.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.plot(fpr[eer_idx], tpr[eer_idx], "ro", markersize=8, label=f"EER ({eer*100:.2f}%)")
    ax1.set(
        xlim=[0, 1], ylim=[0, 1.05],
        xlabel="False Positive Rate (Spoof accepted as Real)",
        ylabel="True Positive Rate (Spoof correctly detected)",
        title="Receiver Operating Characteristic (ROC)",
    )
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    cm = confusion_matrix(labels, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
        xticklabels=["Bonafide (0)", "Spoof (1)"],
        yticklabels=["Bonafide (0)", "Spoof (1)"],
    )
    ax2.set(xlabel="Predicted Label", ylabel="True Label", title="Confusion Matrix")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
