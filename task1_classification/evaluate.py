"""
Evaluation script for Pneumonia Detection models.

Computes metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
generates visualizations (confusion matrix, ROC curve, training curves,
failure cases), and saves everything to the reports directory.

Usage:
    python -m task1_classification.evaluate --model resnet --pretrained true
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloaders import DataLoaderConfig, create_dataloaders
from data.dataset import PneumoniaMNISTDataset
from models.utils import get_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor]]:
    """Run inference and collect predictions.

    Returns:
        Tuple of (true_labels, predicted_labels, probabilities, images_list).
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_images = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        all_labels.extend(labels.numpy().flatten().tolist())
        all_probs.extend(probs.tolist())
        all_images.append(images.cpu())

    true_labels = np.array(all_labels)
    probabilities = np.array(all_probs)
    pred_labels = (probabilities >= 0.5).astype(int)
    images_tensor = torch.cat(all_images, dim=0)

    return true_labels, pred_labels, probabilities, images_tensor


def compute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute all classification metrics.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and roc_auc.
    """
    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
    }
    return metrics


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: str,
    experiment_name: str,
) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, pred_labels)
    class_names = PneumoniaMNISTDataset.get_class_names()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: str,
    experiment_name: str,
) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc = roc_auc_score(true_labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {experiment_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curve saved → %s", save_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    experiment_name: str,
) -> None:
    """Generate and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", lw=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", lw=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # AUC curves
    axes[2].plot(epochs, history["train_auc"], "b-", label="Train AUC", lw=2)
    axes[2].plot(epochs, history["val_auc"], "r-", label="Val AUC", lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Training & Validation AUC")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Training Curves — {experiment_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved → %s", save_path)


def plot_failure_cases(
    images: torch.Tensor,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: str,
    experiment_name: str,
    max_cases: int = 16,
) -> None:
    """Visualize misclassified images."""
    class_names = PneumoniaMNISTDataset.get_class_names()

    # Find misclassified indices
    misclassified = np.where(true_labels != pred_labels)[0]

    if len(misclassified) == 0:
        logger.info("No misclassified cases found!")
        return

    n_show = min(max_cases, len(misclassified))
    indices = misclassified[:n_show]

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()

    for i, idx in enumerate(indices):
        img = images[idx].squeeze().numpy()
        # Denormalize
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap="gray")
        true_cls = class_names[int(true_labels[idx])]
        pred_cls = class_names[int(pred_labels[idx])]
        prob = probabilities[idx]
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({prob:.2f})",
            fontsize=8,
            color="red",
        )
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Failure Cases — {experiment_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Failure cases saved → %s (showing %d/%d)", save_path, n_show, len(misclassified))


def evaluate(
    model_name: str,
    pretrained: bool,
    checkpoint_dir: str = "./checkpoints",
    results_dir: str = "./results",
    reports_dir: str = "./reports",
    data_root: str = "./data_cache",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Dict[str, float]:
    """Full evaluation pipeline for a single model.

    Loads the best checkpoint, runs inference on the test set, computes
    metrics, and generates all visualization plots.

    Args:
        model_name: Model architecture name.
        pretrained: Whether pretrained weights were used.
        checkpoint_dir: Directory containing model checkpoints.
        results_dir: Directory containing training results.
        reports_dir: Directory to save plots and reports.
        data_root: Path to dataset cache.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary of test metrics.
    """
    experiment_name = f"{model_name}_{'pretrained' if pretrained else 'scratch'}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Evaluating: %s", experiment_name)

    # Create output dirs
    os.makedirs(reports_dir, exist_ok=True)

    # Load model
    num_channels = 1
    model = get_model(model_name, pretrained=pretrained, in_channels=num_channels)
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("No checkpoint found at %s, evaluating with current weights", checkpoint_path)

    model = model.to(device)

    # Create test data loader
    dl_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        num_channels=num_channels,
        data_root=data_root,
    )
    _, _, test_loader = create_dataloaders(dl_config)

    # Get predictions
    true_labels, pred_labels, probabilities, images = get_predictions(
        model, test_loader, device
    )

    # Compute metrics
    metrics = compute_metrics(true_labels, pred_labels, probabilities)
    logger.info("Test Metrics for %s:", experiment_name)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    # Generate plots
    plot_confusion_matrix(
        true_labels, pred_labels,
        os.path.join(reports_dir, f"{experiment_name}_confusion_matrix.png"),
        experiment_name,
    )
    plot_roc_curve(
        true_labels, probabilities,
        os.path.join(reports_dir, f"{experiment_name}_roc_curve.png"),
        experiment_name,
    )
    plot_failure_cases(
        images, true_labels, pred_labels, probabilities,
        os.path.join(reports_dir, f"{experiment_name}_failure_cases.png"),
        experiment_name,
    )

    # Load and plot training history if available
    results_path = os.path.join(results_dir, f"{experiment_name}_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        if "history" in results:
            plot_training_curves(
                results["history"],
                os.path.join(reports_dir, f"{experiment_name}_training_curves.png"),
                experiment_name,
            )

    # Save test metrics
    test_metrics_path = os.path.join(results_dir, f"{experiment_name}_test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Test metrics saved → %s", test_metrics_path)

    # Print classification report
    class_names = PneumoniaMNISTDataset.get_class_names()
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    logger.info("\nClassification Report:\n%s", report)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet", "resnet", "efficientnet"],
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether pretrained weights were used",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="task1_classification/config.yaml",
        help="Path to config YAML",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pretrained = args.pretrained.lower() == "true"

    evaluate(
        model_name=args.model,
        pretrained=pretrained,
        checkpoint_dir=config["output"]["checkpoint_dir"],
        results_dir=config["output"]["results_dir"],
        reports_dir=config["output"]["reports_dir"],
        data_root=config["data"]["data_root"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )


if __name__ == "__main__":
    main()
