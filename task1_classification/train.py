"""
Training script for Pneumonia Detection models.

Supports configurable hyperparameters, early stopping, checkpoint saving,
GPU/CPU auto-detection, and reproducibility via seed setting.

Usage:
    python -m task1_classification.train --model resnet --pretrained --epochs 30
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloaders import DataLoaderConfig, create_dataloaders
from models.utils import count_parameters, get_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def get_device() -> torch.device:
    """Auto-detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Validation metric (higher is better, e.g. AUC).

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d/%d", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy, auc).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[float] = []
    all_probs: List[float] = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) if labels.dim() == 1 else labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach()
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().flatten().tolist())
        all_probs.extend(probs.cpu().numpy().flatten().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, accuracy, auc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate the model.

    Returns:
        Tuple of (average_loss, accuracy, auc).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[float] = []
    all_probs: List[float] = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) if labels.dim() == 1 else labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().flatten().tolist())
        all_probs.extend(probs.cpu().numpy().flatten().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, accuracy, auc


def train(
    model_name: str,
    pretrained: bool,
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 7,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    seed: int = 42,
    data_root: str = "./data_cache",
    checkpoint_dir: str = "./checkpoints",
    results_dir: str = "./results",
    num_workers: int = 2,
) -> Dict:
    """Full training pipeline for a single model.

    Args:
        model_name: Model architecture name ('unet', 'resnet', 'efficientnet').
        pretrained: Whether to use pretrained weights.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for data loading.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        patience: Early stopping patience.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR scheduler reduction factor.
        seed: Random seed for reproducibility.
        data_root: Path to dataset cache.
        checkpoint_dir: Directory to save model checkpoints.
        results_dir: Directory to save training results.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)
    device = get_device()

    experiment_name = f"{model_name}_{'pretrained' if pretrained else 'scratch'}"
    logger.info("=" * 60)
    logger.info("Starting experiment: %s", experiment_name)
    logger.info("=" * 60)

    # Determine number of channels
    # Pretrained models expect 3 channels for weight reuse, but we adapt conv1
    # So we can use 1 channel for all models
    num_channels = 1

    # Create data loaders
    dl_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        num_channels=num_channels,
        data_root=data_root,
    )
    train_loader, val_loader, _ = create_dataloaders(dl_config)

    # Create model
    model = get_model(model_name, pretrained=pretrained, in_channels=num_channels)
    model = model.to(device)
    n_params = count_parameters(model)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )
    early_stopping = EarlyStopping(patience=patience)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": [],
        "lr": [],
    }

    best_val_auc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)

        # Train
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.info(
            "Train — Loss: %.4f | Acc: %.4f | AUC: %.4f",
            train_loss, train_acc, train_auc,
        )
        logger.info(
            "Val   — Loss: %.4f | Acc: %.4f | AUC: %.4f | LR: %.6f",
            val_loss, val_acc, val_auc, current_lr,
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "model_name": model_name,
                    "pretrained": pretrained,
                },
                checkpoint_path,
            )
            logger.info("Saved best model (AUC=%.4f) → %s", val_auc, checkpoint_path)

        # Early stopping
        if early_stopping(val_auc):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    training_time = time.time() - start_time

    # Save results
    results = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "n_parameters": n_params,
        "best_val_auc": best_val_auc,
        "best_val_acc": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_trained": len(history["train_loss"]),
        "training_time_seconds": round(training_time, 2),
        "history": history,
    }

    results_path = os.path.join(results_dir, f"{experiment_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)
    logger.info(
        "Training complete: %s — Best Val AUC: %.4f — Time: %.1fs",
        experiment_name,
        best_val_auc,
        training_time,
    )

    return results


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet", "resnet", "efficientnet"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to use pretrained weights",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        default="task1_classification/config.yaml",
        help="Path to config YAML",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    # CLI overrides
    pretrained = args.pretrained.lower() == "true"
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["learning_rate"]
    seed = args.seed or train_cfg["seed"]

    train(
        model_name=args.model,
        pretrained=pretrained,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=train_cfg["weight_decay"],
        patience=train_cfg["early_stopping_patience"],
        scheduler_patience=train_cfg["scheduler_patience"],
        scheduler_factor=train_cfg["scheduler_factor"],
        seed=seed,
        data_root=data_cfg["data_root"],
        checkpoint_dir=output_cfg["checkpoint_dir"],
        results_dir=output_cfg["results_dir"],
        num_workers=data_cfg["num_workers"],
    )


if __name__ == "__main__":
    main()
