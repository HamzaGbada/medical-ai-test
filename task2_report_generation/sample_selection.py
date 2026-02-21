"""
Sample selection for Task 2 report generation.

Selects at least 10 images from the PneumoniaMNIST test set:
- 3 correctly classified as Normal
- 3 correctly classified as Pneumonia
- 4 misclassified by the CNN

Falls back to random selection if Task 1 results are not available.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SelectedSample:
    """A selected image sample with metadata."""
    index: int
    image: np.ndarray  # Raw pixel array
    ground_truth: int  # 0=Normal, 1=Pneumonia
    cnn_prediction: int  # CNN predicted class
    cnn_confidence: float  # CNN probability


CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def load_cnn_predictions(
    model_name: str = "resnet",
    pretrained: bool = True,
    checkpoint_dir: str = "./checkpoints",
    data_root: str = "./data_cache",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CNN predictions from a Task 1 trained model.

    Runs inference on the test set with the best checkpoint.

    Args:
        model_name: Model architecture name.
        pretrained: Whether pretrained weights were used.
        checkpoint_dir: Directory with saved checkpoints.
        data_root: Dataset cache directory.

    Returns:
        Tuple of (true_labels, predicted_labels, probabilities).
    """
    experiment_name = f"{model_name}_{'pretrained' if pretrained else 'scratch'}"
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")

    if not os.path.exists(checkpoint_path):
        logger.warning("No checkpoint found at %s", checkpoint_path)
        return None, None, None

    logger.info("Loading CNN predictions from: %s", checkpoint_path)

    # Import here to avoid circular dependency
    from data.dataloaders import DataLoaderConfig, create_dataloaders
    from models.utils import get_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, pretrained=pretrained, in_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dl_config = DataLoaderConfig(
        batch_size=64,
        num_workers=0,
        num_channels=1,
        data_root=data_root,
    )
    _, _, test_loader = create_dataloaders(dl_config)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_labels.extend(labels.numpy().flatten().tolist())
            all_probs.extend(probs.tolist())

    true_labels = np.array(all_labels)
    probabilities = np.array(all_probs)
    pred_labels = (probabilities >= 0.5).astype(int)

    logger.info("CNN predictions loaded: %d samples, accuracy=%.4f",
                len(true_labels), np.mean(true_labels == pred_labels))

    return true_labels, pred_labels, probabilities


def select_samples(
    n_normal: int = 3,
    n_pneumonia: int = 3,
    n_misclassified: int = 4,
    model_name: str = "resnet",
    pretrained: bool = True,
    checkpoint_dir: str = "./checkpoints",
    data_root: str = "./data_cache",
    seed: int = 42,
) -> List[SelectedSample]:
    """Select diverse samples for VLM evaluation.

    Selects:
    - n_normal correctly classified Normal images
    - n_pneumonia correctly classified Pneumonia images
    - n_misclassified images that the CNN got wrong

    Args:
        n_normal: Number of correct Normal samples.
        n_pneumonia: Number of correct Pneumonia samples.
        n_misclassified: Number of CNN-misclassified samples.
        model_name: CNN model name for loading predictions.
        pretrained: Whether the CNN used pretrained weights.
        checkpoint_dir: Checkpoint directory.
        data_root: Dataset cache directory.
        seed: Random seed for reproducibility.

    Returns:
        List of SelectedSample instances.
    """
    random.seed(seed)
    np.random.seed(seed)

    import medmnist

    os.makedirs(data_root, exist_ok=True)
    test_dataset = medmnist.PneumoniaMNIST(
        split="test", download=True, root=data_root
    )

    # Try loading CNN predictions
    true_labels, pred_labels, probabilities = load_cnn_predictions(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
    )

    if true_labels is None:
        logger.warning("No CNN predictions available — using random selection with synthetic predictions")
        labels_arr = test_dataset.labels.flatten()
        n_total = n_normal + n_pneumonia + n_misclassified

        normal_indices = np.where(labels_arr == 0)[0]
        pneumonia_indices = np.where(labels_arr == 1)[0]

        selected_normal = np.random.choice(normal_indices, min(n_normal, len(normal_indices)), replace=False)
        selected_pneumonia = np.random.choice(pneumonia_indices, min(n_pneumonia, len(pneumonia_indices)), replace=False)
        remaining = np.random.choice(len(labels_arr), n_misclassified, replace=False)

        all_indices = list(selected_normal) + list(selected_pneumonia) + list(remaining)
        samples = []
        for idx in all_indices:
            img, lbl = test_dataset[idx]
            img_arr = np.array(img)
            gt = int(lbl.flatten()[0])
            samples.append(SelectedSample(
                index=int(idx),
                image=img_arr,
                ground_truth=gt,
                cnn_prediction=gt,  # No CNN available
                cnn_confidence=0.5,
            ))
        return samples

    # Select correctly classified Normal
    correct_normal = np.where((true_labels == 0) & (pred_labels == 0))[0]
    correct_pneumonia = np.where((true_labels == 1) & (pred_labels == 1))[0]
    misclassified = np.where(true_labels != pred_labels)[0]

    logger.info("Available for selection: %d correct Normal, %d correct Pneumonia, %d misclassified",
                len(correct_normal), len(correct_pneumonia), len(misclassified))

    # Select with fallback if not enough samples
    sel_normal = np.random.choice(
        correct_normal, min(n_normal, len(correct_normal)), replace=False
    ).tolist()
    sel_pneumonia = np.random.choice(
        correct_pneumonia, min(n_pneumonia, len(correct_pneumonia)), replace=False
    ).tolist()
    sel_misc = np.random.choice(
        misclassified, min(n_misclassified, len(misclassified)), replace=False
    ).tolist()

    all_indices = sel_normal + sel_pneumonia + sel_misc
    logger.info("Selected %d samples: %d Normal, %d Pneumonia, %d misclassified",
                len(all_indices), len(sel_normal), len(sel_pneumonia), len(sel_misc))

    # Build sample objects
    samples = []
    for idx in all_indices:
        img, lbl = test_dataset[idx]
        img_arr = np.array(img)
        gt = int(true_labels[idx])
        pred = int(pred_labels[idx])
        conf = float(probabilities[idx])

        samples.append(SelectedSample(
            index=idx,
            image=img_arr,
            ground_truth=gt,
            cnn_prediction=pred,
            cnn_confidence=conf,
        ))

    return samples


def save_selection_info(
    samples: List[SelectedSample],
    output_path: str = "./results/selected_samples.json",
) -> None:
    """Save sample selection metadata to JSON.

    Args:
        samples: List of selected samples.
        output_path: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    info = []
    for s in samples:
        info.append({
            "index": s.index,
            "ground_truth": s.ground_truth,
            "ground_truth_name": CLASS_NAMES[s.ground_truth],
            "cnn_prediction": s.cnn_prediction,
            "cnn_prediction_name": CLASS_NAMES[s.cnn_prediction],
            "cnn_confidence": round(s.cnn_confidence, 4),
            "correctly_classified": s.ground_truth == s.cnn_prediction,
        })

    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Sample selection saved → %s", output_path)
