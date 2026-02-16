"""
Dataset module for PneumoniaMNIST.

Wraps the MedMNIST PneumoniaMNIST dataset with proper split handling
and configurable transforms for medical image classification.
"""

import logging
import os
from typing import Optional, Tuple

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

DATASET_INFO = INFO["pneumoniamnist"]


class PneumoniaMNISTDataset(Dataset):
    """Wrapper around MedMNIST PneumoniaMNIST for binary classification.

    Attributes:
        split: One of 'train', 'val', 'test'.
        transform: Torchvision transform pipeline to apply.
        num_channels: Number of output channels (1 for grayscale, 3 for RGB).
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        num_channels: int = 1,
        download: bool = True,
        root: str = "./data_cache",
    ) -> None:
        """Initialize PneumoniaMNIST dataset.

        Args:
            split: Dataset split — 'train', 'val', or 'test'.
            transform: Torchvision transforms to apply to images.
            num_channels: Output channels (1=grayscale, 3=RGB for pretrained).
            download: Whether to download the dataset if not found.
            root: Root directory for storing cached data.
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        assert num_channels in (1, 3), f"num_channels must be 1 or 3, got {num_channels}"

        self.split = split
        self.transform = transform
        self.num_channels = num_channels

        os.makedirs(root, exist_ok=True)

        self._dataset = medmnist.PneumoniaMNIST(
            split=split,
            download=download,
            root=root,
            as_rgb=(num_channels == 3),
        )

        logger.info(
            "Loaded PneumoniaMNIST %s split: %d samples, %d classes (%s)",
            split,
            len(self._dataset),
            DATASET_INFO["n_channels"],
            DATASET_INFO["task"],
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, label_tensor).

        The image is a float tensor of shape (C, 28, 28).
        The label is a float tensor of shape (1,) for BCEWithLogitsLoss.
        """
        image, label = self._dataset[idx]

        # image is a PIL Image from MedMNIST
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        # Label: MedMNIST returns ndarray shape (1,), we want float tensor
        label = torch.tensor(label, dtype=torch.float32).squeeze()

        return image, label

    @property
    def n_samples(self) -> int:
        """Total number of samples in this split."""
        return len(self._dataset)

    @staticmethod
    def get_class_names() -> Tuple[str, str]:
        """Return the class names for the binary task."""
        return ("Normal", "Pneumonia")

    def get_class_distribution(self) -> dict:
        """Return count of each class in this split."""
        labels = self._dataset.labels.flatten()
        unique, counts = np.unique(labels, return_counts=True)
        class_names = self.get_class_names()
        return {class_names[int(u)]: int(c) for u, c in zip(unique, counts)}
