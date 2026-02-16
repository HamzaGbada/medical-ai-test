"""
DataLoader factory for PneumoniaMNIST.

Creates train, validation, and test DataLoaders with configurable
batch size and number of workers.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader

from data.dataset import PneumoniaMNISTDataset
from data.transforms import get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader creation."""

    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    num_channels: int = 1
    image_size: int = 28
    data_root: str = "./data_cache"


def create_dataloaders(
    config: DataLoaderConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        config: DataLoader configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform = get_train_transforms(
        num_channels=config.num_channels,
        image_size=config.image_size,
    )
    eval_transform = get_eval_transforms(
        num_channels=config.num_channels,
        image_size=config.image_size,
    )

    train_dataset = PneumoniaMNISTDataset(
        split="train",
        transform=train_transform,
        num_channels=config.num_channels,
        root=config.data_root,
    )
    val_dataset = PneumoniaMNISTDataset(
        split="val",
        transform=eval_transform,
        num_channels=config.num_channels,
        root=config.data_root,
    )
    test_dataset = PneumoniaMNISTDataset(
        split="test",
        transform=eval_transform,
        num_channels=config.num_channels,
        root=config.data_root,
    )

    logger.info("Class distribution (train): %s", train_dataset.get_class_distribution())
    logger.info("Class distribution (val): %s", val_dataset.get_class_distribution())
    logger.info("Class distribution (test): %s", test_dataset.get_class_distribution())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        "DataLoaders created — Train: %d batches, Val: %d batches, Test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return train_loader, val_loader, test_loader
