"""
Data transforms and augmentations for medical image classification.

Provides separate transform pipelines for training (with augmentation)
and evaluation (normalization only). Supports both grayscale (1-ch)
and RGB (3-ch) modes for pretrained model compatibility.
"""

from torchvision import transforms as T


# PneumoniaMNIST mean/std (approximate, computed from training set)
# These are close to the overall MedMNIST statistics for grayscale data.
PNEUMONIA_MEAN_1CH = [0.5]
PNEUMONIA_STD_1CH = [0.5]

PNEUMONIA_MEAN_3CH = [0.5, 0.5, 0.5]
PNEUMONIA_STD_3CH = [0.5, 0.5, 0.5]


def get_train_transforms(num_channels: int = 1, image_size: int = 28) -> T.Compose:
    """Get training transforms with data augmentation.

    Applies:
        - Random horizontal flip
        - Small random rotations (±10°)
        - Random affine (translation, slight scale)
        - Mild contrast/brightness adjustment
        - Normalization

    Args:
        num_channels: 1 for grayscale, 3 for RGB (pretrained models).
        image_size: Target image size (default 28 for PneumoniaMNIST).

    Returns:
        Composed torchvision transforms.
    """
    mean = PNEUMONIA_MEAN_1CH if num_channels == 1 else PNEUMONIA_MEAN_3CH
    std = PNEUMONIA_STD_1CH if num_channels == 1 else PNEUMONIA_STD_3CH

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(num_channels: int = 1, image_size: int = 28) -> T.Compose:
    """Get evaluation transforms (no augmentation, normalization only).

    Args:
        num_channels: 1 for grayscale, 3 for RGB (pretrained models).
        image_size: Target image size (default 28 for PneumoniaMNIST).

    Returns:
        Composed torchvision transforms.
    """
    mean = PNEUMONIA_MEAN_1CH if num_channels == 1 else PNEUMONIA_MEAN_3CH
    std = PNEUMONIA_STD_1CH if num_channels == 1 else PNEUMONIA_STD_3CH

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
