"""
Model utility functions.

Provides a factory function to instantiate any supported model by name,
and a helper to count trainable parameters.
"""

import logging
from typing import Dict

import torch.nn as nn

from models.efficientnet_classifier import EfficientNetClassifier
from models.resnet_classifier import ResNetClassifier
from models.unet_classifier import UNetClassifier, UNetPretrainedClassifier

logger = logging.getLogger(__name__)

# Registry of supported models
MODEL_REGISTRY: Dict[str, type] = {
    "unet": UNetClassifier,
    "unet_pretrained": UNetPretrainedClassifier,
    "resnet": ResNetClassifier,
    "efficientnet": EfficientNetClassifier,
}


def get_model(
    model_name: str,
    pretrained: bool = False,
    in_channels: int = 1,
) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: One of 'unet', 'resnet', 'efficientnet'.
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.

    Returns:
        Initialized nn.Module.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name == "unet":
        if pretrained:
            model = UNetPretrainedClassifier(in_channels=in_channels, pretrained=True)
        else:
            model = UNetClassifier(in_channels=in_channels)
    elif model_name == "resnet":
        model = ResNetClassifier(in_channels=in_channels, pretrained=pretrained)
    elif model_name == "efficientnet":
        model = EfficientNetClassifier(in_channels=in_channels, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: {list(MODEL_REGISTRY.keys())}"
        )

    n_params = count_parameters(model)
    logger.info(
        "Created model '%s' (pretrained=%s): %s trainable parameters",
        model_name,
        pretrained,
        f"{n_params:,}",
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
