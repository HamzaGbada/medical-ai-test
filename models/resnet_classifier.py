"""
ResNet18 classifier for PneumoniaMNIST.

Provides both scratch and pretrained (ImageNet) variants,
adapted for 1-channel grayscale 28x28 input with binary output.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class ResNetClassifier(nn.Module):
    """ResNet18 adapted for binary classification on grayscale images.

    Modifications:
        - First convolutional layer accepts 1-channel input
        - Final FC layer outputs 1 logit for BCEWithLogitsLoss
        - For pretrained: RGB conv1 weights averaged to single channel
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = False) -> None:
        """Initialize ResNet18 classifier.

        Args:
            in_channels: Input channels (1 for grayscale, 3 for RGB).
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Adapt first conv for grayscale input
        if in_channels != 3:
            orig_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Average pretrained RGB weights → single channel
                self.model.conv1.weight.data = orig_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        # Replace final FC for binary output
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1),
        )

        logger.info(
            "ResNetClassifier initialized (pretrained=%s, in_channels=%d)",
            pretrained,
            in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logit of shape (B, 1)."""
        return self.model(x)
