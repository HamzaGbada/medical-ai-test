"""
EfficientNet-B0 classifier for PneumoniaMNIST.

Provides both scratch and pretrained (ImageNet) variants,
adapted for 1-channel grayscale 28x28 input with binary output.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 adapted for binary classification on grayscale images.

    Modifications:
        - First convolutional layer accepts 1-channel input
        - Classifier head outputs 1 logit for BCEWithLogitsLoss
        - For pretrained: RGB conv weights averaged to single channel
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = False) -> None:
        """Initialize EfficientNet-B0 classifier.

        Args:
            in_channels: Input channels (1 for grayscale, 3 for RGB).
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        # Adapt first conv for grayscale input
        if in_channels != 3:
            orig_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            if pretrained:
                # Average pretrained RGB weights → single channel
                self.model.features[0][0].weight.data = orig_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        # Replace classifier head for binary output
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )

        logger.info(
            "EfficientNetClassifier initialized (pretrained=%s, in_channels=%d)",
            pretrained,
            in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logit of shape (B, 1)."""
        return self.model(x)
