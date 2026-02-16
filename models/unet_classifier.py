"""
UNet modified for classification on PneumoniaMNIST.

Provides two variants:
1. UNetClassifier — UNet from scratch with encoder-decoder + GAP + FC head
2. UNetPretrainedClassifier — ResNet18 encoder backbone with decoder + GAP + FC head
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class _DoubleConv(nn.Module):
    """Double convolution block: (Conv → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downsampling: MaxPool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    """Upsampling: Upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if sizes don't match
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetClassifier(nn.Module):
    """UNet architecture modified for binary classification.

    Architecture:
        Encoder (4 down blocks) → Bottleneck → Decoder (4 up blocks)
        → Global Average Pooling → FC(1)

    The segmentation head is removed; instead, we use GAP over the final
    decoder feature map followed by a fully connected layer.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32) -> None:
        """Initialize UNet classifier.

        Args:
            in_channels: Input image channels (1 for grayscale).
            base_filters: Base number of filters (doubled at each level).
        """
        super().__init__()
        bf = base_filters

        # Encoder
        self.inc = _DoubleConv(in_channels, bf)
        self.down1 = _Down(bf, bf * 2)
        self.down2 = _Down(bf * 2, bf * 4)
        self.down3 = _Down(bf * 4, bf * 8)
        self.down4 = _Down(bf * 8, bf * 16)

        # Decoder
        self.up1 = _Up(bf * 16 + bf * 8, bf * 8)
        self.up2 = _Up(bf * 8 + bf * 4, bf * 4)
        self.up3 = _Up(bf * 4 + bf * 2, bf * 2)
        self.up4 = _Up(bf * 2 + bf, bf)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(bf, 1),
        )

        logger.info("UNetClassifier (scratch) initialized with base_filters=%d", bf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logit of shape (B, 1)."""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Classify
        return self.classifier(x)


class UNetPretrainedClassifier(nn.Module):
    """UNet classifier with a pretrained ResNet18 encoder backbone.

    Uses the first 4 layer groups of ResNet18 as the encoder, with a
    lightweight decoder, followed by GAP + FC(1) for classification.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True) -> None:
        """Initialize UNet with pretrained encoder.

        Args:
            in_channels: Input channels (1 for grayscale).
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt first conv for grayscale
        if in_channels != 3:
            orig_conv = backbone.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Average the RGB weights to single channel
                self.conv1.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder stages
        self.enc1 = backbone.layer1  # 64 channels
        self.enc2 = backbone.layer2  # 128 channels
        self.enc3 = backbone.layer3  # 256 channels
        self.enc4 = backbone.layer4  # 512 channels

        # Lightweight decoder (no skip connections for simplicity)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

        logger.info("UNetPretrainedClassifier initialized (pretrained=%s)", pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logit of shape (B, 1)."""
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Classify
        return self.classifier(x)
