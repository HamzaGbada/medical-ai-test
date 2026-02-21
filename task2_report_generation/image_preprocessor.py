"""
Image preprocessing for VLM pipeline.

Handles conversion of PneumoniaMNIST images (28×28 grayscale) to formats
suitable for VLM consumption: base64-encoded PNG strings, upscaled images,
and temporary file paths.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses PneumoniaMNIST images for VLM input.

    Since PneumoniaMNIST images are 28×28 grayscale, they are upscaled
    to a target size (default 224×224) for better VLM processing, converted
    to PNG, and encoded as base64 strings.
    """

    def __init__(
        self,
        target_size: int = 224,
        output_dir: str = "./reports/generated_reports/images",
    ) -> None:
        """Initialize preprocessor.

        Args:
            target_size: Upscale target size (square).
            output_dir: Directory for saving temporary image files.
        """
        self.target_size = target_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("ImagePreprocessor initialized (target_size=%d)", target_size)

    def numpy_to_pil(self, image_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL Image.

        Handles both (H, W) grayscale and (H, W, C) formats.

        Args:
            image_array: Image as numpy array (uint8 or float).

        Returns:
            PIL Image in 'L' (grayscale) mode.
        """
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)

        if image_array.ndim == 3:
            # (H, W, 1) → (H, W)
            if image_array.shape[2] == 1:
                image_array = image_array.squeeze(axis=2)
            # (1, H, W) → (H, W) (torch format)
            elif image_array.shape[0] == 1:
                image_array = image_array.squeeze(axis=0)

        return Image.fromarray(image_array, mode="L")

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale image to target size using bicubic interpolation.

        Args:
            image: PIL Image to upscale.

        Returns:
            Upscaled PIL Image.
        """
        return image.resize(
            (self.target_size, self.target_size),
            Image.BICUBIC,
        )

    def to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded PNG string.

        Args:
            image: PIL Image.

        Returns:
            Base64-encoded string of the PNG image.
        """
        buffer = io.BytesIO()
        # Convert grayscale to RGB for better VLM compatibility
        if image.mode == "L":
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def save_image(self, image: Image.Image, name: str) -> str:
        """Save image to disk and return the file path.

        Args:
            image: PIL Image.
            name: Filename (without extension).

        Returns:
            Absolute path to the saved image.
        """
        path = os.path.join(self.output_dir, f"{name}.png")
        if image.mode == "L":
            image = image.convert("RGB")
        image.save(path, format="PNG")
        logger.debug("Saved image: %s", path)
        return os.path.abspath(path)

    def process(
        self,
        image_array: np.ndarray,
        image_id: str = "image",
        save: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """Full preprocessing pipeline: array → upscale → base64 (+ save).

        Args:
            image_array: Raw image numpy array.
            image_id: Identifier for this image.
            save: Whether to also save the image to disk.

        Returns:
            Tuple of (base64_string, file_path or None).
        """
        pil_image = self.numpy_to_pil(image_array)
        upscaled = self.upscale(pil_image)
        b64 = self.to_base64(upscaled)

        file_path = None
        if save:
            file_path = self.save_image(upscaled, image_id)

        logger.info("Processed image '%s': %dx%d → %dx%d, base64 length=%d",
                     image_id, pil_image.size[0], pil_image.size[1],
                     upscaled.size[0], upscaled.size[1], len(b64))

        return b64, file_path
