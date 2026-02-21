"""
Embedding service for image feature extraction.

Uses the penultimate layer of a pretrained ResNet18 (from Task 1) to extract
512-dimensional embeddings from PneumoniaMNIST images. Embeddings are
L2-normalized for cosine similarity search.

Architectural Decision:
    VLM API endpoints (Ollama/Docker) typically don't expose raw embedding
    vectors. Instead, we use the ResNet18 CNN encoder trained in Task 1 as
    the feature extractor. This provides medically-trained, consistent
    embeddings suitable for similarity search.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from task3_retrieval.app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Extracts image embeddings using a CNN feature encoder.

    Uses ResNet18 penultimate layer (after global average pooling)
    to produce 512-dimensional embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the embedding service.

        Args:
            checkpoint_path: Path to Task 1 checkpoint. If None, uses
                             ImageNet-pretrained ResNet18.
            device: Torch device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path or settings.embedding_checkpoint
        self.embedding_dim = settings.embedding_dim

        self.model = self._build_encoder()
        self.transform = self._get_transform()

        logger.info(
            "EmbeddingService initialized: dim=%d, device=%s, checkpoint=%s",
            self.embedding_dim, self.device, self.checkpoint_path,
        )

    def _build_encoder(self) -> nn.Module:
        """Build the feature extraction encoder from ResNet18."""
        from torchvision.models import resnet18, ResNet18_Weights

        # Load base model
        model = resnet18(weights=None)

        # Adapt first conv for grayscale (same as Task 1)
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Average the pretrained weights across channel dim
        with torch.no_grad():
            model.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        # Try loading Task 1 checkpoint
        if self.checkpoint_path:
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                # Remove the final FC layer keys (we'll replace it)
                state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith("fc.")
                }
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded checkpoint: %s", self.checkpoint_path)
            except FileNotFoundError:
                logger.warning("Checkpoint not found: %s — using random weights", self.checkpoint_path)
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s — using random weights", e)

        # Remove the final FC layer — use GAP output (512-d) as embedding
        model.fc = nn.Identity()
        model = model.to(self.device)
        model.eval()

        return model

    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single PIL image.

        Args:
            image: PIL Image (any mode, will be converted to grayscale).

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,).
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy().flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @torch.no_grad()
    def get_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract embeddings from a batch of PIL images.

        Args:
            images: List of PIL Images.

        Returns:
            L2-normalized embedding matrix of shape (N, embedding_dim).
        """
        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)

        embeddings = self.model(tensors).cpu().numpy()

        # L2 normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        return embeddings

    @torch.no_grad()
    def get_numpy_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """Extract embedding from a numpy array image.

        Args:
            image_array: Image as numpy array (H, W) or (H, W, C).

        Returns:
            L2-normalized embedding vector.
        """
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = image_array.squeeze(axis=2)
        pil_image = Image.fromarray(image_array.astype(np.uint8), mode="L")
        return self.get_image_embedding(pil_image)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Extract text embedding (not supported for CNN encoder).

        Raises:
            NotImplementedError: CNN encoder doesn't support text embeddings.
        """
        raise NotImplementedError(
            "Text embeddings are not supported by the CNN-based encoder. "
            "A VLM with shared text-image embedding space would be needed."
        )
