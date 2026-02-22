"""
BioViL-T Embedding Service for medical image retrieval.

Uses `microsoft/BiomedVLP-BioViL-T` — a hybrid Vision Transformer + ResNet-50
image encoder jointly trained with CXR-BERT in a multi-modal contrastive
framework on 227k MIMIC-CXR chest radiograph + report pairs.

Key advantages over ResNet18:
    - CXR-native pretraining (not ImageNet)
    - 128-d joint image+text embedding space
    - Text-to-image search is FULLY SUPPORTED
    - Temporal pathology understanding (CVPR 2023)

References:
    Bannur et al., "Learning to Exploit Temporal Structure for Biomedical
    Vision–Language Processing", CVPR 2023. arXiv:2301.04558.
    HuggingFace: https://huggingface.co/microsoft/BiomedVLP-BioViL-T
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BIOVIL_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
BIOVIL_EMBEDDING_DIM = 128
# BioViL-T image encoder expects 3-channel images at 512×512
BIOVIL_IMAGE_SIZE = 512


class BioViLTEmbeddingService:
    """Extracts 128-d joint image and text embeddings using BioViL-T.

    Image embeddings and text embeddings live in the same cosine-compatible
    space — enabling both image-to-image AND text-to-image retrieval.

    Args:
        device: Torch device. Auto-detected if None.
        model_id: HuggingFace model identifier.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_id: str = BIOVIL_MODEL_ID,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_id = model_id
        self.embedding_dim = BIOVIL_EMBEDDING_DIM

        logger.info("Loading BioViL-T model: %s on %s ...", model_id, self.device)
        self.model, self.tokenizer = self._load_model()
        self.image_transform = self._get_image_transform()

        logger.info(
            "BioViLTEmbeddingService ready: dim=%d, device=%s",
            self.embedding_dim,
            self.device,
        )

    def _load_model(self):
        """Load BioViL-T model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers>=4.45.0 required. Run: pip install 'transformers>=4.45.0'"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = model.to(self.device).eval()
        return model, tokenizer

    def _get_image_transform(self) -> transforms.Compose:
        """Preprocessing pipeline for BioViL-T image input.

        BioViL-T expects 3-channel RGB float tensors at 512×512,
        normalised with ImageNet mean/std (as used in its pretraining).
        """
        return transforms.Compose([
            transforms.Resize((BIOVIL_IMAGE_SIZE, BIOVIL_IMAGE_SIZE)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract a 128-d L2-normalised image embedding.

        Args:
            image: PIL Image in any mode.

        Returns:
            L2-normalised ndarray of shape (128,).
        """
        tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        embedding = self.model.get_projected_global_embedding(
            pixel_values=tensor
        )
        embedding = embedding.cpu().numpy().flatten()
        return self._l2_normalise(embedding)

    @torch.no_grad()
    def get_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract 128-d L2-normalised embeddings for a batch of images.

        Args:
            images: List of PIL Images.

        Returns:
            L2-normalised embedding matrix of shape (N, 128).
        """
        tensors = torch.stack([self.image_transform(img) for img in images])
        tensors = tensors.to(self.device)
        embeddings = self.model.get_projected_global_embedding(
            pixel_values=tensors
        )
        embeddings = embeddings.cpu().numpy()
        return self._l2_normalise_rows(embeddings)

    @torch.no_grad()
    def get_numpy_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """Extract embedding from a raw numpy array image.

        Args:
            image_array: Image as (H, W) greyscale or (H, W, C).

        Returns:
            L2-normalised 128-d embedding.
        """
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = image_array.squeeze(axis=2)
        # Convert greyscale to RGB PIL image
        if image_array.ndim == 2:
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode="L")
        else:
            pil_image = Image.fromarray(image_array.astype(np.uint8))
        return self.get_image_embedding(pil_image)

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Extract a 128-d L2-normalised text embedding.

        The text embedding lives in the same joint space as image embeddings,
        enabling direct text-to-image cosine similarity search.

        Args:
            text: Radiology query string, e.g.
                  "bilateral lower lobe consolidation".

        Returns:
            L2-normalised ndarray of shape (128,).
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        embedding = self.model.get_projected_text_embeddings(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        embedding = embedding.cpu().numpy().flatten()
        return self._l2_normalise(embedding)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)

    @staticmethod
    def _l2_normalise_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return mat / norms
