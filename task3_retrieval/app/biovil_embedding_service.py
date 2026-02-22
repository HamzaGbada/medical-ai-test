"""
BioViL-T Embedding Service for medical image retrieval.

Uses `microsoft/BiomedVLP-BioViL-T` — a hybrid Vision Transformer + ResNet-50
image encoder jointly trained with CXR-BERT in a multi-modal contrastive
framework on 227k MIMIC-CXR chest radiograph + report pairs.

Architecture breakdown:
    - Image encoder : `health_multimodal.image` (ResNet-50 + ViT hybrid)
                      Loaded via `get_biovil_t_image_encoder()`.
                      `get_patchwise_projected_embeddings()` → (B, H, W, 128).
                      Global embedding = spatial mean-pool → (B, 128).
    - Text encoder  : `microsoft/BiomedVLP-BioViL-T` via HuggingFace AutoModel
                      → CXRBertModel.get_projected_text_embeddings() → (B, 128).
    - Both live in the same 128-d joint space (cosine comparable).

Key capabilities:
    - Image-to-image retrieval (joint space cosine similarity)
    - Text-to-image retrieval (FULLY SUPPORTED — joint space)
    - Zero-shot: no PneumoniaMNIST fine-tuning required

References:
    Bannur et al., "Learning to Exploit Temporal Structure for Biomedical
    Vision–Language Processing", CVPR 2023. arXiv:2301.04558.
    HuggingFace: https://huggingface.co/microsoft/BiomedVLP-BioViL-T
    Image model: https://github.com/microsoft/hi-ml (hi-ml-multimodal package)

Requirements:
    hi-ml-multimodal >= 0.2.2
    transformers >= 4.45.0
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

BIOVIL_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
BIOVIL_EMBEDDING_DIM = 128
# BioViL-T image encoder: center-crop to 480×480 after 512 resize
BIOVIL_IMAGE_RESIZE = 512
BIOVIL_IMAGE_CROP = 480


class BioViLTEmbeddingService:
    """Extracts 128-d joint image and text embeddings using BioViL-T.

    Image embeddings and text embeddings live in the same cosine-compatible
    space — enabling both image-to-image AND text-to-image retrieval.

    Args:
        device: Torch device. Auto-detected if None.
        model_id: HuggingFace model identifier for the text encoder.
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

        logger.info(
            "Loading BioViL-T image encoder (hi-ml-multimodal) on %s ...", self.device
        )
        self.image_model, self.image_transform = self._load_image_model()

        logger.info(
            "Loading BioViL-T text encoder (%s) on %s ...", model_id, self.device
        )
        self.text_model, self.tokenizer = self._load_text_model()

        logger.info(
            "BioViLTEmbeddingService ready: dim=%d, device=%s",
            self.embedding_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_image_model(self):
        """Load BioViL-T image encoder from hi-ml-multimodal package."""
        try:
            from health_multimodal.image.model.pretrained import (
                get_biovil_t_image_encoder,
            )
            from health_multimodal.image.data.transforms import (
                create_chest_xray_transform_for_inference,
            )
        except ImportError as e:
            raise ImportError(
                "hi-ml-multimodal is required for BioViL-T image embeddings. "
                "Install with: uv pip install hi-ml-multimodal"
            ) from e

        model = get_biovil_t_image_encoder()
        model = model.to(self.device).eval()

        # health_multimodal expects grayscale images ([1, H, W] tensor)
        # Do NOT convert to RGB — the transform validates for single-channel input
        transform = create_chest_xray_transform_for_inference(
            resize=BIOVIL_IMAGE_RESIZE,
            center_crop_size=BIOVIL_IMAGE_CROP,
        )
        return model, transform

    def _load_text_model(self):
        """Load CXR-BERT text encoder from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers>=4.45.0 required. "
                "Run: uv pip install 'transformers>=4.45.0'"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = model.to(self.device).eval()
        return model, tokenizer

    # ------------------------------------------------------------------
    # Image embeddings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract a 128-d L2-normalised image embedding.

        Args:
            image: PIL Image in any mode (will be converted to grayscale).

        Returns:
            L2-normalised ndarray of shape (128,).
        """
        # health_multimodal transform expects grayscale [1, H, W] input
        image_gray = image.convert("L")
        tensor = self.image_transform(image_gray).unsqueeze(0).to(self.device)

        # Output: (1, H_patches, W_patches, 128) — already L2-normalised per patch
        patch_embeddings = self.image_model.get_patchwise_projected_embeddings(
            tensor, normalize=True
        )
        # Global embedding: mean-pool over spatial patch dimensions (H, W)
        global_emb = patch_embeddings.mean(dim=(1, 2))
        embedding = global_emb.cpu().numpy().flatten()

        return self._l2_normalise(embedding)

    @torch.no_grad()
    def get_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract 128-d L2-normalised embeddings for a batch of images.

        Args:
            images: List of PIL Images.

        Returns:
            L2-normalised embedding matrix of shape (N, 128).
        """
        # Convert to grayscale for health_multimodal transform
        tensors = torch.stack(
            [self.image_transform(img.convert("L")) for img in images]
        ).to(self.device)

        # (B, H, W, 128) → mean over spatial dims → (B, 128)
        patch_embeddings = self.image_model.get_patchwise_projected_embeddings(
            tensors, normalize=True
        )
        global_embeddings = patch_embeddings.mean(dim=(1, 2))
        embeddings = global_embeddings.cpu().numpy()

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
        if image_array.ndim == 2:
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode="L")
        else:
            pil_image = Image.fromarray(image_array.astype(np.uint8))
        return self.get_image_embedding(pil_image)

    # ------------------------------------------------------------------
    # Text embeddings
    # ------------------------------------------------------------------

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

        embedding = self.text_model.get_projected_text_embeddings(
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
