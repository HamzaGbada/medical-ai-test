"""
Encoder factory for Task 3 retrieval system.

Returns the appropriate embedding service based on the configured model name.
Supported models:
    - "resnet18"  : ResNet18 CNN encoder (Task 1 fine-tuned checkpoint, 512-d)
    - "biovil"    : BiomedVLP-BioViL-T (Microsoft, CVPR 2023, 128-d, text+image)

Usage:
    from task3_retrieval.app.encoder_factory import get_embedding_service
    svc = get_embedding_service("biovil")
    embedding = svc.get_image_embedding(pil_image)     # image search
    text_emb  = svc.get_text_embedding("consolidation")  # text search (biovil only)
"""

import logging

from task3_retrieval.app.config import settings

logger = logging.getLogger(__name__)

# Registry of supported model names
_SUPPORTED_MODELS = ("resnet18", "biovil")


def get_embedding_service(model_name: str | None = None):
    """Return the correct embedding service for the given model name.

    Args:
        model_name: One of 'resnet18' or 'biovil'. Defaults to the value
                    of settings.embedding_model (EMBEDDING_MODEL env var).

    Returns:
        An EmbeddingService or BioViLTEmbeddingService instance.

    Raises:
        ValueError: If model_name is not recognised.
    """
    model_name = (model_name or settings.embedding_model).lower().strip()

    if model_name == "resnet18":
        from task3_retrieval.app.embedding_service import EmbeddingService
        logger.info("Encoder: ResNet18 (dim=512, checkpoint=%s)", settings.embedding_checkpoint)
        return EmbeddingService()

    elif model_name == "biovil":
        from task3_retrieval.app.biovil_embedding_service import BioViLTEmbeddingService
        logger.info("Encoder: BioViL-T (dim=128, model=%s)", settings.biovil_model_id)
        return BioViLTEmbeddingService(model_id=settings.biovil_model_id)

    else:
        raise ValueError(
            f"Unknown embedding model '{model_name}'. "
            f"Supported: {_SUPPORTED_MODELS}. "
            f"Set EMBEDDING_MODEL environment variable to one of these values."
        )
