"""
Configuration for Task 3 Retrieval System.

All settings configurable via environment variables.

Model selection:
    EMBEDDING_MODEL=resnet18   → ResNet18 Task 1 checkpoint (512-d)  [default]
    EMBEDDING_MODEL=biovil     → microsoft/BiomedVLP-BioViL-T (128-d, text+image)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Dimension mapping per model
_MODEL_DIMS = {
    "resnet18": 512,
    "biovil": 128,
}


class Settings:
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )

    # Embedding model selection
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "resnet18")

    # ResNet18 settings
    embedding_checkpoint: str = os.getenv(
        "EMBEDDING_CHECKPOINT",
        "./checkpoints/resnet_pretrained_best.pth",
    )

    # BioViL-T settings
    biovil_model_id: str = os.getenv(
        "BIOVIL_MODEL_ID",
        "microsoft/BiomedVLP-BioViL-T",
    )

    # Data
    data_root: str = os.getenv("DATA_ROOT", "./data_cache")

    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    ivfflat_lists: int = int(os.getenv("IVFFLAT_LISTS", "100"))

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension based on the selected model."""
        dim_env = os.getenv("EMBEDDING_DIM")
        if dim_env:
            return int(dim_env)
        return _MODEL_DIMS.get(self.embedding_model.lower(), 512)

    @property
    def supports_text_search(self) -> bool:
        """Return True if the selected model supports text-to-image search."""
        return self.embedding_model.lower() == "biovil"


settings = Settings()
