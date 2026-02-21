"""
Configuration for Task 3 Retrieval System.

All settings configurable via environment variables.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )

    # Embedding
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "512"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "resnet18")
    embedding_checkpoint: str = os.getenv(
        "EMBEDDING_CHECKPOINT",
        "./checkpoints/resnet_pretrained_best.pth",
    )

    # Data
    data_root: str = os.getenv("DATA_ROOT", "./data_cache")

    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    ivfflat_lists: int = int(os.getenv("IVFFLAT_LISTS", "100"))

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
