"""
Retrieval service combining embedding extraction and database search.

Provides high-level retrieval operations used by the FastAPI endpoints.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from task3_retrieval.app import crud
from task3_retrieval.app.config import settings
from task3_retrieval.app.embedding_service import EmbeddingService
from task3_retrieval.app.schemas import SearchResult

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


class RetrievalService:
    """High-level retrieval service for image similarity search.

    Combines embedding extraction with PGVector-backed similarity search.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None) -> None:
        """Initialize the retrieval service.

        Args:
            embedding_service: Pre-initialized EmbeddingService. If None,
                               a new one is created.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        logger.info("RetrievalService initialized")

    def search_by_image_path(
        self,
        db: Session,
        image_path: str,
        top_k: int = 5,
    ) -> Dict:
        """Search for similar images given an image file path.

        Args:
            db: Database session.
            image_path: Path to the query image.
            top_k: Number of results.

        Returns:
            Dict with query info and search results.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)
        embedding = self.embedding_service.get_image_embedding(image)

        results = crud.search_similar(db, embedding, top_k=top_k)

        return {
            "query_label": None,
            "query_label_name": None,
            "results": [
                SearchResult(
                    image_id=img.image_id,
                    label=img.label,
                    label_name=CLASS_NAMES.get(img.label, "Unknown"),
                    score=round(score, 4),
                )
                for img, score in results
            ],
        }

    def search_by_image_array(
        self,
        db: Session,
        image_array: np.ndarray,
        top_k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar images given a numpy array.

        Args:
            db: Database session.
            image_array: Image as numpy array.
            top_k: Number of results.
            exclude_id: Optional image_id to exclude.

        Returns:
            List of SearchResult objects.
        """
        embedding = self.embedding_service.get_numpy_embedding(image_array)
        results = crud.search_similar(db, embedding, top_k=top_k, exclude_id=exclude_id)

        return [
            SearchResult(
                image_id=img.image_id,
                label=img.label,
                label_name=CLASS_NAMES.get(img.label, "Unknown"),
                score=round(score, 4),
            )
            for img, score in results
        ]

    def search_by_text(
        self,
        db: Session,
        query_text: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for images by text query.

        Args:
            db: Database session.
            query_text: Text description.
            top_k: Number of results.

        Returns:
            List of SearchResult objects.

        Raises:
            NotImplementedError: CNN encoder doesn't support text embeddings.
        """
        embedding = self.embedding_service.get_text_embedding(query_text)
        results = crud.search_similar(db, embedding, top_k=top_k)

        return [
            SearchResult(
                image_id=img.image_id,
                label=img.label,
                label_name=CLASS_NAMES.get(img.label, "Unknown"),
                score=round(score, 4),
            )
            for img, score in results
        ]

    def build_index(
        self,
        db: Session,
        split: str = "test",
        batch_size: int = 64,
        data_root: str = None,
    ) -> int:
        """Build the search index by extracting and storing embeddings.

        Args:
            db: Database session.
            split: Dataset split to index.
            batch_size: Batch size for embedding extraction.
            data_root: Dataset cache directory.

        Returns:
            Total number of indexed images.
        """
        import medmnist

        data_root = data_root or settings.data_root
        os.makedirs(data_root, exist_ok=True)

        dataset = medmnist.PneumoniaMNIST(
            split=split, download=True, root=data_root
        )

        logger.info("Building index for %s split: %d images", split, len(dataset))

        all_ids = []
        all_labels = []
        all_embeddings = []

        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_images = []
            batch_ids = []
            batch_labels = []

            for j in range(i, batch_end):
                img, lbl = dataset[j]
                img_arr = np.array(img)
                if img_arr.ndim == 3 and img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(axis=2)
                pil_img = Image.fromarray(img_arr.astype(np.uint8), mode="L")
                batch_images.append(pil_img)
                batch_ids.append(f"{split}_{j}")
                batch_labels.append(int(lbl.flatten()[0]))

            embeddings = self.embedding_service.get_batch_embeddings(batch_images)

            all_ids.extend(batch_ids)
            all_labels.extend(batch_labels)
            all_embeddings.append(embeddings)

            logger.info("Extracted embeddings: %d / %d", batch_end, len(dataset))

        all_embeddings = np.vstack(all_embeddings)

        # Clear existing data and insert
        crud.clear_all_images(db)
        total = crud.bulk_insert_images(
            db, all_ids, all_labels, all_embeddings, split=split
        )

        # Create IVFFLAT index
        n_records = crud.get_image_count(db)
        n_lists = min(settings.ivfflat_lists, max(1, n_records // 10))
        crud.create_ivfflat_index(db, n_lists=n_lists)

        logger.info("Index built: %d images, embedding_dim=%d", total, all_embeddings.shape[1])
        return total
