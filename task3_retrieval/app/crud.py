"""
CRUD operations for the medical image database.

Handles inserting embeddings, searching by vector similarity,
and index management.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from task3_retrieval.app.config import settings
from task3_retrieval.app.models import MedicalImage

logger = logging.getLogger(__name__)


def insert_image(
    db: Session,
    image_id: str,
    label: int,
    embedding: np.ndarray,
    split: str = "test",
) -> MedicalImage:
    """Insert a single image with its embedding into the database.

    Args:
        db: SQLAlchemy session.
        image_id: Unique identifier for the image.
        label: Ground truth label (0 or 1).
        embedding: Embedding vector.
        split: Dataset split.

    Returns:
        The created MedicalImage instance.
    """
    record = MedicalImage(
        image_id=image_id,
        label=label,
        split=split,
        embedding=embedding.tolist(),
    )
    db.add(record)
    return record


def bulk_insert_images(
    db: Session,
    image_ids: List[str],
    labels: List[int],
    embeddings: np.ndarray,
    split: str = "test",
    batch_size: int = 100,
) -> int:
    """Bulk insert images into the database.

    Args:
        db: SQLAlchemy session.
        image_ids: List of image identifiers.
        labels: List of labels.
        embeddings: Embedding matrix (N, D).
        split: Dataset split.
        batch_size: Commit batch size.

    Returns:
        Total number of inserted records.
    """
    total = len(image_ids)
    inserted = 0

    for i in range(0, total, batch_size):
        batch_ids = image_ids[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        records = []
        for img_id, lbl, emb in zip(batch_ids, batch_labels, batch_embeddings):
            records.append(MedicalImage(
                image_id=img_id,
                label=int(lbl),
                split=split,
                embedding=emb.tolist(),
            ))

        db.bulk_save_objects(records)
        db.commit()
        inserted += len(records)
        logger.info("Inserted batch %d-%d / %d", i, i + len(records), total)

    return inserted


def search_similar(
    db: Session,
    query_embedding: np.ndarray,
    top_k: int = 5,
    exclude_id: Optional[str] = None,
) -> List[Tuple[MedicalImage, float]]:
    """Search for similar images using cosine distance.

    Uses PGVector's `<=>` operator for cosine distance.

    Args:
        db: SQLAlchemy session.
        query_embedding: Query embedding vector.
        top_k: Number of results to return.
        exclude_id: Optional image_id to exclude (e.g., the query itself).

    Returns:
        List of (MedicalImage, similarity_score) tuples, ordered by similarity.
    """
    query_vec = query_embedding.tolist()

    # Build query with cosine distance
    query = db.query(
        MedicalImage,
        (1 - MedicalImage.embedding.cosine_distance(query_vec)).label("similarity"),
    )

    if exclude_id:
        query = query.filter(MedicalImage.image_id != exclude_id)

    query = query.order_by(
        MedicalImage.embedding.cosine_distance(query_vec)
    ).limit(top_k)

    results = query.all()
    return [(row[0], float(row[1])) for row in results]


def get_image_count(db: Session) -> int:
    """Get total number of indexed images."""
    return db.query(func.count(MedicalImage.id)).scalar() or 0


def get_image_by_id(db: Session, image_id: str) -> Optional[MedicalImage]:
    """Get a single image record by its image_id."""
    return db.query(MedicalImage).filter(MedicalImage.image_id == image_id).first()


def create_ivfflat_index(db: Session, n_lists: int = None) -> None:
    """Create an IVFFLAT index for fast similarity search.

    Args:
        db: SQLAlchemy session.
        n_lists: Number of IVF lists (default from settings).
    """
    n_lists = n_lists or settings.ivfflat_lists
    sql = text(f"""
        CREATE INDEX IF NOT EXISTS medical_images_embedding_idx
        ON medical_images
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {n_lists});
    """)
    db.execute(sql)
    db.commit()
    logger.info("IVFFLAT index created with %d lists", n_lists)


def clear_all_images(db: Session) -> int:
    """Delete all image records from the database.

    Returns:
        Number of deleted records.
    """
    count = db.query(MedicalImage).delete()
    db.commit()
    logger.info("Cleared %d image records", count)
    return count
