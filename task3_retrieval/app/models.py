"""
SQLAlchemy ORM models for the retrieval system.

Defines the MedicalImage table with a PGVector embedding column.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String

from task3_retrieval.app.config import settings
from task3_retrieval.app.database import Base


class MedicalImage(Base):
    """ORM model for medical images with vector embeddings.

    Attributes:
        id: Auto-incrementing primary key.
        image_id: Unique string identifier (e.g., 'test_42').
        label: Ground truth label (0=Normal, 1=Pneumonia).
        split: Dataset split ('train', 'val', 'test').
        embedding: Dense vector of dimension `embedding_dim`.
    """

    __tablename__ = "medical_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String, unique=True, index=True, nullable=False)
    label = Column(Integer, index=True, nullable=False)
    split = Column(String, index=True, default="test")
    embedding = Column(Vector(settings.embedding_dim), nullable=False)

    def __repr__(self) -> str:
        return f"<MedicalImage(id={self.id}, image_id='{self.image_id}', label={self.label})>"
