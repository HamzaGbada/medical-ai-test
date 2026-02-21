"""
Pydantic schemas for FastAPI request/response validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# --- Requests ---

class ImageSearchRequest(BaseModel):
    """Request body for image-to-image search."""
    image_path: str = Field(..., description="Path to the query image file")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")


class TextSearchRequest(BaseModel):
    """Request body for text-to-image search."""
    query_text: str = Field(..., description="Text query describing the image to find")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")


class BuildIndexRequest(BaseModel):
    """Request body for building the search index."""
    split: str = Field(default="test", description="Dataset split to index ('train', 'val', 'test')")
    batch_size: int = Field(default=64, ge=1, le=512, description="Batch size for embedding extraction")


# --- Responses ---

class SearchResult(BaseModel):
    """A single search result."""
    image_id: str
    label: int
    label_name: str
    score: float = Field(..., description="Cosine similarity score")


class ImageSearchResponse(BaseModel):
    """Response for image-to-image search."""
    query_label: Optional[int] = None
    query_label_name: Optional[str] = None
    results: List[SearchResult]


class TextSearchResponse(BaseModel):
    """Response for text-to-image search."""
    query_text: str
    results: List[SearchResult]


class BuildIndexResponse(BaseModel):
    """Response after building the index."""
    status: str
    total_images: int
    split: str
    embedding_dim: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    embedding_model: str
    embedding_dim: int
    total_indexed: int
