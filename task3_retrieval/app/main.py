"""
FastAPI application for the Semantic Image Retrieval System.

Endpoints:
    POST /build-index   — Extract embeddings and build the search index
    POST /search/image  — Image-to-image search
    POST /search/text   — Text-to-image search
    GET  /health        — Health check
"""

import logging

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from task3_retrieval.app import crud
from task3_retrieval.app.config import settings
from task3_retrieval.app.database import get_db, init_db
from task3_retrieval.app.retrieval_service import RetrievalService
from task3_retrieval.app.schemas import (
    BuildIndexRequest,
    BuildIndexResponse,
    HealthResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    TextSearchRequest,
    TextSearchResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Image Retrieval System",
    description=(
        "Semantic image retrieval for PneumoniaMNIST using PGVector.\n\n"
        "Supports two embedding models:\n"
        "- **resnet18** (default): ResNet18 Task 1 fine-tuned checkpoint, 512-d, image search only\n"
        "- **biovil**: microsoft/BiomedVLP-BioViL-T (CVPR 2023), 128-d, image + text search\n\n"
        "Select model via `EMBEDDING_MODEL` environment variable."
    ),
    version="2.0.0",
)

# Lazy-initialized retrieval service
_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    """Get or create the retrieval service singleton."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        logger.info("Make sure PGVector Docker container is running")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/build-index", response_model=BuildIndexResponse)
def build_index(
    request: BuildIndexRequest = BuildIndexRequest(),
    db: Session = Depends(get_db),
):
    """Extract embeddings for all images and build the search index."""
    try:
        svc = get_retrieval_service()
        total = svc.build_index(
            db=db,
            split=request.split,
            batch_size=request.batch_size,
        )
        return BuildIndexResponse(
            status="success",
            total_images=total,
            split=request.split,
            embedding_dim=settings.embedding_dim,
        )
    except Exception as e:
        logger.error("Failed to build index: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")


@app.post("/search/image", response_model=ImageSearchResponse)
def search_by_image(
    request: ImageSearchRequest,
    db: Session = Depends(get_db),
):
    """Search for similar images given an image file path."""
    try:
        svc = get_retrieval_service()
        result = svc.search_by_image_path(
            db=db,
            image_path=request.image_path,
            top_k=request.top_k,
        )
        return ImageSearchResponse(
            query_label=result["query_label"],
            query_label_name=result["query_label_name"],
            results=result["results"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    except Exception as e:
        logger.error("Image search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text", response_model=TextSearchResponse)
def search_by_text(
    request: TextSearchRequest,
    db: Session = Depends(get_db),
):
    """Search for images using a text query (requires EMBEDDING_MODEL=biovil).

    With BioViL-T, text queries are projected into the same joint embedding
    space as images, enabling direct cosine similarity search.
    Example queries: 'bilateral lower lobe consolidation', 'normal chest'.
    """
    try:
        svc = get_retrieval_service()
        results = svc.search_by_text(
            db=db,
            query_text=request.query_text,
            top_k=request.top_k,
        )
        return TextSearchResponse(
            query_text=request.query_text,
            results=results,
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail=(
                "Text search requires EMBEDDING_MODEL=biovil. "
                "Current model 'resnet18' only supports image-to-image search. "
                "Restart the server with: EMBEDDING_MODEL=biovil uvicorn task3_retrieval.app.main:app --reload"
            ),
        )
    except Exception as e:
        logger.error("Text search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint — reports DB status, active model, and index size."""
    try:
        total = crud.get_image_count(db)
        db_status = "connected"
    except Exception:
        total = 0
        db_status = "disconnected"

    return HealthResponse(
        status="ok",
        database=db_status,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        total_indexed=total,
    )
