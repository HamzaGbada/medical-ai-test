"""
Build the retrieval index.

Extracts embeddings from PneumoniaMNIST images and stores them in PGVector.
Can be run standalone or called from run_task3.py.

Usage:
    python -m task3_retrieval.scripts.build_index [--split test] [--batch_size 64]
"""

import argparse
import logging
import os
import sys

from task3_retrieval.app.config import settings
from task3_retrieval.app.database import SessionLocal, init_db
from task3_retrieval.app.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def build(split: str = "test", batch_size: int = 64) -> int:
    """Build the search index.

    Args:
        split: Dataset split to index.
        batch_size: Batch size for embedding extraction.

    Returns:
        Total number of indexed images.
    """
    init_db()

    svc = RetrievalService()
    db = SessionLocal()

    try:
        total = svc.build_index(db=db, split=split, batch_size=batch_size)
        logger.info("Index built successfully: %d images", total)
        return total
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    build(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
