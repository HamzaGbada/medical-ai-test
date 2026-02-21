"""
Evaluation script for the retrieval system.

Computes Precision@k for k = {1, 3, 5, 10} across the entire test set.
For each image, retrieves top-k neighbors and checks label agreement.

Usage:
    python -m task3_retrieval.scripts.evaluate [--top_k 10]
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sqlalchemy.orm import Session

from task3_retrieval.app import crud
from task3_retrieval.app.config import settings
from task3_retrieval.app.database import SessionLocal, init_db
from task3_retrieval.app.embedding_service import EmbeddingService
from task3_retrieval.app.models import MedicalImage
from task3_retrieval.app.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def compute_precision_at_k(
    db: Session,
    retrieval_service: RetrievalService,
    k_values: List[int] = None,
    max_queries: int = None,
) -> Dict:
    """Compute Precision@k for all indexed images.

    For each image, retrieves top-max(k_values) neighbors (excluding itself),
    then computes precision at each k.

    Args:
        db: Database session.
        retrieval_service: Retrieval service instance.
        k_values: List of k values to evaluate.
        max_queries: Limit number of query images (for speed).

    Returns:
        Dict with overall and per-class precision metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values)

    # Get all images
    all_images = db.query(MedicalImage).all()
    if max_queries and max_queries < len(all_images):
        np.random.seed(42)
        indices = np.random.choice(len(all_images), max_queries, replace=False)
        all_images = [all_images[i] for i in indices]

    logger.info("Evaluating Precision@k for %d query images", len(all_images))

    # Track per-k, per-class precision
    precision_sums = {k: 0.0 for k in k_values}
    class_precision_sums = {k: defaultdict(float) for k in k_values}
    class_counts = defaultdict(int)

    for i, query_image in enumerate(all_images):
        query_emb = np.array(query_image.embedding)
        query_label = query_image.label
        class_counts[query_label] += 1

        # Retrieve top-k (excluding self)
        results = crud.search_similar(
            db, query_emb, top_k=max_k + 1, exclude_id=query_image.image_id
        )

        # Take only max_k results
        results = results[:max_k]
        retrieved_labels = [img.label for img, _ in results]

        for k in k_values:
            top_k_labels = retrieved_labels[:k]
            relevant = sum(1 for lbl in top_k_labels if lbl == query_label)
            precision = relevant / k
            precision_sums[k] += precision
            class_precision_sums[k][query_label] += precision

        if (i + 1) % 100 == 0:
            logger.info("Evaluated %d / %d queries", i + 1, len(all_images))

    n_queries = len(all_images)

    # Compute averages
    results = {
        "n_queries": n_queries,
        "overall": {},
        "per_class": {},
    }

    for k in k_values:
        avg_prec = precision_sums[k] / max(n_queries, 1)
        results["overall"][f"precision@{k}"] = round(avg_prec, 4)

        for cls_label in sorted(class_counts.keys()):
            cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
            if cls_name not in results["per_class"]:
                results["per_class"][cls_name] = {}
            cls_avg = class_precision_sums[k][cls_label] / max(class_counts[cls_label], 1)
            results["per_class"][cls_name][f"precision@{k}"] = round(cls_avg, 4)

    return results


def run_evaluation(
    output_dir: str = "./results",
    max_queries: int = None,
) -> Dict:
    """Run the full evaluation pipeline.

    Args:
        output_dir: Directory to save results.
        max_queries: Limit queries for faster evaluation.

    Returns:
        Evaluation results dict.
    """
    init_db()
    db = SessionLocal()
    svc = RetrievalService()

    try:
        results = compute_precision_at_k(
            db=db,
            retrieval_service=svc,
            max_queries=max_queries,
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "retrieval_evaluation.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation results saved → %s", output_path)

        # Print summary
        logger.info("=" * 50)
        logger.info("Precision@k Results (%d queries)", results["n_queries"])
        logger.info("=" * 50)
        for k, prec in results["overall"].items():
            logger.info("  %s: %.4f", k, prec)
        for cls_name, metrics in results["per_class"].items():
            logger.info("  %s:", cls_name)
            for k, prec in metrics.items():
                logger.info("    %s: %.4f", k, prec)

        return results
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument("--max_queries", type=int, default=None, help="Limit queries")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(output_dir=args.output_dir, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
