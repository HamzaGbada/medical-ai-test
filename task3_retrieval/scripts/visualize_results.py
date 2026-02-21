"""
Visualization script for retrieval results.

Generates query-vs-results grids showing the query image alongside
its top-k nearest neighbors with labels and similarity scores.

Usage:
    python -m task3_retrieval.scripts.visualize_results [--n_queries 5]
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import medmnist
import numpy as np
from PIL import Image

from task3_retrieval.app import crud
from task3_retrieval.app.config import settings
from task3_retrieval.app.database import SessionLocal, init_db
from task3_retrieval.app.models import MedicalImage
from task3_retrieval.app.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def visualize_retrieval(
    n_queries: int = 5,
    top_k: int = 5,
    output_dir: str = "./reports/retrieval_visualizations",
    data_root: str = None,
    seed: int = 42,
) -> None:
    """Generate retrieval visualization grids.

    For each query image, shows the query and its top-k results in a row.

    Args:
        n_queries: Number of query images to visualize.
        top_k: Number of results per query.
        output_dir: Directory to save visualizations.
        data_root: Dataset cache directory.
        seed: Random seed for query selection.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_root = data_root or settings.data_root

    init_db()
    db = SessionLocal()
    svc = RetrievalService()

    try:
        # Load test dataset for image display
        os.makedirs(data_root, exist_ok=True)
        dataset = medmnist.PneumoniaMNIST(split="test", download=True, root=data_root)

        # Get diverse query images (some Normal, some Pneumonia)
        all_images = db.query(MedicalImage).all()
        if not all_images:
            logger.error("No images in database — run build_index first")
            return

        np.random.seed(seed)

        # Get indices for each class
        normal_ids = [img for img in all_images if img.label == 0]
        pneumonia_ids = [img for img in all_images if img.label == 1]

        n_normal = min(n_queries // 2 + 1, len(normal_ids))
        n_pneumonia = min(n_queries - n_normal, len(pneumonia_ids))

        selected = (
            list(np.random.choice(normal_ids, n_normal, replace=False))
            + list(np.random.choice(pneumonia_ids, n_pneumonia, replace=False))
        )[:n_queries]

        # Create grid visualization
        fig, axes = plt.subplots(
            n_queries, top_k + 1,
            figsize=(3 * (top_k + 1), 3 * n_queries),
        )
        if n_queries == 1:
            axes = [axes]

        for row_idx, query_img_record in enumerate(selected):
            query_emb = np.array(query_img_record.embedding)
            query_label = query_img_record.label

            # Get the actual image from dataset
            idx = int(query_img_record.image_id.split("_")[1])
            img_data, _ = dataset[idx]
            query_pil = np.array(img_data)
            if query_pil.ndim == 3:
                query_pil = query_pil.squeeze()

            # Query
            ax = axes[row_idx][0]
            ax.imshow(query_pil, cmap="gray")
            ax.set_title(
                f"QUERY\n{CLASS_NAMES[query_label]}",
                fontsize=10,
                fontweight="bold",
                color="blue",
            )
            ax.axis("off")

            # Retrieve top-k
            results = crud.search_similar(
                db, query_emb, top_k=top_k,
                exclude_id=query_img_record.image_id,
            )

            for col_idx, (result_img, score) in enumerate(results):
                result_idx = int(result_img.image_id.split("_")[1])
                result_data, _ = dataset[result_idx]
                result_pil = np.array(result_data)
                if result_pil.ndim == 3:
                    result_pil = result_pil.squeeze()

                ax = axes[row_idx][col_idx + 1]
                ax.imshow(result_pil, cmap="gray")

                match = result_img.label == query_label
                color = "green" if match else "red"
                ax.set_title(
                    f"Top-{col_idx + 1}\n{CLASS_NAMES[result_img.label]}\n{score:.3f}",
                    fontsize=9,
                    color=color,
                )
                ax.axis("off")

        plt.suptitle(
            "Image Retrieval Results (Green=Same Class, Red=Different)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = os.path.join(output_dir, "retrieval_grid.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Visualization saved → %s", output_path)

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize retrieval results")
    parser.add_argument("--n_queries", type=int, default=5, help="Number of queries")
    parser.add_argument("--top_k", type=int, default=5, help="Results per query")
    parser.add_argument(
        "--output_dir", default="./reports/retrieval_visualizations",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    visualize_retrieval(
        n_queries=args.n_queries,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
