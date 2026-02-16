"""
Experiment runner for Pneumonia Detection.

Runs all 6 model experiments (3 architectures × 2 variants),
collects results, and generates a comparison CSV.

Usage:
    python -m task1_classification.experiment_runner
    python -m task1_classification.experiment_runner --config task1_classification/config.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from models.utils import count_parameters, get_model
from task1_classification.evaluate import evaluate
from task1_classification.train import set_seed, train

logger = logging.getLogger(__name__)


# All experiment configurations
EXPERIMENTS = [
    {"model": "unet", "pretrained": False},
    {"model": "unet", "pretrained": True},
    {"model": "resnet", "pretrained": False},
    {"model": "resnet", "pretrained": True},
    {"model": "efficientnet", "pretrained": False},
    {"model": "efficientnet", "pretrained": True},
]


def run_all_experiments(config_path: str = "task1_classification/config.yaml") -> None:
    """Run all 6 experiments, evaluate each, and generate comparison table.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    os.makedirs(output_cfg["reports_dir"], exist_ok=True)
    os.makedirs(output_cfg["results_dir"], exist_ok=True)
    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

    all_results: List[Dict] = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        model_name = exp["model"]
        pretrained = exp["pretrained"]
        experiment_name = f"{model_name}_{'pretrained' if pretrained else 'scratch'}"

        logger.info("=" * 70)
        logger.info(
            "EXPERIMENT %d/%d: %s", i, len(EXPERIMENTS), experiment_name
        )
        logger.info("=" * 70)

        # Train
        train_results = train(
            model_name=model_name,
            pretrained=pretrained,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            patience=train_cfg["early_stopping_patience"],
            scheduler_patience=train_cfg["scheduler_patience"],
            scheduler_factor=train_cfg["scheduler_factor"],
            seed=train_cfg["seed"],
            data_root=data_cfg["data_root"],
            checkpoint_dir=output_cfg["checkpoint_dir"],
            results_dir=output_cfg["results_dir"],
            num_workers=data_cfg["num_workers"],
        )

        # Evaluate
        test_metrics = evaluate(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_dir=output_cfg["checkpoint_dir"],
            results_dir=output_cfg["results_dir"],
            reports_dir=output_cfg["reports_dir"],
            data_root=data_cfg["data_root"],
            batch_size=train_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
        )

        # Collect results
        result_row = {
            "Model": model_name,
            "Variant": "Pretrained" if pretrained else "Scratch",
            "Experiment": experiment_name,
            "Parameters": train_results["n_parameters"],
            "Training Time (s)": train_results["training_time_seconds"],
            "Epochs Trained": train_results["epochs_trained"],
            "Best Val AUC": round(train_results["best_val_auc"], 4),
            "Test Accuracy": round(test_metrics["accuracy"], 4),
            "Test Precision": round(test_metrics["precision"], 4),
            "Test Recall": round(test_metrics["recall"], 4),
            "Test F1": round(test_metrics["f1"], 4),
            "Test ROC-AUC": round(test_metrics["roc_auc"], 4),
        }
        all_results.append(result_row)

        logger.info("Completed: %s — Test AUC: %.4f", experiment_name, test_metrics["roc_auc"])

    # Generate comparison table
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_cfg["reports_dir"], "experiment_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Experiment comparison saved → %s", csv_path)

    # Print final comparison
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)
    logger.info("\n%s", df.to_string(index=False))

    # Save as JSON too
    json_path = os.path.join(output_cfg["results_dir"], "all_experiments_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary JSON saved → %s", json_path)

    # Find best model
    best = max(all_results, key=lambda x: x["Test ROC-AUC"])
    logger.info(
        "\nBEST MODEL: %s — Test AUC: %.4f | Test F1: %.4f",
        best["Experiment"], best["Test ROC-AUC"], best["Test F1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Pneumonia Detection experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="task1_classification/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_all_experiments(args.config)


if __name__ == "__main__":
    main()
