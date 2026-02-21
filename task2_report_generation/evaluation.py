"""
Qualitative evaluation for VLM-generated medical reports.

Compares ground truth labels, CNN predictions, and VLM outputs
to assess clinical correctness, hallucinations, and model agreement.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

from task2_report_generation.prompts import PromptStrategy
from task2_report_generation.vlm_pipeline import VLMReport

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def extract_impression(report_text: str) -> str:
    """Extract the Impression section from a generated report.

    Looks for common section headers and extracts the content after.

    Args:
        report_text: Raw VLM response text.

    Returns:
        Extracted impression or summary.
    """
    text = report_text.strip()

    # Try to find Impression section
    for marker in ["**Impression:**", "**Impression**:", "Impression:", "## Impression"]:
        if marker.lower() in text.lower():
            idx = text.lower().index(marker.lower())
            after = text[idx + len(marker):].strip()
            # Take until next section header or end
            for end_marker in ["**Confidence", "**Agreement", "**", "\n##", "\n\n\n"]:
                if end_marker.lower() in after.lower():
                    end_idx = after.lower().index(end_marker.lower())
                    return after[:end_idx].strip()
            return after[:500].strip()

    # Fallback: return first 200 chars
    return text[:200].strip()


def classify_vlm_prediction(report_text: str) -> str:
    """Infer the VLM's diagnostic prediction from the report text.

    Simple keyword-based classification.

    Args:
        report_text: Raw VLM response text.

    Returns:
        "Pneumonia", "Normal", or "Unclear".
    """
    text = report_text.lower()

    pneumonia_keywords = [
        "pneumonia", "infiltrat", "consolidat", "opacit", "opacity",
        "infection", "abnormal", "patholog",
    ]
    normal_keywords = [
        "normal", "no significant", "unremarkable", "clear lung",
        "no abnormal", "no evidence of", "within normal",
    ]

    pneumonia_score = sum(1 for kw in pneumonia_keywords if kw in text)
    normal_score = sum(1 for kw in normal_keywords if kw in text)

    if pneumonia_score > normal_score:
        return "Pneumonia"
    elif normal_score > pneumonia_score:
        return "Normal"
    else:
        return "Unclear"


def evaluate_reports(
    reports: List[VLMReport],
    output_dir: str = "./results",
) -> Dict:
    """Evaluate all generated reports.

    Creates a comparison table and analysis summary.

    Args:
        reports: List of VLMReport objects.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary with evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group reports by image index
    by_image: Dict[int, List[VLMReport]] = defaultdict(list)
    for report in reports:
        by_image[report.image_index].append(report)

    # Build evaluation table
    evaluation_rows = []

    for image_idx in sorted(by_image.keys()):
        image_reports = by_image[image_idx]

        # Take the structured prompt report for main comparison
        best_report = None
        for r in image_reports:
            if r.strategy == PromptStrategy.STRUCTURED:
                best_report = r
                break
        if best_report is None:
            best_report = image_reports[0]

        gt_name = CLASS_NAMES[best_report.ground_truth]
        cnn_name = CLASS_NAMES[best_report.cnn_prediction]
        vlm_prediction = classify_vlm_prediction(best_report.raw_response)
        impression = extract_impression(best_report.raw_response)

        # Check agreement
        cnn_correct = best_report.ground_truth == best_report.cnn_prediction
        vlm_agrees_gt = vlm_prediction == gt_name
        vlm_agrees_cnn = vlm_prediction == cnn_name

        row = {
            "image_index": image_idx,
            "ground_truth": gt_name,
            "cnn_prediction": cnn_name,
            "cnn_correct": cnn_correct,
            "cnn_confidence": round(best_report.cnn_confidence, 4),
            "vlm_model": best_report.model_name,
            "vlm_prediction": vlm_prediction,
            "vlm_agrees_gt": vlm_agrees_gt,
            "vlm_agrees_cnn": vlm_agrees_cnn,
            "impression_summary": impression[:200],
        }
        evaluation_rows.append(row)

    # Compute summary statistics
    total = len(evaluation_rows)
    vlm_gt_agreement = sum(1 for r in evaluation_rows if r["vlm_agrees_gt"]) / max(total, 1)
    vlm_cnn_agreement = sum(1 for r in evaluation_rows if r["vlm_agrees_cnn"]) / max(total, 1)
    cnn_accuracy = sum(1 for r in evaluation_rows if r["cnn_correct"]) / max(total, 1)

    summary = {
        "total_samples": total,
        "cnn_accuracy_on_selected": round(cnn_accuracy, 4),
        "vlm_gt_agreement_rate": round(vlm_gt_agreement, 4),
        "vlm_cnn_agreement_rate": round(vlm_cnn_agreement, 4),
        "evaluation_rows": evaluation_rows,
    }

    # Save as JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation JSON saved → %s", json_path)

    # Save as Markdown table
    md_path = os.path.join(output_dir, "evaluation_summary.md")
    _write_markdown_summary(summary, md_path)
    logger.info("Evaluation Markdown saved → %s", md_path)

    return summary


def _write_markdown_summary(summary: Dict, output_path: str) -> None:
    """Write evaluation summary as a Markdown file."""
    lines = [
        "# VLM Evaluation Summary\n",
        f"**Total Samples**: {summary['total_samples']}  ",
        f"**CNN Accuracy (on selected)**: {summary['cnn_accuracy_on_selected']:.2%}  ",
        f"**VLM-GT Agreement**: {summary['vlm_gt_agreement_rate']:.2%}  ",
        f"**VLM-CNN Agreement**: {summary['vlm_cnn_agreement_rate']:.2%}  ",
        "",
        "## Comparison Table\n",
        "| Image ID | Ground Truth | CNN Pred | CNN Correct | VLM Pred | VLM=GT | VLM=CNN | Impression (excerpt) |",
        "| -------- | ------------ | -------- | ----------- | -------- | ------ | ------- | -------------------- |",
    ]

    for row in summary["evaluation_rows"]:
        lines.append(
            f"| {row['image_index']} "
            f"| {row['ground_truth']} "
            f"| {row['cnn_prediction']} "
            f"| {'✅' if row['cnn_correct'] else '❌'} "
            f"| {row['vlm_prediction']} "
            f"| {'✅' if row['vlm_agrees_gt'] else '❌'} "
            f"| {'✅' if row['vlm_agrees_cnn'] else '❌'} "
            f"| {row['impression_summary'][:80]}... |"
        )

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_per_strategy_analysis(
    reports: List[VLMReport],
    output_dir: str = "./results",
) -> Dict:
    """Analyze how different prompting strategies affect output quality.

    Args:
        reports: All generated reports.
        output_dir: Directory to save analysis.

    Returns:
        Dictionary with per-strategy statistics.
    """
    by_strategy: Dict[str, List[VLMReport]] = defaultdict(list)
    for r in reports:
        by_strategy[r.strategy.value].append(r)

    strategy_analysis = {}
    for strategy_name, strat_reports in by_strategy.items():
        predictions = [classify_vlm_prediction(r.raw_response) for r in strat_reports]
        gt_labels = [CLASS_NAMES[r.ground_truth] for r in strat_reports]

        agreement = sum(1 for p, g in zip(predictions, gt_labels) if p == g) / max(len(predictions), 1)
        avg_length = sum(len(r.raw_response) for r in strat_reports) / max(len(strat_reports), 1)

        strategy_analysis[strategy_name] = {
            "n_reports": len(strat_reports),
            "gt_agreement_rate": round(agreement, 4),
            "avg_response_length": round(avg_length),
            "prediction_distribution": {
                "Normal": predictions.count("Normal"),
                "Pneumonia": predictions.count("Pneumonia"),
                "Unclear": predictions.count("Unclear"),
            },
        }

    # Save
    path = os.path.join(output_dir, "strategy_analysis.json")
    with open(path, "w") as f:
        json.dump(strategy_analysis, f, indent=2)
    logger.info("Strategy analysis saved → %s", path)

    return strategy_analysis
