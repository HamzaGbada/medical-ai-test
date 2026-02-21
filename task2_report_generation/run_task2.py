"""
Task 2 runner — Medical Report Generation with VLMs.

Orchestrates the full pipeline:
1. Load/select samples from PneumoniaMNIST
2. Generate reports using VLMs (MedGemma, Qwen-VL)
3. Evaluate and compare outputs
4. Generate Markdown report

Usage:
    python -m task2_report_generation.run_task2 --provider ollama --model qwen2-vl
    python -m task2_report_generation.run_task2 --provider docker --model medgemma
    python -m task2_report_generation.run_task2 --provider ollama --model qwen2-vl --num_samples 10
"""

import argparse
import asyncio
import json
import logging
import os
from typing import List, Optional

from task2_report_generation.evaluation import (
    evaluate_reports,
    generate_per_strategy_analysis,
)
from task2_report_generation.prompts import PromptStrategy, get_all_strategies
from task2_report_generation.report_generator import ReportGenerator
from task2_report_generation.sample_selection import (
    SelectedSample,
    save_selection_info,
    select_samples,
)
from task2_report_generation.vlm_pipeline import VLMReport

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def generate_markdown_report(
    all_reports: List[VLMReport],
    evaluation_results: dict,
    strategy_analysis: dict,
    samples: List[SelectedSample],
    model_name: str,
    provider: str,
    output_path: str = "./reports/task2_report_generation.md",
) -> None:
    """Generate the final Markdown report for Task 2.

    Args:
        all_reports: All generated VLM reports.
        evaluation_results: Evaluation summary dict.
        strategy_analysis: Per-strategy analysis dict.
        samples: Selected image samples.
        model_name: VLM model name.
        provider: LLM provider name.
        output_path: Path to save the report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []

    # Title
    lines.append("# Task 2: Medical Report Generation using Visual Language Models\n")
    lines.append(f"**Model**: {model_name}  ")
    lines.append(f"**Provider**: {provider}  ")
    lines.append(f"**Total Samples**: {len(samples)}  ")
    lines.append(f"**Prompt Strategies**: {len(get_all_strategies())}  ")
    lines.append("")
    lines.append("---\n")

    # Section 1: Model Selection Justification
    lines.append("## 1. Model Selection Justification\n")
    lines.append("### MedGemma (Google Medical VLM)\n")
    lines.append(
        "MedGemma is a medical-domain-specific vision-language model developed by Google. "
        "Its key advantage is **domain specialization** — it was trained on medical imaging datasets "
        "and clinical text, making it well-suited for generating radiologically accurate reports. "
        "It understands medical terminology, anatomical landmarks, and pathological patterns "
        "specific to chest X-rays.\n"
    )
    lines.append("### Qwen-VL (Multimodal Qwen)\n")
    lines.append(
        "Qwen-VL is a strong **general-purpose multimodal model** with broad visual understanding "
        "capabilities. While not specifically trained for medical imaging, its strong reasoning "
        "abilities and large-scale pretraining allow it to provide detailed image descriptions "
        "and structured analyses. It serves as a comparison baseline to evaluate whether "
        "domain-specific training (MedGemma) provides tangible benefits over general multimodal "
        "reasoning.\n"
    )
    lines.append(
        "**Key Comparison**: Domain specialization (MedGemma) vs. general multimodal capability (Qwen-VL). "
        "Medical VLMs should exhibit fewer hallucinations and more accurate anatomical descriptions, "
        "while general VLMs may provide more creative but potentially less clinically accurate analyses.\n"
    )
    lines.append("---\n")

    # Section 2: Prompting Strategies
    lines.append("## 2. Prompting Strategies Tested\n")

    lines.append("### Strategy 1: Basic Prompt\n")
    lines.append('**Prompt**: "Describe this chest X-ray image."  ')
    lines.append("**Temperature**: 0.3  ")
    lines.append("**Characteristics**: Free-form, no structure enforcement. Tests the model's "
                 "default medical knowledge without guidance.\n")

    lines.append("### Strategy 2: Structured Radiologist Prompt\n")
    lines.append("**Prompt**: Detailed structured format requesting Examination, Findings, "
                 "Impression, and Confidence sections.  ")
    lines.append("**Temperature**: 0.3  ")
    lines.append("**Characteristics**: Enforces professional radiology report structure. "
                 "Should produce more clinically useful output.\n")

    lines.append("### Strategy 3: Diagnostic-Guided Prompt\n")
    lines.append("**Prompt**: Includes CNN prediction and ground truth as context. "
                 "Asks for step-by-step reasoning and CNN agreement assessment.  ")
    lines.append("**Temperature**: 0.3  ")
    lines.append("**Characteristics**: Tests whether providing CNN context improves or "
                 "biases VLM analysis. Enables evaluation of model's independent reasoning.\n")

    if strategy_analysis:
        lines.append("### Strategy Comparison\n")
        lines.append("| Strategy | Reports | GT Agreement | Avg Length | Pneumonia | Normal | Unclear |")
        lines.append("| -------- | ------- | ------------ | ---------- | --------- | ------ | ------- |")
        for name, stats in strategy_analysis.items():
            dist = stats["prediction_distribution"]
            lines.append(
                f"| {name} | {stats['n_reports']} | {stats['gt_agreement_rate']:.2%} "
                f"| {stats['avg_response_length']} chars "
                f"| {dist.get('Pneumonia', 0)} | {dist.get('Normal', 0)} | {dist.get('Unclear', 0)} |"
            )
        lines.append("")

    lines.append("---\n")

    # Section 3: Sample Generated Reports
    lines.append("## 3. Sample Generated Reports\n")

    # Group reports by image
    from collections import defaultdict
    by_image = defaultdict(list)
    for r in all_reports:
        by_image[r.image_index].append(r)

    for sample in samples:
        idx = sample.index
        gt_name = CLASS_NAMES[sample.ground_truth]
        cnn_name = CLASS_NAMES[sample.cnn_prediction]
        correctly = "✅" if sample.ground_truth == sample.cnn_prediction else "❌"

        lines.append(f"### Image {idx}\n")
        lines.append(f"- **Ground Truth**: {gt_name}")
        lines.append(f"- **CNN Prediction**: {cnn_name} (confidence: {sample.cnn_confidence:.2f}) {correctly}")
        lines.append("")

        image_reports = by_image.get(idx, [])
        for report in image_reports:
            lines.append(f"#### {report.strategy.value} prompt\n")
            # Truncate very long responses for readability
            response_text = report.raw_response[:1000]
            if len(report.raw_response) > 1000:
                response_text += "\n\n*[Response truncated]*"
            lines.append(f"```\n{response_text}\n```\n")

        lines.append("---\n")

    # Section 4: Qualitative Analysis
    lines.append("## 4. Qualitative Analysis\n")

    if evaluation_results:
        lines.append(f"**VLM-Ground Truth Agreement**: {evaluation_results.get('vlm_gt_agreement_rate', 0):.2%}  ")
        lines.append(f"**VLM-CNN Agreement**: {evaluation_results.get('vlm_cnn_agreement_rate', 0):.2%}  ")
        lines.append(f"**CNN Accuracy (selected samples)**: {evaluation_results.get('cnn_accuracy_on_selected', 0):.2%}  ")
        lines.append("")

    lines.append("### Comparison Table\n")
    lines.append("| Image ID | Ground Truth | CNN Pred | VLM Pred | VLM=GT | VLM=CNN |")
    lines.append("| -------- | ------------ | -------- | -------- | ------ | ------- |")

    if evaluation_results and "evaluation_rows" in evaluation_results:
        for row in evaluation_results["evaluation_rows"]:
            lines.append(
                f"| {row['image_index']} "
                f"| {row['ground_truth']} "
                f"| {row['cnn_prediction']} "
                f"| {row['vlm_prediction']} "
                f"| {'✅' if row['vlm_agrees_gt'] else '❌'} "
                f"| {'✅' if row['vlm_agrees_cnn'] else '❌'} |"
            )
    lines.append("")

    lines.append("### Key Observations\n")
    lines.append(
        "- **Pneumonia detection**: The VLM's ability to detect pneumonia patterns depends heavily on image resolution. "
        "28×28 images are extremely low resolution for radiological analysis.\n"
        "- **Hallucinations**: VLMs may describe anatomical structures (like costophrenic angles, cardiac silhouette) "
        "that are difficult or impossible to assess at 28×28 resolution. These descriptions, while medically sound "
        "in format, may not accurately reflect what's visible in the image.\n"
        "- **Conservative vs. sensitive**: Domain-specific models tend to be more conservative, while general "
        "models may over-diagnose or provide less specific findings.\n"
        "- **CNN failure cases**: On misclassified images, VLMs may provide additional diagnostic perspective "
        "that the CNN missed, or may agree with the CNN's (incorrect) assessment.\n"
    )

    lines.append("---\n")

    # Section 5: Strengths & Limitations
    lines.append("## 5. Strengths & Limitations\n")

    lines.append("### Resolution Limitations\n")
    lines.append(
        "PneumoniaMNIST uses 28×28 pixel images — far below clinical resolution (typically 2000×2000+). "
        "While upscaling to 224×224 helps VLMs process the images, the underlying information is limited. "
        "This affects all models equally and represents the most significant limitation of this evaluation.\n"
    )

    lines.append("### Domain Specificity\n")
    lines.append(
        "Medical VLMs (like MedGemma) are expected to show:\n"
        "- More appropriate medical terminology\n"
        "- Better understanding of what's clinically relevant\n"
        "- Fewer false positive findings\n"
        "- More calibrated confidence levels\n"
    )

    lines.append("### Sensitivity vs. Specificity Bias\n")
    lines.append(
        "In clinical settings, high sensitivity (detecting all pneumonia cases) is typically preferred "
        "over high specificity. VLMs may exhibit different biases depending on their training data distribution.\n"
    )

    lines.append("### Explainability\n")
    lines.append(
        "A key advantage of VLMs over CNN classifiers is their ability to provide natural language explanations "
        "for their assessments. This interpretability is crucial for clinical trust, even if the underlying "
        "diagnostic accuracy may be lower than a specialized CNN classifier.\n"
    )

    lines.append("---\n")
    lines.append(
        "*Report auto-generated by the Task 2 pipeline.*  \n"
        "*Dataset: PneumoniaMNIST (MedMNIST v2) — 28×28 grayscale chest X-rays.*"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Task 2 Markdown report saved → %s", output_path)


async def run_task2(
    provider: str = "ollama",
    model: str = "qwen2-vl",
    base_url: Optional[str] = None,
    num_samples: int = 10,
    cnn_model: str = "resnet",
    cnn_pretrained: bool = True,
    temperature: float = 0.3,
    seed: int = 42,
    data_root: str = "./data_cache",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./reports/generated_reports",
    reports_dir: str = "./reports",
    results_dir: str = "./results",
) -> None:
    """Run the complete Task 2 pipeline.

    Steps:
    1. Select samples from the test set
    2. Generate reports with VLM
    3. Evaluate results
    4. Generate Markdown report

    Args:
        provider: LLM provider ('ollama' or 'docker').
        model: VLM model name.
        base_url: Optional custom base URL.
        num_samples: Number of samples to process.
        cnn_model: CNN model name for loading predictions.
        cnn_pretrained: Whether the CNN was pretrained.
        temperature: Sampling temperature.
        seed: Random seed.
        data_root: Dataset cache directory.
        checkpoint_dir: CNN checkpoint directory.
        output_dir: Directory for generated reports.
        reports_dir: Directory for final report.
        results_dir: Directory for evaluation results.
    """
    logger.info("=" * 70)
    logger.info("TASK 2: Medical Report Generation with VLMs")
    logger.info("Provider: %s | Model: %s", provider, model)
    logger.info("=" * 70)

    # Step 1: Select samples
    logger.info("Step 1: Selecting samples...")
    n_normal = max(3, num_samples // 3)
    n_pneumonia = max(3, num_samples // 3)
    n_misc = max(4, num_samples - n_normal - n_pneumonia)

    samples = select_samples(
        n_normal=n_normal,
        n_pneumonia=n_pneumonia,
        n_misclassified=n_misc,
        model_name=cnn_model,
        pretrained=cnn_pretrained,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        seed=seed,
    )
    save_selection_info(samples, os.path.join(results_dir, "selected_samples.json"))
    logger.info("Selected %d samples", len(samples))

    # Step 2: Generate reports
    logger.info("Step 2: Generating reports with %s (%s)...", model, provider)
    generator = ReportGenerator(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        output_dir=output_dir,
    )
    all_reports = await generator.generate_all(samples)
    logger.info("Generated %d reports", len(all_reports))

    # Step 3: Evaluate
    logger.info("Step 3: Evaluating reports...")
    evaluation_results = evaluate_reports(all_reports, output_dir=results_dir)
    strategy_analysis = generate_per_strategy_analysis(all_reports, output_dir=results_dir)

    # Step 4: Generate Markdown report
    logger.info("Step 4: Generating Markdown report...")
    generate_markdown_report(
        all_reports=all_reports,
        evaluation_results=evaluation_results,
        strategy_analysis=strategy_analysis,
        samples=samples,
        model_name=model,
        provider=provider,
        output_path=os.path.join(reports_dir, "task2_report_generation.md"),
    )

    logger.info("=" * 70)
    logger.info("Task 2 complete. Results:")
    logger.info("  Reports: %s", output_dir)
    logger.info("  Evaluation: %s", results_dir)
    logger.info("  Markdown Report: %s", os.path.join(reports_dir, "task2_report_generation.md"))
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 2: Medical Report Generation with VLMs"
    )
    parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "docker"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model", type=str, default="qwen2-vl",
        help="VLM model name (e.g., qwen2-vl, medgemma, llava)",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Custom base URL for the provider",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--cnn_model", type=str, default="resnet",
        choices=["unet", "resnet", "efficientnet"],
        help="CNN model to load predictions from",
    )
    parser.add_argument(
        "--cnn_pretrained", type=str, default="true",
        choices=["true", "false"],
        help="Whether the CNN used pretrained weights",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(run_task2(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        num_samples=args.num_samples,
        cnn_model=args.cnn_model,
        cnn_pretrained=args.cnn_pretrained.lower() == "true",
        temperature=args.temperature,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
