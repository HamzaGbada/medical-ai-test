"""
Report generator for Task 2.

Runs the VLM pipeline across all selected samples and all prompt strategies.
Saves individual reports as JSON and aggregates results.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

from task2_report_generation.image_preprocessor import ImagePreprocessor
from task2_report_generation.prompts import PromptStrategy, get_all_strategies
from task2_report_generation.sample_selection import SelectedSample
from task2_report_generation.vlm_pipeline import VLMPipeline, VLMReport

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


class ReportGenerator:
    """Generates medical reports for all selected samples using VLMs.

    Runs each sample through all prompt strategies, saves individual
    reports, and collects structured results.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2-vl",
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        output_dir: str = "./reports/generated_reports",
    ) -> None:
        """Initialize the report generator.

        Args:
            provider: LLM provider ('ollama' or 'docker').
            model: Model name.
            base_url: Optional custom base URL.
            temperature: Sampling temperature.
            max_tokens: Maximum generation tokens.
            output_dir: Directory to save generated reports.
        """
        self.pipeline = VLMPipeline(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.preprocessor = ImagePreprocessor(
            output_dir=os.path.join(output_dir, "images"),
        )
        self.output_dir = output_dir
        self.model_name = model
        # Sanitized name safe for use in filenames (removes / and : characters)
        self.model_name_safe = model.replace("/", "_").replace(":", "-").replace("\\", "_")
        os.makedirs(output_dir, exist_ok=True)

        logger.info("ReportGenerator initialized: model=%s, output=%s", model, output_dir)

    async def generate_for_sample(
        self,
        sample: SelectedSample,
        strategies: Optional[List[PromptStrategy]] = None,
    ) -> List[VLMReport]:
        """Generate reports for a single sample across all strategies.

        Args:
            sample: The selected image sample.
            strategies: List of prompt strategies to use (default: all).

        Returns:
            List of VLMReport for each strategy.
        """
        if strategies is None:
            strategies = get_all_strategies()

        # Preprocess image
        image_id = f"sample_{sample.index}"
        image_base64, image_path = self.preprocessor.process(
            sample.image, image_id=image_id, save=True
        )

        reports = []
        for strategy in strategies:
            report = await self.pipeline.generate_report(
                image_base64=image_base64,
                strategy=strategy,
                image_index=sample.index,
                ground_truth=sample.ground_truth,
                cnn_prediction=sample.cnn_prediction,
                cnn_confidence=sample.cnn_confidence,
            )
            reports.append(report)

            # Save individual report (use sanitized model name for safe filenames)
            report_filename = f"{image_id}_{self.model_name_safe}_{strategy.value}.json"
            report_path = os.path.join(self.output_dir, report_filename)
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info("Report saved: %s", report_path)

        return reports

    async def generate_all(
        self,
        samples: List[SelectedSample],
        strategies: Optional[List[PromptStrategy]] = None,
    ) -> List[VLMReport]:
        """Generate reports for all samples.

        Processes samples sequentially to avoid overwhelming the LLM server.

        Args:
            samples: List of selected samples.
            strategies: Prompt strategies to use (default: all).

        Returns:
            Flat list of all generated VLMReports.
        """
        all_reports = []

        for i, sample in enumerate(samples):
            logger.info(
                "Processing sample %d/%d (index=%d, GT=%s, CNN=%s)",
                i + 1, len(samples), sample.index,
                CLASS_NAMES[sample.ground_truth],
                CLASS_NAMES[sample.cnn_prediction],
            )
            sample_reports = await self.generate_for_sample(sample, strategies)
            all_reports.extend(sample_reports)

        # Save aggregated results (sanitized model name for safe filename)
        all_results_path = os.path.join(
            self.output_dir, f"all_reports_{self.model_name_safe}.json"
        )
        with open(all_results_path, "w") as f:
            json.dump([r.to_dict() for r in all_reports], f, indent=2)

        logger.info(
            "All reports generated: %d total (%d samples × %d strategies) → %s",
            len(all_reports), len(samples),
            len(strategies or get_all_strategies()),
            all_results_path,
        )

        return all_reports
