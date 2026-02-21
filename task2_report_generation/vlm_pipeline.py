"""
VLM Pipeline for medical report generation.

Orchestrates image preprocessing → prompt construction → LLM generation
for both text-only and vision-capable models.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from task2_report_generation.llm_service import BaseLLM, LLMFactory
from task2_report_generation.image_preprocessor import ImagePreprocessor
from task2_report_generation.prompts import PromptStrategy, get_prompt

# Optional import — used to catch Ollama-specific ResponseError
try:
    import ollama as _ollama_module
    _OllamaResponseError = _ollama_module.ResponseError
except (ImportError, AttributeError):
    _OllamaResponseError = None

logger = logging.getLogger(__name__)


@dataclass
class VLMReport:
    """A generated report from a VLM."""
    image_index: int
    model_name: str
    provider: str
    strategy: PromptStrategy
    system_prompt: str
    user_prompt: str
    raw_response: str
    ground_truth: int
    cnn_prediction: int
    cnn_confidence: float

    def to_dict(self) -> Dict:
        return {
            "image_index": self.image_index,
            "model_name": self.model_name,
            "provider": self.provider,
            "strategy": self.strategy.value,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "ground_truth": self.ground_truth,
            "cnn_prediction": self.cnn_prediction,
            "cnn_confidence": self.cnn_confidence,
        }


class VLMPipeline:
    """Pipeline for generating medical reports using VLMs.

    Handles image preprocessing, prompt construction, and async LLM calls
    for both Ollama and Docker Model Runner providers.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2-vl",
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        image_target_size: int = 224,
    ) -> None:
        """Initialize the VLM pipeline.

        Args:
            provider: LLM provider ('ollama' or 'docker').
            model: Model name (e.g., 'qwen2-vl', 'medgemma').
            base_url: Optional custom base URL for the provider.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens to generate.
            image_target_size: Target image size for upscaling.
        """
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create LLM
        kwargs = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm: BaseLLM = LLMFactory.create_llm(provider=provider, **kwargs)

        # Image preprocessor
        self.preprocessor = ImagePreprocessor(target_size=image_target_size)

        logger.info(
            "VLMPipeline initialized: provider=%s, model=%s, temperature=%.2f",
            provider, model, temperature,
        )

    async def generate_report(
        self,
        image_base64: str,
        strategy: PromptStrategy,
        image_index: int,
        ground_truth: int = 0,
        cnn_prediction: int = 0,
        cnn_confidence: float = 0.5,
    ) -> VLMReport:
        """Generate a medical report for a single image.

        Args:
            image_base64: Base64-encoded image string.
            strategy: Prompting strategy to use.
            image_index: Image index for tracking.
            ground_truth: Ground truth label.
            cnn_prediction: CNN predicted label.
            cnn_confidence: CNN confidence score.

        Returns:
            VLMReport with the generated response.
        """
        # Get prompt
        prompt_pair = get_prompt(
            strategy=strategy,
            cnn_prediction=cnn_prediction,
            ground_truth=ground_truth,
            cnn_confidence=cnn_confidence,
        )
        logger.info(
            "Generating report: image=%d, model=%s, strategy=%s",
            image_index, self.model_name, strategy.value,
        )

        try:
            # Attempt vision API
            response = await self.llm.generate_with_image(
                prompt=prompt_pair.user_prompt,
                image_base64=image_base64,
                system_prompt=prompt_pair.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except (NotImplementedError, *([_OllamaResponseError] if _OllamaResponseError else [])):
            # Fallback: model doesn't support image input (NotImplementedError) OR
            # Ollama returned 500 because the model has no vision capability
            logger.warning(
                "Model '%s' does not support image input — falling back to text-only generation",
                self.model_name,
            )
            fallback_prompt = (
                f"{prompt_pair.user_prompt}\n\n"
                "[Context: The image is a 28×28 grayscale chest X-ray from the PneumoniaMNIST "
                f"dataset. CNN prediction: {'Pneumonia' if cnn_prediction else 'Normal'} "
                f"(confidence: {cnn_confidence:.2f}). "
                f"Ground truth label: {'Pneumonia' if ground_truth else 'Normal'}.]"
            )
            response = await self.llm.generate(
                prompt=fallback_prompt,
                system_prompt=prompt_pair.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(
                "Unexpected error for image %d: %s — using text-only fallback",
                image_index, e,
            )
            fallback_prompt = (
                f"{prompt_pair.user_prompt}\n\n"
                "[Context: This is a 28×28 grayscale chest X-ray. "
                f"CNN prediction: {'Pneumonia' if cnn_prediction else 'Normal'} "
                f"(confidence: {cnn_confidence:.2f}). "
                f"Ground truth: {'Pneumonia' if ground_truth else 'Normal'}.]"
            )
            try:
                response = await self.llm.generate(
                    prompt=fallback_prompt,
                    system_prompt=prompt_pair.system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e2:
                logger.error("Text fallback also failed for image %d: %s", image_index, e2)
                response = f"[ERROR] Failed to generate report: {str(e2)}"

        return VLMReport(
            image_index=image_index,
            model_name=self.model_name,
            provider=self.provider,
            strategy=strategy,
            system_prompt=prompt_pair.system_prompt,
            user_prompt=prompt_pair.user_prompt,
            raw_response=response,
            ground_truth=ground_truth,
            cnn_prediction=cnn_prediction,
            cnn_confidence=cnn_confidence,
        )
