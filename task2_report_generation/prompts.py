"""
Prompt templates for medical report generation.

Implements 3 prompting strategies:
1. Basic — simple image description
2. Structured Radiologist — professional structured format
3. Diagnostic-Guided — includes CNN prediction context for step-by-step reasoning
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PromptStrategy(str, Enum):
    """Available prompting strategies."""
    BASIC = "basic"
    STRUCTURED = "structured"
    DIAGNOSTIC_GUIDED = "diagnostic_guided"


@dataclass
class PromptPair:
    """A system prompt and user prompt pair."""
    system_prompt: str
    user_prompt: str
    strategy: PromptStrategy


# ---------------------------------------------------------------------------
# Prompt Strategy 1: Basic
# ---------------------------------------------------------------------------

BASIC_SYSTEM = (
    "You are a medical imaging AI assistant."
)

BASIC_USER = (
    "Describe this chest X-ray image. "
    "Identify any abnormalities you observe and provide your assessment."
)


# ---------------------------------------------------------------------------
# Prompt Strategy 2: Structured Radiologist
# ---------------------------------------------------------------------------

STRUCTURED_SYSTEM = (
    "You are an experienced board-certified radiologist analyzing chest X-ray images. "
    "Provide your analysis in a formal, structured radiology report format. "
    "Be precise, use appropriate medical terminology, and note any limitations "
    "of the image quality."
)

STRUCTURED_USER = """Analyze this chest X-ray image and provide a structured radiology report with the following sections:

**Examination:** Describe the type of examination and image quality.

**Findings:** Describe all observable findings systematically:
- Lung fields (clarity, opacities, infiltrates, consolidation)
- Cardiac silhouette (size, shape)
- Mediastinum and hilum
- Costophrenic angles
- Bony structures (if visible)

**Impression:** Summarize the key findings and provide your diagnostic impression.

**Confidence Level:** Rate your confidence (Low/Medium/High) and explain any factors affecting it."""


# ---------------------------------------------------------------------------
# Prompt Strategy 3: Diagnostic-Guided
# ---------------------------------------------------------------------------

DIAGNOSTIC_SYSTEM = (
    "You are an expert radiologist performing a detailed analysis of a chest X-ray. "
    "You have access to the results of an automated CNN classifier. "
    "Use this information as additional context, but form your own independent assessment. "
    "Think step-by-step and explain your reasoning."
)

DIAGNOSTIC_USER_TEMPLATE = """Analyze this chest X-ray image.

**Additional Context (from automated CNN classifier):**
- CNN Prediction: {cnn_prediction}
- Ground Truth Label: {ground_truth}
- CNN Confidence: {cnn_confidence:.2f}

Please provide a structured analysis:

**Examination:** Describe the type and quality of the image.

**Findings:** Systematically describe all observable findings. Reason step-by-step:
1. First, describe what you see in the lung fields
2. Note any opacities, consolidations, or infiltrates
3. Assess the cardiac silhouette
4. Check costophrenic angles
5. Note any other relevant findings

**Impression:** Your diagnostic impression, considering both your visual analysis and the CNN result.

**Confidence Level:** Rate your confidence (Low/Medium/High).

**Agreement with CNN:** Do you agree with the CNN prediction? Why or why not?"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def get_prompt(
    strategy: PromptStrategy,
    cnn_prediction: Optional[int] = None,
    ground_truth: Optional[int] = None,
    cnn_confidence: Optional[float] = None,
) -> PromptPair:
    """Get a prompt pair for the specified strategy.

    Args:
        strategy: Which prompting strategy to use.
        cnn_prediction: CNN predicted class (0=Normal, 1=Pneumonia).
        ground_truth: Ground truth class (0=Normal, 1=Pneumonia).
        cnn_confidence: CNN confidence score (probability).

    Returns:
        PromptPair with system and user prompts.
    """
    if strategy == PromptStrategy.BASIC:
        return PromptPair(
            system_prompt=BASIC_SYSTEM,
            user_prompt=BASIC_USER,
            strategy=strategy,
        )

    elif strategy == PromptStrategy.STRUCTURED:
        return PromptPair(
            system_prompt=STRUCTURED_SYSTEM,
            user_prompt=STRUCTURED_USER,
            strategy=strategy,
        )

    elif strategy == PromptStrategy.DIAGNOSTIC_GUIDED:
        cnn_pred_name = CLASS_NAMES.get(cnn_prediction, "Unknown")
        gt_name = CLASS_NAMES.get(ground_truth, "Unknown")
        conf = cnn_confidence if cnn_confidence is not None else 0.0

        user_prompt = DIAGNOSTIC_USER_TEMPLATE.format(
            cnn_prediction=cnn_pred_name,
            ground_truth=gt_name,
            cnn_confidence=conf,
        )
        return PromptPair(
            system_prompt=DIAGNOSTIC_SYSTEM,
            user_prompt=user_prompt,
            strategy=strategy,
        )

    else:
        raise ValueError(f"Unknown prompt strategy: {strategy}")


def get_all_strategies() -> list:
    """Return all available prompt strategies."""
    return list(PromptStrategy)
