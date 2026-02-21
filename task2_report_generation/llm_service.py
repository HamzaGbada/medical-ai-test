"""
LLM Service — Factory and provider implementations for LLM interaction.

Supports text-only and vision (VLM) generation via:
- Ollama (local models)
- Docker Model Runner (OpenAI-compatible endpoint)
- OpenAI, Anthropic, Google (placeholders)

Extended with `generate_with_image()` for multimodal VLM support.
"""

import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import httpx
except ImportError:
    httpx = None
    logger.warning("httpx not installed — DockerLLM will not be available")

try:
    import ollama
except ImportError:
    ollama = None
    logger.warning("ollama not installed — OllamaLLM will not be available")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    """Default configuration for LLM providers."""

    # Ollama defaults
    ollama_model: str = "qwen2-vl"
    ollama_base_url: str = "http://localhost:11434"

    # Docker Model Runner defaults (OpenAI-compatible)
    # Port 12434 is the default for Docker Model Runner
    docker_model: str = "ai/gemma3-qat:270M-F16"
    docker_base_url: str = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"

    # General defaults
    default_llm_provider: str = "ollama"
    default_temperature: float = 0.3
    default_max_tokens: int = 2000


# Global settings instance — can be overridden
settings = LLMSettings()


# ---------------------------------------------------------------------------
# Base LLM
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text with retrieved context (for RAG)."""
        pass

    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from a prompt with an image (VLM).

        Args:
            prompt: Text prompt describing the task.
            image_base64: Base64-encoded image string (PNG/JPEG).
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.

        Raises:
            NotImplementedError: If the provider does not support images.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image inputs. "
            "Use a VLM-capable provider (OllamaLLM or DockerLLM)."
        )


# ---------------------------------------------------------------------------
# Docker Model Runner (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class DockerLLM(BaseLLM):
    """LLM implementation using Docker Model Runner (OpenAI-compatible API).

    Supports both text-only and vision (VLM) generation.
    Vision uses the OpenAI-compatible content array format with base64 images.
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: float = 120.0,
    ):
        if httpx is None:
            raise ImportError(
                "httpx is required for DockerLLM. Install with: pip install httpx"
            )
        self.model = model or settings.docker_model
        self.base_url = base_url or settings.docker_base_url
        self.timeout = timeout
        logger.info("DockerLLM initialized: model=%s, base_url=%s", self.model, self.base_url)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = "You are a helpful research assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using Docker Model Runner."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=payload)

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: Optional[str] = "You are a helpful medical assistant.",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from image + prompt using OpenAI-compatible vision format.

        Uses the content array format with image_url containing base64 data.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # OpenAI vision API format: content is an array of parts
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug("DockerLLM vision request to %s (model=%s)", self.base_url, self.model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=payload)

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a response with additional context."""
        context_text = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)]
        )
        full_prompt = f"""Based on the following context, answer the query.

{context_text}

Query: {query}

Answer:"""
        return await self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------

class OllamaLLM(BaseLLM):
    """Ollama LLM implementation for local models.

    Supports both text-only and vision (VLM) generation.
    Vision uses Ollama's native `images` parameter.
    """

    def __init__(self, model: str = None, base_url: str = None):
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = ollama.Client(host=self.base_url)
        logger.info("OllamaLLM initialized: model=%s, base_url=%s", self.model, self.base_url)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Ollama client is synchronous — run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            ),
        )
        return response["message"]["content"]

    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from image + prompt using Ollama's native images format.

        Ollama accepts base64 images directly in the `images` field of a message.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Ollama multimodal format: images list in user message
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_base64],
        })

        logger.debug("OllamaLLM vision request (model=%s)", self.model)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            ),
        )
        return response["message"]["content"]

    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text with context using Ollama."""
        context_text = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)]
        )
        full_prompt = f"""Based on the following context, answer the query.

{context_text}

Query: {query}

Answer:"""
        return await self.generate(full_prompt, system_prompt, temperature, max_tokens)


# ---------------------------------------------------------------------------
# Placeholder providers (not implemented for this project)
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """OpenAI LLM — placeholder for future implementation."""

    def __init__(self, **kwargs):
        logger.info("OpenAILLM initialized (placeholder)")

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("OpenAILLM not implemented for this project")

    async def generate_with_context(self, query, context, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("OpenAILLM not implemented for this project")


class AnthropicLLM(BaseLLM):
    """Anthropic LLM — placeholder for future implementation."""

    def __init__(self, **kwargs):
        logger.info("AnthropicLLM initialized (placeholder)")

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("AnthropicLLM not implemented for this project")

    async def generate_with_context(self, query, context, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("AnthropicLLM not implemented for this project")


class GoogleLLM(BaseLLM):
    """Google LLM — placeholder for future implementation."""

    def __init__(self, **kwargs):
        logger.info("GoogleLLM initialized (placeholder)")

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("GoogleLLM not implemented for this project")

    async def generate_with_context(self, query, context, system_prompt=None, temperature=0.7, max_tokens=2000):
        raise NotImplementedError("GoogleLLM not implemented for this project")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMFactory:
    """Factory class for creating LLM instances.

    Usage:
        llm = LLMFactory.create_llm(provider="ollama", model="qwen2-vl")
        llm = LLMFactory.create_llm(provider="docker", model="medgemma")
    """

    PROVIDERS = {
        "ollama": OllamaLLM,
        "docker": DockerLLM,
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GoogleLLM,
    }

    @staticmethod
    def create_llm(provider: str = None, **kwargs) -> BaseLLM:
        """Create an LLM instance based on provider.

        Args:
            provider: LLM provider ('ollama', 'docker', 'openai', 'anthropic', 'google').
            **kwargs: Additional arguments passed to the provider constructor
                      (e.g., model, base_url, timeout).

        Returns:
            BaseLLM instance.

        Raises:
            ValueError: If provider is not supported.
        """
        provider = (provider or settings.default_llm_provider).lower()

        if provider not in LLMFactory.PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported: {list(LLMFactory.PROVIDERS.keys())}"
            )

        logger.info("Creating LLM: provider=%s, kwargs=%s", provider, kwargs)
        return LLMFactory.PROVIDERS[provider](**kwargs)
