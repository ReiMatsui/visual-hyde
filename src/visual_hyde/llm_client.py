"""
Unified LLM client for Visual HyDE.

Abstracts over Anthropic (Claude) and OpenAI (GPT-4o etc.) so that
all VLM-based generation code (matplotlib codegen, TCD-HyDE descriptions)
can switch providers simply by changing VH_GEN_LLM_PROVIDER in .env.

Usage:
    from visual_hyde.llm_client import LLMClient, get_llm_client

    client = get_llm_client()            # uses global config
    response = client.generate(system="...", user="...")

Or pass settings explicitly:
    client = LLMClient.from_settings(my_settings)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from visual_hyde.config import LLMProvider
from visual_hyde.logging import get_logger

if TYPE_CHECKING:
    from visual_hyde.config import GenerationSettings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Minimal interface shared by all LLM provider wrappers."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """
        Send a system + user message pair and return the assistant's reply.

        Args:
            system: System prompt (role instructions).
            user:   User message.

        Returns:
            The model's text response as a plain string.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'anthropic', 'openai')."""
        ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


class AnthropicLLMClient(BaseLLMClient):
    """
    Wraps the Anthropic Python SDK (anthropic>=0.26).

    Args:
        api_key:    Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model:      Claude model string (e.g. 'claude-opus-4-6').
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or None  # None → SDK reads ANTHROPIC_API_KEY
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: uv add anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.debug(f"Anthropic client initialized (model={self._model})")
        return self._client

    def generate(self, system: str, user: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Extract text from first content block
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        raise ValueError("Anthropic response contained no text block")


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAILLMClient(BaseLLMClient):
    """
    Wraps the OpenAI Python SDK (openai>=1.30).

    Compatible with:
      - OpenAI API (GPT-4o, GPT-4-turbo, GPT-3.5-turbo, …)
      - Azure OpenAI (set base_url and api_key accordingly)
      - Any OpenAI-compatible local server (Ollama, LM Studio, vLLM, …)

    Args:
        api_key:    OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model:      Model name (e.g. 'gpt-4o', 'gpt-4-turbo').
        max_tokens: Maximum tokens in the completion.
        base_url:   Optional API base URL override (for Azure / local proxies).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        base_url: str = "",
    ) -> None:
        self._api_key = api_key or None  # None → SDK reads OPENAI_API_KEY
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url or None
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: uv add openai"
                )
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
            logger.debug(
                f"OpenAI client initialized (model={self._model}"
                + (f", base_url={self._base_url}" if self._base_url else "")
                + ")"
            )
        return self._client

    def generate(self, system: str, user: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response contained no content")
        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def LLMClient(settings: GenerationSettings | None = None) -> BaseLLMClient:  # noqa: N802
    """
    Factory function that returns the appropriate provider client.

    Reads VH_GEN_LLM_PROVIDER from settings (or global config) and
    instantiates either AnthropicLLMClient or OpenAILLMClient.

    Args:
        settings: GenerationSettings instance. Uses global config if None.

    Returns:
        An initialized BaseLLMClient ready to call .generate().
    """
    if settings is None:
        from visual_hyde.config import get_settings
        settings = get_settings().generation

    provider = settings.llm_provider

    if provider == LLMProvider.ANTHROPIC:
        client = AnthropicLLMClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.max_code_tokens,
        )
        logger.info(f"LLM provider: Anthropic ({settings.anthropic_model})")
        return client

    elif provider == LLMProvider.OPENAI:
        client = OpenAILLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.max_code_tokens,
            base_url=settings.openai_base_url,
        )
        logger.info(f"LLM provider: OpenAI ({settings.openai_model})")
        return client

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_client() -> BaseLLMClient:
    """Convenience wrapper: returns a client using the global config."""
    return LLMClient()


from typing import Any
