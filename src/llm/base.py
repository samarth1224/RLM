"""Base interfaces and configuration models for language-model adapters."""

from __future__ import annotations

import asyncio
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class LLMCallError(RuntimeError):
    """Raised when a configured model cannot produce a text response."""


class LLMConfig(BaseModel):
    """Unified configuration parameters for LLM generation."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    system_prompt: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class BaseLLM:
    """Minimal text-completion interface consumed by ``RLM``."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str = "",
        config: LLMConfig | None = None,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(api_key, str):
            raise ValueError("api_key must be a string")

        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.config = config if config is not None else LLMConfig()

    def call_llm(self, prompt: str) -> str:
        """Public entry point: validates inputs, executes provider call, and validates output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            response = self._call_llm(prompt)
        except Exception as error:
            if isinstance(error, LLMCallError):
                raise
            raise LLMCallError(
                f"{self.provider} request failed: {error}"
            ) from error

        if not isinstance(response, str) or not response.strip():
            raise LLMCallError(
                f"{self.provider} returned an empty or invalid response."
            )
        return response

    def _call_llm(self, prompt: str) -> str:
        """Private synchronous provider call to be overridden by adapter subclasses."""
        raise NotImplementedError("Subclasses must implement _call_llm")

    async def call_llm_async(self, prompt: str) -> str:
        """Public async entry point: validates inputs, awaits provider call, and validates output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            response = await self._call_llm_async(prompt)
        except Exception as error:
            if isinstance(error, LLMCallError):
                raise
            raise LLMCallError(
                f"{self.provider} request failed: {error}"
            ) from error

        if not isinstance(response, str) or not response.strip():
            raise LLMCallError(
                f"{self.provider} returned an empty or invalid response."
            )
        return response

    async def _call_llm_async(self, prompt: str) -> str:
        """Private async provider call. Defaults to executing _call_llm in a worker thread."""
        return await asyncio.to_thread(self._call_llm, prompt)
