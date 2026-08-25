"""LiteLLM adapter for universal model completion support."""

from __future__ import annotations

from typing import Any

from .base import BaseLLM, LLMConfig

import litellm


class LiteLLM(BaseLLM):
    """Universal model completion adapter wrapping the official ``litellm`` SDK.

    Supports OpenAI, Anthropic, Gemini, Ollama, Bedrock, Vertex AI, OpenRouter,
    and 100+ other providers with identical calling semantics.

    Install the dependency before constructing this class:

        pip install litellm
    """

    DEFAULT_MODEL = "gemini-3.1-flash-lite"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: str = "",
        base_url: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        super().__init__(
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            config=config,
        )
        self.base_url = base_url
        self.api_version = api_version
        self.organization = organization

        self._litellm = litellm

    def _prepare_call_kwargs(self, prompt: str) -> dict[str, Any]:
        """Build keyword arguments for litellm.completion / litellm.acompletion."""
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.organization:
            kwargs["organization"] = self.organization
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)

        return kwargs

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract generated message content from a LiteLLM response object."""
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content or ""
        return ""

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt synchronously via LiteLLM and return generated text."""
        kwargs = self._prepare_call_kwargs(prompt)
        response = self._litellm.completion(**kwargs)
        return self._extract_text(response)

    async def _call_llm_async(self, prompt: str) -> str:
        """Send a prompt asynchronously via LiteLLM and return generated text."""
        kwargs = self._prepare_call_kwargs(prompt)
        response = await self._litellm.acompletion(**kwargs)
        return self._extract_text(response)
