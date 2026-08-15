"""Language-model adapters used by the RLM controller."""

from __future__ import annotations

from typing import Any


class LLMCallError(RuntimeError):
    """Raised when a configured model cannot produce a text response."""


class BaseLLM:
    """Minimal text-completion interface consumed by ``RLM``."""

    def __init__(self, provider: str, api_key: str) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        self.provider = provider
        self.api_key = api_key

    def call_llm(self, prompt: str) -> str:
        """Return a text completion for ``prompt``.

        Provider adapters must override this method.
        """
        raise NotImplementedError("BaseLLM does not implement a provider call")


class GeminiLLM(BaseLLM):
    """Text-only Gemini Developer API adapter using the Google Gen AI SDK.

    Install the optional dependency before constructing this class:

        pip install google-genai
    """

    DEFAULT_MODEL = "gemini-3.7-flash"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__(provider="gemini", api_key=api_key)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        try:
            from google import genai
        except ImportError as error:
            raise ImportError(
                "GeminiLLM requires the Google Gen AI SDK. Install it with: pip install google-genai"
            ) from error

        self.model_name = model_name
        self._client: Any = genai.Client(api_key=api_key)

    def call_llm(self, prompt: str) -> str:
        """Send a text prompt to Gemini and return its generated text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = response.text
        except Exception as error:
            raise LLMCallError(
                f"Gemini request failed for model '{self.model_name}'."
            ) from error

        if not isinstance(text, str) or not text.strip():
            raise LLMCallError(
                f"Gemini returned no text for model '{self.model_name}'."
            )
        return text
