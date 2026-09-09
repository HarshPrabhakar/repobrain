from __future__ import annotations

import os

from ollama import Client

from repobrain.llm.base import (
    LLMProvider,
)


DEFAULT_OLLAMA_HOST = (
    "http://localhost:11434"
)

DEFAULT_OLLAMA_MODEL = (
    "qwen2.5-coder:14b"
)


class OllamaLLMProvider(
    LLMProvider
):
    """
    Local Ollama-backed LLM provider.

    Default model:
        qwen2.5-coder:14b

    No cloud API key is required.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        host: str | None = None,
        temperature: float = 0.2,
    ) -> None:

        self._model_name = (
            model_name
            or os.getenv(
                "REPOBRAIN_OLLAMA_MODEL"
            )
            or DEFAULT_OLLAMA_MODEL
        )

        self._host = (
            host
            or os.getenv(
                "REPOBRAIN_OLLAMA_HOST"
            )
            or DEFAULT_OLLAMA_HOST
        )

        if not self._model_name.strip():
            raise ValueError(
                "model_name must not be empty."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        self._temperature = float(
            temperature
        )

        self._client = Client(
            host=self._host
        )

    @property
    def model_name(
        self,
    ) -> str:
        return self._model_name

    @property
    def host(
        self,
    ) -> str:
        return self._host

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:

        if not instructions.strip():
            raise ValueError(
                "instructions must not be empty."
            )

        if not input_text.strip():
            raise ValueError(
                "input_text must not be empty."
            )

        response = self._client.chat(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            options={
                "temperature": (
                    self._temperature
                )
            },
        )

        content = (
            response.message.content
            or ""
        ).strip()

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return content