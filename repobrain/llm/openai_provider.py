from __future__ import annotations

import os

from openai import OpenAI

from repobrain.llm.base import (
    LLMProvider,
)


DEFAULT_MAX_OUTPUT_TOKENS = 1200


class OpenAILLMProvider(
    LLMProvider
):
    """
    OpenAI Responses API implementation.

    The model is deliberately configurable instead of being
    hardcoded into RepoBrain.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int = (
            DEFAULT_MAX_OUTPUT_TOKENS
        ),
    ) -> None:

        selected_model = (
            model_name
            or os.getenv(
                "REPOBRAIN_LLM_MODEL"
            )
        )

        if not selected_model:
            raise ValueError(
                "No LLM model configured. "
                "Pass model_name=... or set "
                "REPOBRAIN_LLM_MODEL."
            )

        if max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be > 0."
            )

        self._model_name = (
            selected_model.strip()
        )

        self._max_output_tokens = int(
            max_output_tokens
        )

        resolved_api_key = (
            api_key
            or os.getenv(
                "OPENAI_API_KEY"
            )
        )

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self._client = OpenAI(
            api_key=resolved_api_key
        )

    @property
    def model_name(
        self,
    ) -> str:

        return self._model_name

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

        response = (
            self._client.responses.create(
                model=self._model_name,

                instructions=(
                    instructions
                ),

                input=(
                    input_text
                ),

                max_output_tokens=(
                    self._max_output_tokens
                ),
            )
        )

        output_text = (
            response.output_text
            or ""
        ).strip()

        if not output_text:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return output_text