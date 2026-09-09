from __future__ import annotations

from repobrain.llm.base import (
    LLMProvider,
)


class FakeProvider(
    LLMProvider
):

    @property
    def model_name(
        self,
    ) -> str:

        return "fake-model"

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:

        return "hello"


def test_fake_provider_contract() -> None:

    provider = FakeProvider()

    assert (
        provider.model_name
        == "fake-model"
    )

    assert (
        provider.generate(
            instructions="system",
            input_text="input",
        )
        == "hello"
    )