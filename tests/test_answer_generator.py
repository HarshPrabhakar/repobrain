from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
)

from repobrain.llm.base import (
    LLMProvider,
)

from tests.test_prompt_builder import (
    make_result,
)


class FakeProvider(
    LLMProvider
):

    def __init__(
        self,
        response: str,
    ) -> None:

        self.response = response

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

        return self.response


def test_valid_citation_is_resolved() -> None:

    generator = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "The hash function is `_hash`. [E1]"
            )
        )
    )

    answer = generator.generate(
        make_result()
    )

    assert len(
        answer.citations
    ) == 1

    assert (
        answer.citations[0]
        .relative_path
        == "app/scanner.py"
    )


def test_invalid_citation_detected() -> None:

    generator = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "This is supported by [E99]."
            )
        )
    )

    answer = generator.generate(
        make_result()
    )

    assert (
        answer.invalid_citation_ids
        == [
            "E99",
        ]
    )

    assert answer.grounded is False


def test_duplicate_citations_deduplicated() -> None:

    generator = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "Supported [E1]. Again [E1]."
            )
        )
    )

    answer = generator.generate(
        make_result()
    )

    assert (
        len(
            answer.citations
        )
        == 1
    )


def test_model_name_preserved() -> None:

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "Answer [E1]"
            )
        )
        .generate(
            make_result()
        )
    )

    assert (
        answer.model_name
        == "fake-model"
    )