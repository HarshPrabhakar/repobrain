from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
)

from repobrain.llm.base import (
    LLMProvider,
)

from repobrain.models.agent import (
    AgentIntent,
    AgentRunResult,
    AgentState,
    AgentStatus,
)

from repobrain.models.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
)

from repobrain.models.hybrid import (
    RetrievalChannel,
)


class SequenceProvider(
    LLMProvider
):
    """
    Fake provider that returns predefined responses in order.
    """

    def __init__(
        self,
        responses: list[str],
    ) -> None:

        self.responses = list(
            responses
        )

        self.calls = 0

    @property
    def model_name(
        self,
    ) -> str:

        return "sequence-model"

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:

        if (
            self.calls
            >= len(
                self.responses
            )
        ):
            raise AssertionError(
                "Provider called more times than expected."
            )

        response = (
            self.responses[
                self.calls
            ]
        )

        self.calls += 1

        return response


def make_result() -> AgentRunResult:

    item = EvidenceItem(
        evidence_id="ev1",

        result_key="symbol:ev1",

        file_id="file1",

        relative_path=(
            "repobrain/example.py"
        ),

        symbol_id="sym1",

        qualified_name=(
            "repobrain.example.answer"
        ),

        start_line=10,

        end_line=20,

        source_text=(
            "def answer():\n"
            "    return 42"
        ),

        evidence_kind=(
            EvidenceKind.SYMBOL
        ),

        fused_score=0.9,

        retrieval_channels=[
            RetrievalChannel.SEMANTIC,
        ],

        character_count=30,

        estimated_tokens=8,
    )

    bundle = EvidenceBundle(
        query="what does answer do?",

        items=[
            item,
        ],

        total_characters=30,

        estimated_tokens=8,

        omitted_items=0,

        max_items=8,

        max_characters=24_000,
    )

    state = AgentState(
        query="what does answer do?",

        intent=(
            AgentIntent.IMPLEMENTATION
        ),

        status=(
            AgentStatus.COMPLETED
        ),

        max_steps=3,
    )

    return AgentRunResult(
        state=state,

        evidence=bundle,

        graph_facts=[],
    )


def test_no_retry_when_first_answer_has_valid_citation() -> None:

    provider = SequenceProvider(
        responses=[
            "The function returns 42. [E1]",
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 1
    )

    assert (
        answer.grounded
        is True
    )

    assert (
        len(
            answer.citations
        )
        == 1
    )


def test_retry_occurs_when_first_answer_has_no_citations() -> None:

    provider = SequenceProvider(
        responses=[
            "The function returns 42.",
            "The function returns 42. [E1]",
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 2
    )

    assert (
        answer.grounded
        is True
    )

    assert (
        len(
            answer.citations
        )
        == 1
    )


def test_retry_happens_only_once() -> None:

    provider = SequenceProvider(
        responses=[
            "The function returns 42.",
            "The function still returns 42.",
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 2
    )

    assert (
        answer.grounded
        is False
    )

    assert (
        answer.citations
        == []
    )


def test_invalid_first_citation_can_be_repaired() -> None:

    provider = SequenceProvider(
        responses=[
            "The function returns 42. [E99]",
            "The function returns 42. [E1]",
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 2
    )

    assert (
        answer.grounded
        is True
    )

    assert (
        answer.invalid_citation_ids
        == []
    )

    assert (
        answer.citations[0]
        .citation_id
        == "E1"
    )


def test_failed_repair_preserves_not_grounded_status() -> None:

    provider = SequenceProvider(
        responses=[
            "The function returns 42.",
            "Unsupported answer [E77].",
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 2
    )

    assert (
        answer.grounded
        is False
    )

    assert (
        answer.invalid_citation_ids
        == [
            "E77",
        ]
    )

def test_mixed_valid_and_invalid_citations_trigger_repair() -> None:

    provider = SequenceProvider(
        responses=[
            (
                "The function returns 42. "
                "[E1] Additional claim. [E99]"
            ),
            (
                "The function returns 42. [E1]"
            ),
        ]
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            make_result()
        )
    )

    assert (
        provider.calls
        == 2
    )

    assert (
        answer.grounded
        is True
    )

    assert (
        answer.invalid_citation_ids
        == []
    )

    assert (
        len(
            answer.citations
        )
        == 1
    )

    assert (
        answer.citations[0]
        .citation_id
        == "E1"
    )