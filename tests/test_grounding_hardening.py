from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
    GroundedPromptBuilder,
)

from repobrain.llm.base import (
    LLMProvider,
)

from repobrain.models.agent import (
    AgentGraphFact,
    AgentIntent,
    AgentRunResult,
    AgentState,
    AgentStatus,
)

from repobrain.models.answer import (
    AnswerCitationKind,
)

from repobrain.models.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
)

from repobrain.models.hybrid import (
    RetrievalChannel,
)

from repobrain.models.symbols import (
    RelationshipType,
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


def make_source_item(
    *,
    evidence_id: str = "ev1",
    path: str = "app/scanner.py",
    symbol: str = "app.Scanner._hash",
    kind: EvidenceKind = EvidenceKind.SYMBOL,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        result_key=f"symbol:{evidence_id}",
        file_id=f"file_{evidence_id}",
        relative_path=path,
        symbol_id=evidence_id,
        qualified_name=symbol,
        start_line=10,
        end_line=20,
        source_text=(
            "def _hash(path):\n"
            "    return digest(path)"
        ),
        evidence_kind=kind,
        fused_score=0.5,
        retrieval_channels=[
            RetrievalChannel.SEMANTIC,
        ],
        character_count=40,
        estimated_tokens=10,
    )


def make_result(
    *,
    intent: AgentIntent = AgentIntent.IMPLEMENTATION,
    items: list[EvidenceItem] | None = None,
    graph_facts: list[AgentGraphFact] | None = None,
) -> AgentRunResult:
    evidence_items = (
        items
        if items is not None
        else [
            make_source_item()
        ]
    )

    bundle = EvidenceBundle(
        query="question",
        items=evidence_items,
        total_characters=sum(
            item.character_count
            for item in evidence_items
        ),
        estimated_tokens=sum(
            item.estimated_tokens
            for item in evidence_items
        ),
        omitted_items=0,
        max_items=8,
        max_characters=24_000,
    )

    state = AgentState(
        query="question",
        intent=intent,
        status=AgentStatus.COMPLETED,
        max_steps=3,
    )

    return AgentRunResult(
        state=state,
        evidence=bundle,
        graph_facts=(
            graph_facts
            or []
        ),
    )


def make_call_fact() -> AgentGraphFact:
    return AgentGraphFact(
        relationship_type=RelationshipType.CALLS,
        source_symbol_id="caller",
        source_qualified_name="app.Scanner.build",
        target_symbol_id="hash",
        target_qualified_name="app.Scanner._hash",
    )


def test_graph_fact_receives_g_citation() -> None:
    result = make_result(
        intent=AgentIntent.CALLERS,
        graph_facts=[
            make_call_fact(),
        ],
    )

    prompt, lookup = (
        GroundedPromptBuilder()
        .build(
            result
        )
    )

    assert "[G1]" in prompt
    assert "G1" in lookup

    assert (
        lookup["G1"]
        .source_qualified_name
        == "app.Scanner.build"
    )


def test_valid_graph_citation_is_resolved() -> None:
    """
    Use GENERAL intent so the LLM-backed citation-validation path
    is exercised.

    CALLERS/CALLEES are now rendered deterministically.
    """

    result = make_result(
        intent=AgentIntent.GENERAL,
        graph_facts=[
            make_call_fact(),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "`build` calls `_hash`. [G1]"
            )
        )
        .generate(
            result
        )
    )

    assert len(
        answer.citations
    ) == 1

    citation = answer.citations[0]

    assert (
        citation.citation_kind
        == AnswerCitationKind.GRAPH
    )

    assert (
        citation.source_qualified_name
        == "app.Scanner.build"
    )

    assert (
        citation.target_qualified_name
        == "app.Scanner._hash"
    )

    assert answer.grounded is True


def test_invalid_graph_citation_is_rejected() -> None:
    """
    Invalid citation testing belongs to the LLM-backed path.

    CALLERS/CALLEES bypass the LLM entirely, so GENERAL is used.
    """

    result = make_result(
        intent=AgentIntent.GENERAL,
        graph_facts=[
            make_call_fact(),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "This relationship exists. [G99]"
            )
        )
        .generate(
            result
        )
    )

    assert (
        answer.invalid_citation_ids
        == [
            "G99",
        ]
    )

    assert answer.grounded is False


def test_zero_citation_repository_answer_is_not_grounded() -> None:
    result = make_result()

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                (
                    "The repository contains "
                    "a hash implementation."
                )
            )
        )
        .generate(
            result
        )
    )

    assert answer.citations == []
    assert answer.grounded is False


def test_valid_source_citation_remains_grounded() -> None:
    result = make_result()

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                "The implementation is here. [E1]"
            )
        )
        .generate(
            result
        )
    )

    assert len(
        answer.citations
    ) == 1

    assert (
        answer.citations[0]
        .citation_kind
        == AnswerCitationKind.SOURCE
    )

    assert answer.grounded is True


def test_duplicate_graph_citations_are_deduplicated() -> None:
    """
    Use GENERAL so graph citation parsing is tested through
    the LLM path rather than the deterministic renderer.
    """

    result = make_result(
        intent=AgentIntent.GENERAL,
        graph_facts=[
            make_call_fact(),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                (
                    "Supported [G1]. "
                    "Again [G1]."
                )
            )
        )
        .generate(
            result
        )
    )

    assert (
        len(
            answer.citations
        )
        == 1
    )


def test_documentation_intent_prefers_documentation() -> None:
    source_item = (
        make_source_item(
            evidence_id="source"
        )
    )

    readme = (
        make_source_item(
            evidence_id="readme",
            path="README.md",
            symbol="README.md",
            kind=EvidenceKind.DOCUMENTATION,
        )
    )

    result = make_result(
        intent=AgentIntent.DOCUMENTATION,
        items=[
            source_item,
            readme,
        ],
    )

    prompt, lookup = (
        GroundedPromptBuilder()
        .build(
            result
        )
    )

    assert (
        lookup["E1"]
        .evidence_kind
        == EvidenceKind.DOCUMENTATION
    )

    assert (
        "app/scanner.py"
        not in prompt
    )


def test_implementation_intent_removes_test_noise_when_production_exists() -> None:
    production = (
        make_source_item(
            evidence_id="production",
            path="app/scanner.py",
            kind=EvidenceKind.SYMBOL,
        )
    )

    test_item = (
        make_source_item(
            evidence_id="test",
            path="tests/test_scanner.py",
            symbol="tests.test_scanner",
            kind=EvidenceKind.TEST,
        )
    )

    result = make_result(
        intent=AgentIntent.IMPLEMENTATION,
        items=[
            production,
            test_item,
        ],
    )

    prompt, lookup = (
        GroundedPromptBuilder()
        .build(
            result
        )
    )

    assert (
        lookup["E1"]
        .evidence_id
        == "production"
    )

    assert (
        "tests/test_scanner.py"
        not in prompt
    )


def test_mixed_source_and_graph_citations_are_supported() -> None:
    """
    Mixed E/G citation validation must run through the LLM-backed path.

    CALLERS and CALLEES now bypass the LLM and render directly from
    deterministic graph facts.

    Therefore GENERAL intent is required here.
    """

    result = make_result(
        intent=AgentIntent.GENERAL,
        graph_facts=[
            make_call_fact(),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FakeProvider(
                (
                    "`build` calls `_hash`. "
                    "[G1] "
                    "The implementation is shown "
                    "in source. [E1]"
                )
            )
        )
        .generate(
            result
        )
    )

    assert (
        len(
            answer.citations
        )
        == 2
    )

    assert {
        citation.citation_kind
        for citation in answer.citations
    } == {
        AnswerCitationKind.SOURCE,
        AnswerCitationKind.GRAPH,
    }

    assert answer.grounded is True