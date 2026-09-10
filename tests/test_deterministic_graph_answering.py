from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
)

from repobrain.llm import (
    LLMProvider,
)

from repobrain.models.agent import (
    AgentGraphFact,
    AgentIntent,
    AgentRunResult,
    AgentState,
    AgentStatus,
)

from repobrain.models.evidence import (
    EvidenceBundle,
)

from repobrain.models.symbols import (
    RelationshipType,
)


# =============================================================================
# Test LLM provider
# =============================================================================


class RecordingProvider(
    LLMProvider
):
    """
    Fake provider used to prove that structural
    CALLERS/CALLEES answers never reach the LLM.
    """

    def __init__(
        self,
    ) -> None:
        self.calls = 0

    @property
    def model_name(
        self,
    ) -> str:
        return "recording-provider"

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.calls += 1

        raise AssertionError(
            "LLM provider must not be called for "
            "deterministic graph answers."
        )


# =============================================================================
# Helpers
# =============================================================================


def make_fact(
    *,
    source: str,
    target: str,
) -> AgentGraphFact:
    """
    Build one deterministic internal CALLS fact.
    """

    return AgentGraphFact(
        relationship_type=(
            RelationshipType.CALLS
        ),
        source_symbol_id=(
            f"sym_{source}"
        ),
        source_qualified_name=(
            source
        ),
        target_symbol_id=(
            f"sym_{target}"
        ),
        target_qualified_name=(
            target
        ),
    )


def make_result(
    *,
    intent: AgentIntent,
    resolved_name: str,
    facts: list[AgentGraphFact],
) -> AgentRunResult:
    """
    Build a minimal completed AgentRunResult for
    deterministic graph-answering tests.
    """

    evidence = EvidenceBundle(
        query="test query",
        items=[],
        max_items=8,
        max_characters=24_000,
    )

    return AgentRunResult(
        state=AgentState(
            query="test query",
            intent=intent,
            status=(
                AgentStatus.COMPLETED
            ),
        ),
        evidence=evidence,
        graph_facts=facts,
        resolved_symbol_id=(
            "sym_resolved"
        ),
        resolved_qualified_name=(
            resolved_name
        ),
    )


# =============================================================================
# CALLERS
# =============================================================================


def test_callers_bypass_llm() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLERS
        ),
        resolved_name=(
            "pkg.Scanner._hash"
        ),
        facts=[
            make_fact(
                source=(
                    "pkg.Scanner.build"
                ),
                target=(
                    "pkg.Scanner._hash"
                ),
            )
        ],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    assert (
        provider.calls
        == 0
    )

    assert (
        answer.model_name
        == "deterministic-graph"
    )

    assert (
        "pkg.Scanner.build"
        in answer.answer_text
    )

    assert (
        "pkg.Scanner._hash"
        in answer.answer_text
    )

    assert (
        answer.graph_facts_available
        == 1
    )

    assert (
        answer.grounded
        is True
    )


# =============================================================================
# CALLEES
# =============================================================================


def test_callees_bypass_llm() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLEES
        ),
        resolved_name=(
            "pkg.Scanner.scan"
        ),
        facts=[
            make_fact(
                source=(
                    "pkg.Scanner.scan"
                ),
                target=(
                    "pkg.Scanner.walk"
                ),
            ),
            make_fact(
                source=(
                    "pkg.Scanner.scan"
                ),
                target=(
                    "pkg.Scanner.build"
                ),
            ),
        ],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    assert (
        provider.calls
        == 0
    )

    assert (
        answer.model_name
        == "deterministic-graph"
    )

    assert (
        "pkg.Scanner.walk"
        in answer.answer_text
    )

    assert (
        "pkg.Scanner.build"
        in answer.answer_text
    )

    assert (
        answer.graph_facts_available
        == 2
    )

    assert (
        answer.grounded
        is True
    )


# =============================================================================
# Citation generation
# =============================================================================


def test_every_graph_fact_becomes_citation() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLEES
        ),
        resolved_name=(
            "pkg.Scanner.scan"
        ),
        facts=[
            make_fact(
                source=(
                    "pkg.Scanner.scan"
                ),
                target=(
                    "pkg.Scanner.walk"
                ),
            ),
            make_fact(
                source=(
                    "pkg.Scanner.scan"
                ),
                target=(
                    "pkg.Scanner.build"
                ),
            ),
            make_fact(
                source=(
                    "pkg.Scanner.scan"
                ),
                target=(
                    "pkg.Scanner.finish"
                ),
            ),
        ],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    assert (
        provider.calls
        == 0
    )

    assert (
        len(
            answer.citations
        )
        == 3
    )

    citation_ids = [
        citation.citation_id
        for citation
        in answer.citations
    ]

    assert citation_ids == [
        "G1",
        "G2",
        "G3",
    ]

    assert (
        answer.graph_facts_available
        == 3
    )

    assert (
        answer.grounded
        is True
    )


# =============================================================================
# Citation validity
# =============================================================================


def test_graph_answer_has_no_invalid_citations() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLERS
        ),
        resolved_name=(
            "pkg.Scanner._hash"
        ),
        facts=[
            make_fact(
                source=(
                    "pkg.Scanner.build"
                ),
                target=(
                    "pkg.Scanner._hash"
                ),
            )
        ],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    assert (
        provider.calls
        == 0
    )

    assert (
        answer.invalid_citation_ids
        == []
    )

    assert (
        answer.grounded
        is True
    )


# =============================================================================
# Empty CALLEES result
# =============================================================================


def test_empty_callees_is_deterministic() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLEES
        ),
        resolved_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner._calculate_sha256"
        ),
        facts=[],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    # ---------------------------------------------------------
    # The LLM must never be invoked.
    # ---------------------------------------------------------

    assert (
        provider.calls
        == 0
    )

    # ---------------------------------------------------------
    # This is a deterministic structural answer.
    # ---------------------------------------------------------

    assert (
        answer.model_name
        == "deterministic-graph"
    )

    assert (
        "does not call any resolved "
        "internal repository symbols"
        in answer.answer_text
    )

    assert (
        answer.graph_facts_available
        == 0
    )

    assert (
        answer.citations
        == []
    )

    assert (
        answer.invalid_citation_ids
        == []
    )

    assert (
        answer.grounded
        is True
    )


# =============================================================================
# Empty CALLERS result
# =============================================================================


def test_empty_callers_is_deterministic() -> None:

    result = make_result(
        intent=(
            AgentIntent.CALLERS
        ),
        resolved_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner.scan"
        ),
        facts=[],
    )

    provider = (
        RecordingProvider()
    )

    answer = (
        GroundedAnswerGenerator(
            provider=provider
        )
        .generate(
            result
        )
    )

    # ---------------------------------------------------------
    # The LLM must never be invoked.
    # ---------------------------------------------------------

    assert (
        provider.calls
        == 0
    )

    # ---------------------------------------------------------
    # This is a deterministic structural answer.
    # ---------------------------------------------------------

    assert (
        answer.model_name
        == "deterministic-graph"
    )

    assert (
        "has no resolved internal "
        "repository callers"
        in answer.answer_text
    )

    assert (
        answer.graph_facts_available
        == 0
    )

    assert (
        answer.citations
        == []
    )

    assert (
        answer.invalid_citation_ids
        == []
    )

    assert (
        answer.grounded
        is True
    )