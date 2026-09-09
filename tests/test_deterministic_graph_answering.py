from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
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

from repobrain.models.symbols import (
    RelationshipType,
)


class FailingProvider(
    LLMProvider
):
    """
    The provider must never be called for deterministic
    CALLERS/CALLEES answers.
    """

    @property
    def model_name(
        self,
    ) -> str:
        return "should-not-run"

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:

        raise AssertionError(
            "LLM must not be called "
            "for deterministic graph questions."
        )


def make_fact(
    *,
    source: str,
    target: str,
) -> AgentGraphFact:

    return AgentGraphFact(
        relationship_type=(
            RelationshipType.CALLS
        ),

        source_symbol_id=(
            f"id:{source}"
        ),

        source_qualified_name=source,

        target_symbol_id=(
            f"id:{target}"
        ),

        target_qualified_name=target,
    )


def make_result(
    *,
    intent: AgentIntent,
    facts: list[
        AgentGraphFact
    ],
    resolved_name: str,
) -> AgentRunResult:

    state = AgentState(
        query="graph question",

        intent=intent,

        status=(
            AgentStatus.COMPLETED
        ),

        max_steps=3,
    )

    return AgentRunResult(
        state=state,

        evidence=None,

        graph_facts=facts,

        resolved_qualified_name=(
            resolved_name
        ),
    )


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

    answer = (
        GroundedAnswerGenerator(
            provider=FailingProvider()
        )
        .generate(
            result
        )
    )

    assert (
        "pkg.Scanner.build"
        in answer.answer_text
    )

    assert (
        "[G1]"
        in answer.answer_text
    )

    assert (
        answer.model_name
        == "deterministic-graph"
    )

    assert (
        answer.grounded
        is True
    )


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
                source="pkg.Scanner.scan",
                target="pkg.Scanner.walk",
            ),

            make_fact(
                source="pkg.Scanner.scan",
                target="pkg.Scanner.build",
            ),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FailingProvider()
        )
        .generate(
            result
        )
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
        "[G1]"
        in answer.answer_text
    )

    assert (
        "[G2]"
        in answer.answer_text
    )

    assert (
        answer.grounded
        is True
    )


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
                source="pkg.Scanner.scan",
                target="pkg.Scanner.walk",
            ),

            make_fact(
                source="pkg.Scanner.scan",
                target="pkg.Scanner.build",
            ),

            make_fact(
                source="pkg.Scanner.scan",
                target="pkg.Scanner.finish",
            ),
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FailingProvider()
        )
        .generate(
            result
        )
    )

    assert (
        len(
            answer.citations
        )
        == 3
    )

    assert all(
        citation.citation_kind
        == AnswerCitationKind.GRAPH

        for citation
        in answer.citations
    )


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
                source="pkg.Scanner.build",
                target="pkg.Scanner._hash",
            )
        ],
    )

    answer = (
        GroundedAnswerGenerator(
            provider=FailingProvider()
        )
        .generate(
            result
        )
    )

    assert (
        answer.invalid_citation_ids
        == []
    )