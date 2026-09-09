from __future__ import annotations

from types import SimpleNamespace

from repobrain.agent import (
    RepoBrainAgentOrchestrator,
)

from repobrain.models.agent import (
    AgentAction,
    AgentGraphFact,
    AgentIntent,
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

from repobrain.models.symbols import (
    RelationshipType,
)


class FakeToolbox:
    """
    Typed fake toolbox matching the real AgentToolbox contract.
    """

    def __init__(
        self,
    ) -> None:

        self.symbol = SimpleNamespace(
            symbol_id="hash",
            qualified_name=(
                "app.Scanner._hash"
            ),
        )

    def resolve_symbol(
        self,
        reference: str,
    ):

        if reference == "_hash":
            return self.symbol

        return None

    def callers(
        self,
        symbol_id: str,
    ) -> list[AgentGraphFact]:

        if symbol_id != "hash":
            return []

        return [
            AgentGraphFact(
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source_symbol_id=(
                    "caller"
                ),
                source_qualified_name=(
                    "app.Scanner.build"
                ),
                target_symbol_id=(
                    "hash"
                ),
                target_qualified_name=(
                    "app.Scanner._hash"
                ),
            )
        ]

    def callees(
        self,
        symbol_id: str,
    ) -> list[AgentGraphFact]:

        if symbol_id != "hash":
            return []

        return [
            AgentGraphFact(
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source_symbol_id=(
                    "hash"
                ),
                source_qualified_name=(
                    "app.Scanner._hash"
                ),
                target_symbol_id=(
                    "callee_a"
                ),
                target_qualified_name=(
                    "app.helpers.read"
                ),
            ),

            AgentGraphFact(
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source_symbol_id=(
                    "hash"
                ),
                source_qualified_name=(
                    "app.Scanner._hash"
                ),
                target_symbol_id=(
                    "callee_b"
                ),
                target_qualified_name=(
                    "app.helpers.digest"
                ),
            ),
        ]

    def retrieve_evidence(
        self,
        query: str,
        *,
        retrieval_top_k: int = 20,
    ) -> EvidenceBundle:

        items = [
            EvidenceItem(
                evidence_id="evidence_1",
                result_key="symbol:hash",
                file_id="file_hash",
                relative_path=(
                    "app/scanner.py"
                ),
                symbol_id="hash",
                qualified_name=(
                    "app.Scanner._hash"
                ),
                start_line=10,
                end_line=20,
                source_text=(
                    "def _hash(path):\n"
                    "    return digest(path)"
                ),
                evidence_kind=(
                    EvidenceKind.SYMBOL
                ),
                fused_score=0.5,
                retrieval_channels=[
                    RetrievalChannel.SEMANTIC,
                ],
                graph_context=[],
                estimated_tokens=8,
                character_count=32,
            ),

            EvidenceItem(
                evidence_id="evidence_2",
                result_key="symbol:caller",
                file_id="file_caller",
                relative_path=(
                    "app/scanner.py"
                ),
                symbol_id="caller",
                qualified_name=(
                    "app.Scanner.build"
                ),
                start_line=1,
                end_line=8,
                source_text=(
                    "def build():\n"
                    "    return _hash(path)"
                ),
                evidence_kind=(
                    EvidenceKind.SYMBOL
                ),
                fused_score=0.4,
                retrieval_channels=[
                    RetrievalChannel.GRAPH,
                ],
                graph_context=[],
                estimated_tokens=8,
                character_count=32,
            ),
        ]

        return EvidenceBundle(
            query=query,
            items=items,
            total_characters=sum(
                item.character_count
                for item in items
            ),
            estimated_tokens=sum(
                item.estimated_tokens
                for item in items
            ),
            omitted_items=0,
            max_items=8,
            max_characters=24_000,
        )


def test_general_query_runs_evidence_retrieval() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "explain the repository"
    )

    assert (
        result.state.intent
        == AgentIntent.GENERAL
    )

    assert (
        result.state.steps[0].action
        == AgentAction.HYBRID_EVIDENCE
    )

    assert result.evidence is not None

    assert (
        len(
            result.evidence.items
        )
        == 2
    )


def test_implementation_query_uses_evidence() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "which function computes a digest?"
    )

    assert (
        result.state.intent
        == AgentIntent.IMPLEMENTATION
    )

    assert (
        result.state.steps[0].action
        == AgentAction.HYBRID_EVIDENCE
    )

    assert (
        result.graph_facts
        == []
    )


def test_callers_query_runs_graph_first() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "who calls _hash?"
    )

    assert (
        result.state.steps[0].action
        == AgentAction.GRAPH_CALLERS
    )

    assert (
        result.state.steps[1].action
        == AgentAction.HYBRID_EVIDENCE
    )

    assert (
        len(
            result.graph_facts
        )
        == 1
    )

    assert (
        result.graph_facts[0]
        .source_qualified_name
        == "app.Scanner.build"
    )

    assert (
        result.graph_facts[0]
        .target_qualified_name
        == "app.Scanner._hash"
    )


def test_callees_query_runs_graph_first() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "what does _hash call?"
    )

    assert (
        result.state.steps[0].action
        == AgentAction.GRAPH_CALLEES
    )

    assert (
        len(
            result.graph_facts
        )
        == 2
    )

    assert all(
        fact.relationship_type
        == RelationshipType.CALLS
        for fact
        in result.graph_facts
    )


def test_completed_status() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox(),
            max_steps=3,
        )
    )

    result = agent.run(
        "who calls _hash?"
    )

    assert (
        result.state.status
        == AgentStatus.COMPLETED
    )

    assert (
        result.state.step_count
        == 2
    )


def test_max_steps_is_enforced() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox(),
            max_steps=1,
        )
    )

    result = agent.run(
        "who calls _hash?"
    )

    assert (
        len(
            result.state.steps
        )
        == 1
    )

    assert (
        result.state.steps[0].action
        == AgentAction.GRAPH_CALLERS
    )

    assert (
        result.state.status
        == AgentStatus.MAX_STEPS_REACHED
    )

    # Because the single allowed step was consumed by graph
    # inspection, evidence retrieval must not run afterward.
    assert result.evidence is None


def test_callers_query_resolves_symbol() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "who calls _hash?"
    )

    assert (
        result.resolved_symbol_id
        == "hash"
    )

    assert (
        result.resolved_qualified_name
        == "app.Scanner._hash"
    )


def test_unknown_graph_symbol_falls_back_to_evidence() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    result = agent.run(
        "who calls _missing?"
    )

    assert (
        result.state.intent
        == AgentIntent.CALLERS
    )

    assert (
        result.resolved_symbol_id
        is None
    )

    assert (
        result.graph_facts
        == []
    )

    assert (
        len(
            result.state.steps
        )
        == 1
    )

    assert (
        result.state.steps[0].action
        == AgentAction.HYBRID_EVIDENCE
    )


def test_evidence_query_is_preserved() -> None:

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=FakeToolbox()
        )
    )

    query = (
        "which function computes a digest?"
    )

    result = agent.run(
        query
    )

    assert result.evidence is not None

    assert (
        result.evidence.query
        == query
    )