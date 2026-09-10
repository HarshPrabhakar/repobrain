from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from repobrain.conversation import (
    InvestigationStateManager,
    StatefulRepoBrainOrchestrator,
    UnresolvedFollowUpReferenceError,
)

from repobrain.models.agent import (
    AgentGraphFact,
    AgentIntent,
)

from repobrain.models.conversation import (
    FollowUpReferenceKind,
    InvestigationState,
    RelationshipFocusType,
)

from repobrain.models.symbols import (
    RelationshipType,
)


CURRENT = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._calculate_sha256"
)

CALLER = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._build_file_metadata"
)

CALLEE = (
    "pkg.Scanner.helper"
)


# ======================================================================
# Test doubles
# ======================================================================


@dataclass
class FakeAgentState:
    intent: AgentIntent


@dataclass
class FakeAgentResult:
    state: FakeAgentState

    resolved_symbol_id: str | None = None

    resolved_qualified_name: str | None = None

    graph_facts: list[
        AgentGraphFact
    ] = field(
        default_factory=list
    )


class RecordingAgentRunner:
    """
    Fake existing RepoBrain agent.

    Records every query that Phase 9 sends into it.
    """

    def __init__(
        self,
        *,
        result: FakeAgentResult,
    ) -> None:

        self.result = (
            result
        )

        self.queries: list[
            str
        ] = []

    def __call__(
        self,
        query: str,
    ) -> FakeAgentResult:

        self.queries.append(
            query
        )

        return (
            self.result
        )


def make_focused_state(
) -> InvestigationState:

    return InvestigationState(
        turn_number=1,
        current_symbol_id=(
            "symbol-current"
        ),
        current_qualified_name=(
            CURRENT
        ),
    )


def make_call_fact(
    *,
    source_id: str,
    source_name: str,
    target_id: str,
    target_name: str,
) -> AgentGraphFact:

    return AgentGraphFact(
        relationship_type=(
            RelationshipType.CALLS
        ),
        source_symbol_id=(
            source_id
        ),
        source_qualified_name=(
            source_name
        ),
        target_symbol_id=(
            target_id
        ),
        target_qualified_name=(
            target_name
        ),
    )


# ======================================================================
# Existing primary-focus behavior
# ======================================================================


def test_plain_query_passes_to_agent_unchanged() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "symbol-scanner"
            ),
            resolved_qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner"
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    turn = orchestrator.run(
        "Where is RepositoryScanner defined?"
    )

    assert runner.queries == [
        "Where is RepositoryScanner defined?"
    ]

    assert (
        turn.original_query
        == "Where is RepositoryScanner defined?"
    )

    assert (
        turn.resolved_query
        == "Where is RepositoryScanner defined?"
    )


def test_agent_resolved_symbol_becomes_current_focus() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "symbol-scanner"
            ),
            resolved_qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner"
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    turn = orchestrator.run(
        "Where is RepositoryScanner defined?"
    )

    assert (
        turn.state_after
        .current_symbol_id
        == "symbol-scanner"
    )

    assert (
        turn.state_after
        .current_qualified_name
        == (
            "repobrain.ingestion.scanner."
            "RepositoryScanner"
        )
    )


def test_followup_query_is_rewritten_before_agent() -> None:

    state_manager = (
        InvestigationStateManager(
            initial_state=(
                make_focused_state()
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLEES
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=(
                state_manager
            ),
        )
    )

    turn = orchestrator.run(
        "What does it call?"
    )

    expected = (
        f"What does {CURRENT} call?"
    )

    assert (
        runner.queries
        == [
            expected,
        ]
    )

    assert (
        turn.resolved_query
        == expected
    )

    assert (
        turn.followup_resolution
        .used_context
        is True
    )


def test_followup_retains_focus_when_agent_has_no_new_symbol() -> None:

    state_manager = (
        InvestigationStateManager(
            initial_state=(
                make_focused_state()
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLEES
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=(
                state_manager
            ),
        )
    )

    turn = orchestrator.run(
        "What does it call?"
    )

    assert (
        turn.state_after
        .current_symbol_id
        == "symbol-current"
    )

    assert (
        turn.state_after
        .current_qualified_name
        == CURRENT
    )


def test_state_records_original_and_resolved_queries() -> None:

    state_manager = (
        InvestigationStateManager(
            initial_state=(
                make_focused_state()
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLEES
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=(
                state_manager
            ),
        )
    )

    orchestrator.run(
        "What does it call?"
    )

    state = (
        orchestrator.state
    )

    assert (
        state.last_query
        == "What does it call?"
    )

    assert (
        state.last_resolved_query
        == f"What does {CURRENT} call?"
    )


def test_turn_number_increments() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    assert (
        orchestrator.state
        .turn_number
        == 0
    )

    orchestrator.run(
        "What is this project?"
    )

    assert (
        orchestrator.state
        .turn_number
        == 1
    )

    orchestrator.run(
        "Explain the architecture."
    )

    assert (
        orchestrator.state
        .turn_number
        == 2
    )


def test_unresolved_reference_never_reaches_agent() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLEES
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    with pytest.raises(
        UnresolvedFollowUpReferenceError,
        match=(
            "Cannot resolve follow-up reference"
        ),
    ):

        orchestrator.run(
            "What does it call?"
        )

    assert (
        runner.queries
        == []
    )

    assert (
        orchestrator.state
        .turn_number
        == 0
    )


# ======================================================================
# Phase 9.6 relationship focus
# ======================================================================


def test_single_caller_becomes_relationship_focus() -> None:

    fact = make_call_fact(
        source_id=(
            "symbol-caller"
        ),
        source_name=(
            CALLER
        ),
        target_id=(
            "symbol-current"
        ),
        target_name=(
            CURRENT
        ),
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLERS
                )
            ),
            resolved_symbol_id=(
                "symbol-current"
            ),
            resolved_qualified_name=(
                CURRENT
            ),
            graph_facts=[
                fact,
            ],
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    turn = orchestrator.run(
        "Who calls _calculate_sha256?"
    )

    assert (
        turn.state_after
        .current_qualified_name
        == CURRENT
    )

    assert (
        turn.state_after
        .relationship_focus_type
        == RelationshipFocusType.CALLER
    )

    assert (
        turn.state_after
        .relationship_symbol_id
        == "symbol-caller"
    )

    assert (
        turn.state_after
        .relationship_qualified_name
        == CALLER
    )


def test_single_callee_becomes_relationship_focus() -> None:

    fact = make_call_fact(
        source_id=(
            "symbol-current"
        ),
        source_name=(
            CURRENT
        ),
        target_id=(
            "symbol-callee"
        ),
        target_name=(
            CALLEE
        ),
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLEES
                )
            ),
            resolved_symbol_id=(
                "symbol-current"
            ),
            resolved_qualified_name=(
                CURRENT
            ),
            graph_facts=[
                fact,
            ],
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    turn = orchestrator.run(
        "What does current call?"
    )

    assert (
        turn.state_after
        .relationship_focus_type
        == RelationshipFocusType.CALLEE
    )

    assert (
        turn.state_after
        .relationship_symbol_id
        == "symbol-callee"
    )

    assert (
        turn.state_after
        .relationship_qualified_name
        == CALLEE
    )


def test_zero_callers_clears_relationship_focus() -> None:

    state = InvestigationState(
        relationship_focus_type=(
            RelationshipFocusType.CALLER
        ),
        relationship_symbol_id=(
            "old-caller"
        ),
        relationship_qualified_name=(
            "pkg.old_caller"
        ),
    )

    manager = (
        InvestigationStateManager(
            initial_state=(
                state
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLERS
                )
            ),
            graph_facts=[],
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    turn = orchestrator.run(
        "Who calls function?"
    )

    assert (
        turn.state_after
        .relationship_focus_type
        is None
    )


def test_multiple_callers_do_not_create_relationship_focus() -> None:

    fact_one = make_call_fact(
        source_id="caller-1",
        source_name="pkg.caller_one",
        target_id="target",
        target_name=CURRENT,
    )

    fact_two = make_call_fact(
        source_id="caller-2",
        source_name="pkg.caller_two",
        target_id="target",
        target_name=CURRENT,
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.CALLERS
                )
            ),
            graph_facts=[
                fact_one,
                fact_two,
            ],
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    turn = orchestrator.run(
        "Who calls function?"
    )

    assert (
        turn.state_after
        .relationship_focus_type
        is None
    )

    assert (
        turn.state_after
        .relationship_qualified_name
        is None
    )


def test_relationship_followup_rewrites_before_agent() -> None:

    initial_state = InvestigationState(
        turn_number=1,
        current_symbol_id=(
            "symbol-current"
        ),
        current_qualified_name=(
            CURRENT
        ),
        relationship_focus_type=(
            RelationshipFocusType.CALLER
        ),
        relationship_symbol_id=(
            "symbol-caller"
        ),
        relationship_qualified_name=(
            CALLER
        ),
    )

    manager = (
        InvestigationStateManager(
            initial_state=(
                initial_state
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "symbol-caller"
            ),
            resolved_qualified_name=(
                CALLER
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    turn = orchestrator.run(
        "Where is the caller defined?"
    )

    expected = (
        f"Where is {CALLER} defined?"
    )

    assert (
        runner.queries
        == [
            expected,
        ]
    )

    assert (
        turn.resolved_query
        == expected
    )

    assert (
        turn.followup_resolution
        .reference_kind
        == FollowUpReferenceKind.CALLER_FOCUS
    )


def test_relation_followup_moves_caller_to_primary_focus() -> None:

    initial_state = InvestigationState(
        turn_number=1,
        current_symbol_id="target",
        current_qualified_name=(
            CURRENT
        ),
        relationship_focus_type=(
            RelationshipFocusType.CALLER
        ),
        relationship_symbol_id=(
            "caller"
        ),
        relationship_qualified_name=(
            CALLER
        ),
    )

    manager = (
        InvestigationStateManager(
            initial_state=(
                initial_state
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "caller"
            ),
            resolved_qualified_name=(
                CALLER
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    turn = orchestrator.run(
        "Where is the caller defined?"
    )

    assert (
        turn.state_after
        .current_qualified_name
        == CALLER
    )

    assert (
        turn.state_after
        .previous_qualified_name
        == CURRENT
    )

    assert (
        turn.state_after
        .relationship_qualified_name
        == CALLER
    )


def test_caller_followup_without_relationship_focus_is_rejected() -> None:

    manager = (
        InvestigationStateManager(
            initial_state=(
                make_focused_state()
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    with pytest.raises(
        UnresolvedFollowUpReferenceError
    ):

        orchestrator.run(
            "Where is the caller defined?"
        )

    assert (
        runner.queries
        == []
    )


def test_plain_turn_clears_stale_relationship_focus() -> None:

    initial_state = InvestigationState(
        turn_number=1,
        current_qualified_name=(
            CURRENT
        ),
        relationship_focus_type=(
            RelationshipFocusType.CALLER
        ),
        relationship_symbol_id=(
            "caller"
        ),
        relationship_qualified_name=(
            CALLER
        ),
    )

    manager = (
        InvestigationStateManager(
            initial_state=(
                initial_state
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "other"
            ),
            resolved_qualified_name=(
                "pkg.Other.function"
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    turn = orchestrator.run(
        "Where is Other.function defined?"
    )

    assert (
        turn.state_after
        .relationship_focus_type
        is None
    )


def test_clear_resets_relationship_focus() -> None:

    initial_state = InvestigationState(
        relationship_focus_type=(
            RelationshipFocusType.CALLER
        ),
        relationship_symbol_id=(
            "caller"
        ),
        relationship_qualified_name=(
            CALLER
        ),
    )

    manager = (
        InvestigationStateManager(
            initial_state=(
                initial_state
            )
        )
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=manager,
        )
    )

    state = (
        orchestrator.clear()
    )

    assert (
        state.relationship_focus_type
        is None
    )

    assert (
        state.relationship_qualified_name
        is None
    )


# ======================================================================
# Existing context-control behavior
# ======================================================================


def test_clear_resets_context_but_preserves_conversation_id() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    orchestrator.run(
        "What is RepoBrain?"
    )

    old_id = (
        orchestrator.state
        .conversation_id
    )

    state = (
        orchestrator.clear()
    )

    assert (
        state.conversation_id
        == old_id
    )

    assert state.turn_number == 0

    assert state.history == ()

    assert state.has_focus is False


def test_new_conversation_changes_id() -> None:

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
        )
    )

    old_id = (
        orchestrator.state
        .conversation_id
    )

    state = (
        orchestrator
        .new_conversation()
    )

    assert (
        state.conversation_id
        != old_id
    )

    assert state.turn_number == 0


def test_agent_new_focus_moves_old_focus_to_previous() -> None:

    state_manager = (
        InvestigationStateManager(
            initial_state=(
                make_focused_state()
            )
        )
    )

    new_name = (
        "repobrain.ingestion.scanner."
        "RepositoryScanner._build_file_metadata"
    )

    runner = RecordingAgentRunner(
        result=FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.DEFINITION
                )
            ),
            resolved_symbol_id=(
                "symbol-new"
            ),
            resolved_qualified_name=(
                new_name
            ),
        )
    )

    orchestrator = (
        StatefulRepoBrainOrchestrator(
            agent_runner=runner,
            state_manager=(
                state_manager
            ),
        )
    )

    turn = orchestrator.run(
        "Where is RepositoryScanner."
        "_build_file_metadata defined?"
    )

    assert (
        turn.state_after
        .current_symbol_id
        == "symbol-new"
    )

    assert (
        turn.state_after
        .current_qualified_name
        == new_name
    )

    assert (
        turn.state_after
        .previous_symbol_id
        == "symbol-current"
    )

    assert (
        turn.state_after
        .previous_qualified_name
        == CURRENT
    )