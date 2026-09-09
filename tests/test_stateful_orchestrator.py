from __future__ import annotations

from dataclasses import dataclass

import pytest

from repobrain.conversation import (
    InvestigationStateManager,
    StatefulRepoBrainOrchestrator,
    UnresolvedFollowUpReferenceError,
)

from repobrain.models.agent import (
    AgentIntent,
)

from repobrain.models.conversation import (
    InvestigationState,
)


CURRENT = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._calculate_sha256"
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

        self.result = result

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

        return self.result


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


# ======================================================================
# Tests
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
        == (
            "Where is RepositoryScanner defined?"
        )
    )

    assert (
        turn.resolved_query
        == (
            "Where is RepositoryScanner defined?"
        )
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
        == (
            f"What does {CURRENT} call?"
        )
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


def test_unsupported_caller_reference_never_reaches_agent() -> None:

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
                    AgentIntent.DEFINITION
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

    with pytest.raises(
        UnresolvedFollowUpReferenceError
    ):

        orchestrator.run(
            "Where is the caller defined?"
        )

    assert runner.queries == []


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