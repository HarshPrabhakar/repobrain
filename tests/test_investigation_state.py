from __future__ import annotations

import pytest

from repobrain.conversation import (
    InvestigationHistory,
    InvestigationStateManager,
)

from repobrain.models.agent import (
    AgentIntent,
)

from repobrain.models.conversation import (
    InvestigationState,
)


def test_new_investigation_state_is_empty() -> None:

    state = InvestigationState()

    assert state.turn_number == 0

    assert (
        state.current_symbol_id
        is None
    )

    assert (
        state.current_qualified_name
        is None
    )

    assert (
        state.previous_symbol_id
        is None
    )

    assert (
        state.previous_qualified_name
        is None
    )

    assert (
        state.previous_intent
        is None
    )

    assert state.history == ()

    assert state.has_focus is False

    assert state.current_turn is None


def test_record_turn_sets_initial_focus() -> None:

    manager = (
        InvestigationStateManager()
    )

    state = manager.record_turn(
        query=(
            "Who calls _calculate_sha256?"
        ),
        intent=(
            AgentIntent.CALLERS
        ),
        focused_symbol_id=(
            "symbol-sha256"
        ),
        focused_qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner._calculate_sha256"
        ),
        grounded=True,
        citation_ids=(
            "G1",
        ),
    )

    assert state.turn_number == 1

    assert (
        state.current_symbol_id
        == "symbol-sha256"
    )

    assert (
        state.current_qualified_name
        == (
            "repobrain.ingestion.scanner."
            "RepositoryScanner._calculate_sha256"
        )
    )

    assert (
        state.previous_symbol_id
        is None
    )

    assert (
        state.previous_intent
        == AgentIntent.CALLERS
    )

    assert (
        state.last_grounded
        is True
    )

    assert (
        state.last_citation_ids
        == (
            "G1",
        )
    )

    assert len(
        state.history
    ) == 1


def test_new_focus_moves_old_focus_to_previous() -> None:

    manager = (
        InvestigationStateManager()
    )

    manager.record_turn(
        query=(
            "Who calls _calculate_sha256?"
        ),
        intent=(
            AgentIntent.CALLERS
        ),
        focused_symbol_id=(
            "symbol-sha256"
        ),
        focused_qualified_name=(
            "pkg.Scanner._calculate_sha256"
        ),
    )

    state = manager.record_turn(
        query=(
            "Where is that caller defined?"
        ),
        resolved_query=(
            "Where is pkg.Scanner."
            "_build_file_metadata defined?"
        ),
        intent=(
            AgentIntent.DEFINITION
        ),
        focused_symbol_id=(
            "symbol-build-metadata"
        ),
        focused_qualified_name=(
            "pkg.Scanner._build_file_metadata"
        ),
    )

    assert (
        state.current_symbol_id
        == "symbol-build-metadata"
    )

    assert (
        state.current_qualified_name
        == (
            "pkg.Scanner._build_file_metadata"
        )
    )

    assert (
        state.previous_symbol_id
        == "symbol-sha256"
    )

    assert (
        state.previous_qualified_name
        == (
            "pkg.Scanner._calculate_sha256"
        )
    )


def test_turn_without_new_focus_retains_current_focus() -> None:

    manager = (
        InvestigationStateManager()
    )

    manager.record_turn(
        query="Find function.",
        focused_symbol_id="symbol-1",
        focused_qualified_name=(
            "pkg.module.function"
        ),
    )

    state = manager.record_turn(
        query=(
            "Explain its implementation."
        ),
        resolved_query=(
            "Explain pkg.module.function "
            "implementation."
        ),
        intent=(
            AgentIntent.IMPLEMENTATION
        ),
    )

    assert (
        state.current_symbol_id
        == "symbol-1"
    )

    assert (
        state.current_qualified_name
        == "pkg.module.function"
    )

    assert state.turn_number == 2


def test_history_is_bounded() -> None:

    manager = (
        InvestigationStateManager(
            max_history=3,
        )
    )

    for index in range(
        1,
        6,
    ):

        manager.record_turn(
            query=(
                f"Question {index}"
            )
        )

    state = manager.state

    assert state.turn_number == 5

    assert len(
        state.history
    ) == 3

    assert (
        state.history[0].query
        == "Question 3"
    )

    assert (
        state.history[-1].query
        == "Question 5"
    )


def test_clear_context_preserves_conversation_id() -> None:

    manager = (
        InvestigationStateManager()
    )

    manager.record_turn(
        query="Find symbol.",
        focused_symbol_id="symbol-1",
        focused_qualified_name=(
            "pkg.symbol"
        ),
    )

    old_id = (
        manager.state
        .conversation_id
    )

    state = (
        manager.clear_context()
    )

    assert (
        state.conversation_id
        == old_id
    )

    assert state.turn_number == 0

    assert state.history == ()

    assert state.has_focus is False


def test_new_conversation_changes_conversation_id() -> None:

    manager = (
        InvestigationStateManager()
    )

    old_id = (
        manager.state
        .conversation_id
    )

    state = (
        manager.new_conversation()
    )

    assert (
        state.conversation_id
        != old_id
    )

    assert state.turn_number == 0

    assert state.history == ()


def test_history_helpers() -> None:

    manager = (
        InvestigationStateManager()
    )

    manager.record_turn(
        query="Question one",
        focused_qualified_name=(
            "pkg.alpha"
        ),
    )

    manager.record_turn(
        query="Question two",
        focused_qualified_name=(
            "pkg.beta"
        ),
    )

    manager.record_turn(
        query="Question three",
        focused_qualified_name=(
            "pkg.alpha"
        ),
    )

    state = manager.state

    latest = (
        InvestigationHistory.latest(
            state
        )
    )

    previous = (
        InvestigationHistory.previous(
            state
        )
    )

    recent = (
        InvestigationHistory.recent(
            state,
            limit=2,
        )
    )

    matches = (
        InvestigationHistory
        .find_by_qualified_name(
            state,
            "pkg.alpha",
        )
    )

    assert latest is not None

    assert (
        latest.query
        == "Question three"
    )

    assert previous is not None

    assert (
        previous.query
        == "Question two"
    )

    assert len(
        recent
    ) == 2

    assert [
        turn.query
        for turn in recent
    ] == [
        "Question two",
        "Question three",
    ]

    assert len(
        matches
    ) == 2


def test_empty_query_is_rejected() -> None:

    manager = (
        InvestigationStateManager()
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):

        manager.record_turn(
            query="   ",
        )


def test_invalid_history_limit_is_rejected() -> None:

    with pytest.raises(
        ValueError,
        match="max_history must be at least 1",
    ):

        InvestigationStateManager(
            max_history=0,
        )