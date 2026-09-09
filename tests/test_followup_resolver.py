from __future__ import annotations

import pytest

from repobrain.conversation import (
    DeterministicFollowUpResolver,
)

from repobrain.models.conversation import (
    FollowUpReferenceKind,
    InvestigationState,
)


CURRENT = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._calculate_sha256"
)

PREVIOUS = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._build_file_metadata"
)


def make_state() -> InvestigationState:

    return InvestigationState(
        turn_number=2,
        current_symbol_id=(
            "symbol-current"
        ),
        current_qualified_name=(
            CURRENT
        ),
        previous_symbol_id=(
            "symbol-previous"
        ),
        previous_qualified_name=(
            PREVIOUS
        ),
    )


def test_query_without_reference_passes_through() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Where is RepositoryScanner defined?",
        make_state(),
    )

    assert (
        result.original_query
        == (
            "Where is RepositoryScanner defined?"
        )
    )

    assert (
        result.resolved_query
        == (
            "Where is RepositoryScanner defined?"
        )
    )

    assert (
        result.used_context
        is False
    )

    assert (
        result.reference_kind
        == FollowUpReferenceKind.NONE
    )

    assert (
        result.unresolved_reference
        is False
    )


def test_it_resolves_to_current_focus() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "What does it call?",
        make_state(),
    )

    assert (
        result.resolved_query
        == f"What does {CURRENT} call?"
    )

    assert (
        result.used_context
        is True
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .CURRENT_FOCUS
        )
    )

    assert (
        result.resolved_symbol_id
        == "symbol-current"
    )

    assert (
        result.resolved_qualified_name
        == CURRENT
    )


def test_that_resolves_to_current_focus() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Where is that defined?",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Where is {CURRENT} defined?"
        )
    )


def test_this_function_resolves_as_single_reference() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Show me this function implementation.",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Show me {CURRENT} implementation."
        )
    )

    assert (
        result.reference_text
        == "this function"
    )


def test_that_method_resolves_as_single_reference() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Who calls that method?",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Who calls {CURRENT}?"
        )
    )


def test_same_function_resolves_to_current_focus() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Explain the same function.",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Explain the {CURRENT}."
        )
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .CURRENT_FOCUS
        )
    )


def test_previous_function_resolves_to_previous_focus() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Show the previous function.",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Show {PREVIOUS}."
        )
    )

    assert (
        result.reference_text
        == "the previous function"
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .PREVIOUS_FOCUS
        )
    )

    assert (
        result.resolved_symbol_id
        == "symbol-previous"
    )

    assert (
        result.resolved_qualified_name
        == PREVIOUS
    )

    assert (
        result.used_context
        is True
    )

    assert (
        result.unresolved_reference
        is False
    )


def test_current_reference_without_focus_is_unresolved() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    state = InvestigationState()

    result = resolver.resolve(
        "What does it call?",
        state,
    )

    assert (
        result.resolved_query
        == "What does it call?"
    )

    assert (
        result.used_context
        is False
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .CURRENT_FOCUS
        )
    )

    assert (
        result.unresolved_reference
        is True
    )


def test_previous_reference_without_previous_focus_is_unresolved() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    state = InvestigationState(
        current_symbol_id=(
            "current"
        ),
        current_qualified_name=(
            CURRENT
        ),
    )

    result = resolver.resolve(
        "Where is the previous function?",
        state,
    )

    assert (
        result.resolved_query
        == (
            "Where is the previous function?"
        )
    )

    assert (
        result.unresolved_reference
        is True
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .PREVIOUS_FOCUS
        )
    )


def test_caller_reference_is_intentionally_unresolved() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Where is the caller defined?",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            "Where is the caller defined?"
        )
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .UNSUPPORTED_RELATION
        )
    )

    assert (
        result.unresolved_reference
        is True
    )

    assert (
        result.used_context
        is False
    )


def test_callee_reference_is_intentionally_unresolved() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Show me that callee.",
        make_state(),
    )

    assert (
        result.reference_kind
        == (
            FollowUpReferenceKind
            .UNSUPPORTED_RELATION
        )
    )

    assert (
        result.unresolved_reference
        is True
    )


def test_resolution_is_case_insensitive() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "What does IT call?",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"What does {CURRENT} call?"
        )
    )


def test_whitespace_is_normalized() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "  What   does   it   call?  ",
        make_state(),
    )

    assert (
        result.original_query
        == "What does it call?"
    )

    assert (
        result.resolved_query
        == (
            f"What does {CURRENT} call?"
        )
    )


def test_empty_query_is_rejected() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):

        resolver.resolve(
            "   ",
            make_state(),
        )

def test_this_project_is_not_treated_as_followup() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    state = InvestigationState()

    result = resolver.resolve(
        "What is this project?",
        state,
    )

    assert (
        result.resolved_query
        == "What is this project?"
    )

    assert (
        result.used_context
        is False
    )

    assert (
        result.reference_kind
        == FollowUpReferenceKind.NONE
    )

    assert (
        result.unresolved_reference
        is False
    )


def test_this_repository_is_not_treated_as_followup() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "How does this repository work?",
        make_state(),
    )

    assert (
        result.resolved_query
        == "How does this repository work?"
    )

    assert (
        result.used_context
        is False
    )

    assert (
        result.reference_kind
        == FollowUpReferenceKind.NONE
    )


def test_bare_that_before_defined_is_followup() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Where is that defined?",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Where is {CURRENT} defined?"
        )
    )

    assert (
        result.used_context
        is True
    )


def test_bare_this_at_sentence_end_is_followup() -> None:

    resolver = (
        DeterministicFollowUpResolver()
    )

    result = resolver.resolve(
        "Explain this.",
        make_state(),
    )

    assert (
        result.resolved_query
        == (
            f"Explain {CURRENT}."
        )
    )

    assert (
        result.used_context
        is True
    )