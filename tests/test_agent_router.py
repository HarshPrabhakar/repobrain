from __future__ import annotations

from repobrain.agent import (
    DeterministicAgentRouter,
)

from repobrain.models.agent import (
    AgentIntent,
)


def test_detects_callers_intent() -> None:

    assert (
        DeterministicAgentRouter
        .classify(
            "who calls _calculate_sha256?"
        )
        == AgentIntent.CALLERS
    )


def test_detects_callees_intent() -> None:

    assert (
        DeterministicAgentRouter
        .classify(
            "what does RepositoryScanner.scan call?"
        )
        == AgentIntent.CALLEES
    )


def test_detects_documentation_intent() -> None:

    assert (
        DeterministicAgentRouter
        .classify(
            "what is this project trying to build?"
        )
        == AgentIntent.DOCUMENTATION
    )


def test_detects_implementation_intent() -> None:

    assert (
        DeterministicAgentRouter
        .classify(
            (
                "which function computes a "
                "stable digest?"
            )
        )
        == AgentIntent.IMPLEMENTATION
    )


def test_extracts_private_symbol() -> None:

    assert (
        DeterministicAgentRouter
        .extract_symbol_reference(
            "who calls _calculate_sha256?"
        )
        == "_calculate_sha256"
    )


def test_extracts_dotted_symbol() -> None:

    assert (
        DeterministicAgentRouter
        .extract_symbol_reference(
            (
                "what does "
                "RepositoryScanner.scan call?"
            )
        )
        == "RepositoryScanner.scan"
    )