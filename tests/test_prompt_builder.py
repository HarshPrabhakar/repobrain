from __future__ import annotations

from repobrain.answering import (
    GroundedPromptBuilder,
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


def make_result() -> AgentRunResult:

    item = EvidenceItem(
        evidence_id="ev1",
        result_key="symbol:hash",
        relative_path="app/scanner.py",
        symbol_id="hash",
        qualified_name="app.Scanner._hash",
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
        character_count=40,
        estimated_tokens=10,
    )

    bundle = EvidenceBundle(
        query="which function hashes files?",
        items=[
            item,
        ],
        total_characters=40,
        estimated_tokens=10,
        omitted_items=0,
        max_items=8,
        max_characters=24_000,
    )

    state = AgentState(
        query="which function hashes files?",
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
    )


def test_prompt_contains_question() -> None:

    prompt, _ = (
        GroundedPromptBuilder()
        .build(
            make_result()
        )
    )

    assert (
        "which function hashes files?"
        in prompt
    )


def test_prompt_contains_evidence_id() -> None:

    prompt, _ = (
        GroundedPromptBuilder()
        .build(
            make_result()
        )
    )

    assert "[E1]" in prompt


def test_prompt_contains_source() -> None:

    prompt, _ = (
        GroundedPromptBuilder()
        .build(
            make_result()
        )
    )

    assert (
        "def _hash(path)"
        in prompt
    )


def test_lookup_maps_evidence() -> None:

    _, lookup = (
        GroundedPromptBuilder()
        .build(
            make_result()
        )
    )

    assert "E1" in lookup

    assert (
        lookup["E1"].evidence_id
        == "ev1"
    )