from __future__ import annotations

from repobrain.answering import (
    GroundedAnswerGenerator,
    GroundedPromptBuilder,
)


def test_system_prompt_requires_citations() -> None:
    instructions = (
        GroundedPromptBuilder
        .SYSTEM_INSTRUCTIONS
    )

    assert (
        "MUST contain"
        in instructions
    )

    assert (
        "[E#]"
        in instructions
    )

    assert (
        "[G#]"
        in instructions
    )

    assert (
        "INVALID"
        in instructions
    )


def test_system_prompt_requests_concise_answers() -> None:
    instructions = (
        GroundedPromptBuilder
        .SYSTEM_INSTRUCTIONS
    )

    assert (
        "Keep the answer concise"
        in instructions
    )


def test_repair_prompt_marks_previous_answer_rejected() -> None:
    prompt = (
        GroundedAnswerGenerator
        ._build_citation_repair_prompt(
            original_prompt=(
                "SOURCE EVIDENCE\n[E1]"
            ),
            original_answer=(
                "Uncited answer."
            ),
        )
    )

    assert (
        "GROUNDING VALIDATION FAILURE"
        in prompt
    )

    assert (
        "RepoBrain rejected your previous answer"
        in prompt
    )

    assert (
        "no valid citations"
        in prompt
    )

    assert (
        "citation IDs that were not supplied"
        in prompt
    )

    assert (
        "Every repository-specific factual claim MUST end with"
        in prompt
    )

    assert (
        "Citation tokens must be copied EXACTLY"
        in prompt
    )

    assert (
        "at most 4 short paragraphs or bullets"
        in prompt
    )

    assert (
        "NEVER invent another citation ID"
        in prompt
    )

    assert (
        "PREVIOUS REJECTED ANSWER"
        in prompt
    )