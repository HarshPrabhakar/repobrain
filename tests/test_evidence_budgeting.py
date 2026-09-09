from __future__ import annotations

from repobrain.evidence import (
    EvidenceBudget,
    EvidenceBudgeter,
)

from repobrain.models.evidence import (
    EvidenceItem,
    EvidenceKind,
)


def make_item(
    evidence_id: str,
    text: str,
) -> EvidenceItem:

    return EvidenceItem(
        evidence_id=evidence_id,
        result_key=evidence_id,
        source_text=text,
        evidence_kind=(
            EvidenceKind.OTHER
        ),
        fused_score=0.1,
    )


def test_token_estimate() -> None:

    budgeter = (
        EvidenceBudgeter(
            EvidenceBudget(
                characters_per_token=4
            )
        )
    )

    assert (
        budgeter.estimate_tokens(
            "abcdefgh"
        )
        == 2
    )


def test_empty_text_zero_tokens() -> None:

    budgeter = (
        EvidenceBudgeter()
    )

    assert (
        budgeter.estimate_tokens("")
        == 0
    )


def test_item_limit() -> None:

    budgeter = (
        EvidenceBudgeter(
            EvidenceBudget(
                max_items=2,
                max_characters=1000,
            )
        )
    )

    selected, omitted = (
        budgeter.apply(
            [
                make_item("a", "aaa"),
                make_item("b", "bbb"),
                make_item("c", "ccc"),
            ]
        )
    )

    assert len(selected) == 2
    assert omitted == 1


def test_character_limit() -> None:

    budgeter = (
        EvidenceBudgeter(
            EvidenceBudget(
                max_items=5,
                max_characters=5,
            )
        )
    )

    selected, _ = (
        budgeter.apply(
            [
                make_item(
                    "a",
                    "abcdefghij",
                )
            ]
        )
    )

    assert len(selected) == 1

    assert (
        selected[0].source_text
        == "abcde"
    )


def test_budget_preserves_order() -> None:

    budgeter = (
        EvidenceBudgeter(
            EvidenceBudget(
                max_items=2,
                max_characters=100,
            )
        )
    )

    selected, _ = (
        budgeter.apply(
            [
                make_item("first", "a"),
                make_item("second", "b"),
                make_item("third", "c"),
            ]
        )
    )

    assert [
        item.evidence_id
        for item in selected
    ] == [
        "first",
        "second",
    ]


def test_invalid_budget_rejected() -> None:

    try:
        EvidenceBudget(
            max_items=0
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError."
    )