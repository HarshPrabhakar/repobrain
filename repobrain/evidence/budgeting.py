from __future__ import annotations

from dataclasses import dataclass

from repobrain.models.evidence import (
    EvidenceItem,
)


DEFAULT_MAX_EVIDENCE_ITEMS = 8

DEFAULT_MAX_CHARACTERS = 24_000

DEFAULT_CHARACTERS_PER_TOKEN = 4


@dataclass(frozen=True)
class EvidenceBudget:
    """
    Hard deterministic context budget.

    We deliberately use a character-based estimate instead of
    introducing another tokenizer dependency during Phase 6.
    """

    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS

    max_characters: int = DEFAULT_MAX_CHARACTERS

    characters_per_token: int = (
        DEFAULT_CHARACTERS_PER_TOKEN
    )

    def __post_init__(
        self,
    ) -> None:

        if self.max_items <= 0:
            raise ValueError(
                "max_items must be > 0."
            )

        if self.max_characters <= 0:
            raise ValueError(
                "max_characters must be > 0."
            )

        if self.characters_per_token <= 0:
            raise ValueError(
                "characters_per_token must be > 0."
            )


class EvidenceBudgeter:
    """
    Applies RepoBrain's deterministic evidence budget.
    """

    def __init__(
        self,
        budget: EvidenceBudget | None = None,
    ) -> None:

        self.budget = (
            budget
            or EvidenceBudget()
        )

    def estimate_tokens(
        self,
        text: str,
    ) -> int:
        """
        Cheap deterministic token estimate.

        Intentionally approximate.
        """

        if not text:
            return 0

        length = len(text)

        divisor = (
            self.budget.characters_per_token
        )

        return max(
            1,
            (
                length
                + divisor
                - 1
            )
            // divisor,
        )

    def apply(
        self,
        items: list[EvidenceItem],
    ) -> tuple[
        list[EvidenceItem],
        int,
    ]:
        """
        Select items while respecting both item count and
        character count.

        Ranking order is preserved.
        """

        selected: list[
            EvidenceItem
        ] = []

        used_characters = 0

        for item in items:

            if (
                len(selected)
                >= self.budget.max_items
            ):
                break

            source_length = len(
                item.source_text
            )

            remaining = (
                self.budget.max_characters
                - used_characters
            )

            if remaining <= 0:
                break

            # -------------------------------------------------
            # If the first/high-ranked item is larger than the
            # remaining budget, retain a bounded excerpt rather
            # than dropping the result completely.
            # -------------------------------------------------

            if source_length > remaining:

                bounded_text = (
                    item.source_text[
                        :remaining
                    ]
                )

                if not bounded_text:
                    break

                item = item.model_copy(
                    update={
                        "source_text": (
                            bounded_text
                        ),
                        "character_count": (
                            len(
                                bounded_text
                            )
                        ),
                        "estimated_tokens": (
                            self.estimate_tokens(
                                bounded_text
                            )
                        ),
                    }
                )

                selected.append(
                    item
                )

                used_characters += len(
                    bounded_text
                )

                break

            item = item.model_copy(
                update={
                    "character_count": (
                        source_length
                    ),
                    "estimated_tokens": (
                        self.estimate_tokens(
                            item.source_text
                        )
                    ),
                }
            )

            selected.append(
                item
            )

            used_characters += (
                source_length
            )

        omitted = max(
            0,
            len(items)
            - len(selected),
        )

        return (
            selected,
            omitted,
        )