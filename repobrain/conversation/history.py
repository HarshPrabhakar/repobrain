from __future__ import annotations

from repobrain.models.conversation import (
    ConversationTurn,
    InvestigationState,
)


class InvestigationHistory:
    """
    Read-only helpers for inspecting bounded investigation history.

    This class does not mutate state.
    """

    @staticmethod
    def latest(
        state: InvestigationState,
    ) -> ConversationTurn | None:
        """
        Return the newest investigation turn.
        """

        if not state.history:
            return None

        return state.history[-1]

    @staticmethod
    def previous(
        state: InvestigationState,
    ) -> ConversationTurn | None:
        """
        Return the turn immediately before the latest one.
        """

        if len(state.history) < 2:
            return None

        return state.history[-2]

    @staticmethod
    def recent(
        state: InvestigationState,
        *,
        limit: int = 5,
    ) -> tuple[
        ConversationTurn,
        ...,
    ]:
        """
        Return up to the most recent N turns.
        """

        if limit < 1:
            raise ValueError(
                "limit must be at least 1."
            )

        return (
            state.history[
                -limit:
            ]
        )

    @staticmethod
    def find_by_qualified_name(
        state: InvestigationState,
        qualified_name: str,
    ) -> tuple[
        ConversationTurn,
        ...,
    ]:
        """
        Return turns whose focus matches the supplied qualified name.
        """

        normalized = (
            qualified_name.strip()
        )

        if not normalized:
            return ()

        return tuple(
            turn
            for turn in state.history
            if (
                turn.focused_qualified_name
                == normalized
            )
        )