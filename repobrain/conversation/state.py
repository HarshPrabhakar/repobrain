from __future__ import annotations

from repobrain.models.agent import (
    AgentIntent,
)

from repobrain.models.conversation import (
    ConversationTurn,
    InvestigationState,
)


class InvestigationStateManager:
    """
    Deterministic state manager for one RepoBrain investigation.

    Responsibilities:

    - increment turn numbers
    - maintain current symbol focus
    - maintain previous symbol focus
    - store previous intent/query
    - retain validated grounding metadata
    - maintain bounded turn history
    - support transactional state restoration

    It does not perform:

    - retrieval
    - graph traversal
    - semantic search
    - LLM calls
    - follow-up resolution
    """

    DEFAULT_MAX_HISTORY = 20

    def __init__(
        self,
        *,
        initial_state: InvestigationState | None = None,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:

        if max_history < 1:
            raise ValueError(
                "max_history must be at least 1."
            )

        self._state = (
            initial_state
            if initial_state is not None
            else InvestigationState()
        )

        self._max_history = (
            max_history
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def state(
        self,
    ) -> InvestigationState:
        """
        Return the current immutable investigation state.
        """

        return self._state

    @property
    def max_history(
        self,
    ) -> int:
        return self._max_history

    # ==================================================================
    # Record completed agent turn
    # ==================================================================

    def record_turn(
        self,
        *,
        query: str,
        resolved_query: str | None = None,
        intent: AgentIntent | None = None,
        focused_symbol_id: str | None = None,
        focused_qualified_name: str | None = None,
        grounded: bool | None = None,
        citation_ids: tuple[str, ...] = (),
        invalid_citation_ids: tuple[str, ...] = (),
    ) -> InvestigationState:
        """
        Record a completed RepoBrain investigation turn.

        If a new focus is supplied, it becomes the current focus and
        the old current focus becomes the previous focus.

        If no focus is supplied, the current focus is retained.
        """

        normalized_query = (
            query.strip()
        )

        if not normalized_query:

            raise ValueError(
                "query cannot be empty."
            )

        normalized_resolved_query = (
            resolved_query.strip()
            if resolved_query is not None
            else None
        )

        if (
            normalized_resolved_query
            == ""
        ):

            normalized_resolved_query = None

        old_state = (
            self._state
        )

        next_turn_number = (
            old_state.turn_number
            + 1
        )

        has_new_focus = (
            focused_symbol_id is not None
            or focused_qualified_name is not None
        )

        # --------------------------------------------------------------
        # New deterministic focus
        # --------------------------------------------------------------

        if has_new_focus:

            next_current_symbol_id = (
                focused_symbol_id
            )

            next_current_qualified_name = (
                focused_qualified_name
            )

            next_previous_symbol_id = (
                old_state.current_symbol_id
            )

            next_previous_qualified_name = (
                old_state.current_qualified_name
            )

        # --------------------------------------------------------------
        # Retain existing focus
        # --------------------------------------------------------------

        else:

            next_current_symbol_id = (
                old_state.current_symbol_id
            )

            next_current_qualified_name = (
                old_state.current_qualified_name
            )

            next_previous_symbol_id = (
                old_state.previous_symbol_id
            )

            next_previous_qualified_name = (
                old_state.previous_qualified_name
            )

        turn = ConversationTurn(
            turn_number=(
                next_turn_number
            ),
            query=(
                normalized_query
            ),
            resolved_query=(
                normalized_resolved_query
            ),
            intent=(
                intent
            ),
            focused_symbol_id=(
                next_current_symbol_id
            ),
            focused_qualified_name=(
                next_current_qualified_name
            ),
            grounded=(
                grounded
            ),
            citation_ids=tuple(
                citation_ids
            ),
            invalid_citation_ids=tuple(
                invalid_citation_ids
            ),
        )

        next_history = (
            old_state.history
            + (
                turn,
            )
        )

        if (
            len(next_history)
            > self._max_history
        ):

            next_history = (
                next_history[
                    -self._max_history:
                ]
            )

        self._state = InvestigationState(
            conversation_id=(
                old_state.conversation_id
            ),
            turn_number=(
                next_turn_number
            ),
            current_symbol_id=(
                next_current_symbol_id
            ),
            current_qualified_name=(
                next_current_qualified_name
            ),
            previous_symbol_id=(
                next_previous_symbol_id
            ),
            previous_qualified_name=(
                next_previous_qualified_name
            ),
            previous_intent=(
                intent
            ),
            last_query=(
                normalized_query
            ),
            last_resolved_query=(
                normalized_resolved_query
            ),
            last_citation_ids=tuple(
                citation_ids
            ),
            last_invalid_citation_ids=tuple(
                invalid_citation_ids
            ),
            last_grounded=(
                grounded
            ),
            history=(
                next_history
            ),
        )

        return self._state

    # ==================================================================
    # Grounding update
    # ==================================================================

    def update_last_turn_grounding(
        self,
        *,
        grounded: bool,
        citation_ids: tuple[str, ...] = (),
        invalid_citation_ids: tuple[str, ...] = (),
    ) -> InvestigationState:
        """
        Attach Phase 8 grounding validation metadata to the most
        recently recorded investigation turn.

        This method does not trust raw model output.

        The caller must provide citation data that has already passed
        through GroundedAnswerGenerator validation.
        """

        old_state = (
            self._state
        )

        if not old_state.history:

            raise ValueError(
                "cannot update grounding without a recorded turn."
            )

        normalized_citations = (
            self._deduplicate(
                citation_ids
            )
        )

        normalized_invalid = (
            self._deduplicate(
                invalid_citation_ids
            )
        )

        old_turn = (
            old_state.history[-1]
        )

        updated_turn = ConversationTurn(
            turn_number=(
                old_turn.turn_number
            ),
            query=(
                old_turn.query
            ),
            resolved_query=(
                old_turn.resolved_query
            ),
            intent=(
                old_turn.intent
            ),
            focused_symbol_id=(
                old_turn.focused_symbol_id
            ),
            focused_qualified_name=(
                old_turn.focused_qualified_name
            ),
            grounded=(
                grounded
            ),
            citation_ids=(
                normalized_citations
            ),
            invalid_citation_ids=(
                normalized_invalid
            ),
        )

        updated_history = (
            old_state.history[:-1]
            + (
                updated_turn,
            )
        )

        self._state = InvestigationState(
            conversation_id=(
                old_state.conversation_id
            ),
            turn_number=(
                old_state.turn_number
            ),
            current_symbol_id=(
                old_state.current_symbol_id
            ),
            current_qualified_name=(
                old_state.current_qualified_name
            ),
            previous_symbol_id=(
                old_state.previous_symbol_id
            ),
            previous_qualified_name=(
                old_state.previous_qualified_name
            ),
            previous_intent=(
                old_state.previous_intent
            ),
            last_query=(
                old_state.last_query
            ),
            last_resolved_query=(
                old_state.last_resolved_query
            ),
            last_citation_ids=(
                normalized_citations
            ),
            last_invalid_citation_ids=(
                normalized_invalid
            ),
            last_grounded=(
                grounded
            ),
            history=(
                updated_history
            ),
        )

        return self._state

    # ==================================================================
    # Transaction support
    # ==================================================================

    def restore_state(
        self,
        state: InvestigationState,
    ) -> InvestigationState:
        """
        Restore a previously captured immutable InvestigationState.

        Phase 9.4 uses this when answer generation fails after the
        stateful agent has run.

        A failed answer should not leave behind a partially completed
        conversation turn.
        """

        self._state = (
            state
        )

        return self._state

    # ==================================================================
    # Context controls
    # ==================================================================

    def clear_context(
        self,
    ) -> InvestigationState:
        """
        Clear investigation context while preserving the current
        conversation/session identifier.
        """

        conversation_id = (
            self._state
            .conversation_id
        )

        self._state = (
            InvestigationState(
                conversation_id=(
                    conversation_id
                )
            )
        )

        return self._state

    def new_conversation(
        self,
    ) -> InvestigationState:
        """
        Start a new conversation with a new conversation ID.
        """

        self._state = (
            InvestigationState()
        )

        return self._state

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _deduplicate(
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """
        Deduplicate strings while preserving order.
        """

        seen: set[str] = set()

        result: list[str] = []

        for value in values:

            normalized = (
                value.strip()
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                normalized
            )

        return tuple(
            result
        )