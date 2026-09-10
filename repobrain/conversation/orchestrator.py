from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from repobrain.conversation.followup import (
    DeterministicFollowUpResolver,
)

from repobrain.conversation.state import (
    InvestigationStateManager,
)

from repobrain.models.agent import (
    AgentIntent,
    AgentRunResult,
)

from repobrain.models.conversation import (
    FollowUpReferenceKind,
    FollowUpResolution,
    InvestigationState,
    RelationshipFocusType,
)


class AgentRunner(Protocol):
    """
    Minimal interface required from RepoBrain's deterministic
    agent orchestrator.
    """

    def __call__(
        self,
        query: str,
    ) -> AgentRunResult:
        ...


@dataclass(
    frozen=True,
    slots=True,
)
class StatefulAgentTurn:
    """
    Result of one stateful RepoBrain investigation turn.
    """

    original_query: str

    resolved_query: str

    followup_resolution: FollowUpResolution

    agent_result: AgentRunResult

    state_before: InvestigationState

    state_after: InvestigationState


class StatefulRepoBrainOrchestrator:
    """
    Stateful deterministic wrapper around RepoBrain's existing
    agent pipeline.

    Phase 9.6 adds unambiguous relationship focus.

    A relationship target is remembered only when deterministic
    graph evidence produces exactly one result.
    """

    def __init__(
        self,
        *,
        agent_runner: AgentRunner,
        state_manager: InvestigationStateManager | None = None,
        followup_resolver: DeterministicFollowUpResolver | None = None,
    ) -> None:

        self._agent_runner = (
            agent_runner
        )

        self._state_manager = (
            state_manager
            if state_manager is not None
            else InvestigationStateManager()
        )

        self._followup_resolver = (
            followup_resolver
            if followup_resolver is not None
            else DeterministicFollowUpResolver()
        )

    # ==================================================================
    # Public properties
    # ==================================================================

    @property
    def state(
        self,
    ) -> InvestigationState:

        return (
            self._state_manager.state
        )

    @property
    def state_manager(
        self,
    ) -> InvestigationStateManager:

        return (
            self._state_manager
        )

    # ==================================================================
    # Main turn execution
    # ==================================================================

    def run(
        self,
        query: str,
    ) -> StatefulAgentTurn:
        """
        Run one deterministic stateful investigation turn.
        """

        state_before = (
            self._state_manager.state
        )

        followup_resolution = (
            self._followup_resolver.resolve(
                query,
                state_before,
            )
        )

        # --------------------------------------------------------------
        # Never allow unresolved coreference to reach retrieval/LLM.
        # --------------------------------------------------------------

        if (
            followup_resolution
            .unresolved_reference
        ):

            raise UnresolvedFollowUpReferenceError(
                query=(
                    followup_resolution
                    .original_query
                ),
                reference_text=(
                    followup_resolution
                    .reference_text
                ),
            )

        resolved_query = (
            followup_resolution
            .resolved_query
        )

        agent_result = (
            self._agent_runner(
                resolved_query
            )
        )

        # ==================================================================
        # Primary symbol focus
        # ==================================================================

        (
            focus_symbol_id,
            focus_qualified_name,
        ) = self._derive_focus(
            agent_result=(
                agent_result
            ),
            followup_resolution=(
                followup_resolution
            ),
            state_before=(
                state_before
            ),
        )

        # ==================================================================
        # Relationship focus
        # ==================================================================

        (
            relationship_focus_type,
            relationship_symbol_id,
            relationship_qualified_name,
            clear_relationship_focus,
        ) = self._derive_relationship_focus(
            agent_result=(
                agent_result
            ),
            followup_resolution=(
                followup_resolution
            ),
            state_before=(
                state_before
            ),
        )

        state_after = (
            self._state_manager
            .record_turn(
                query=(
                    followup_resolution
                    .original_query
                ),
                resolved_query=(
                    resolved_query
                ),
                intent=(
                    agent_result
                    .state
                    .intent
                ),
                focused_symbol_id=(
                    focus_symbol_id
                ),
                focused_qualified_name=(
                    focus_qualified_name
                ),
                relationship_focus_type=(
                    relationship_focus_type
                ),
                relationship_symbol_id=(
                    relationship_symbol_id
                ),
                relationship_qualified_name=(
                    relationship_qualified_name
                ),
                clear_relationship_focus=(
                    clear_relationship_focus
                ),
            )
        )

        return StatefulAgentTurn(
            original_query=(
                followup_resolution
                .original_query
            ),
            resolved_query=(
                resolved_query
            ),
            followup_resolution=(
                followup_resolution
            ),
            agent_result=(
                agent_result
            ),
            state_before=(
                state_before
            ),
            state_after=(
                state_after
            ),
        )

    # ==================================================================
    # State controls
    # ==================================================================

    def clear(
        self,
    ) -> InvestigationState:
        """
        Clear investigation context while retaining conversation ID.
        """

        return (
            self._state_manager
            .clear_context()
        )

    def new_conversation(
        self,
    ) -> InvestigationState:
        """
        Start a completely new investigation session.
        """

        return (
            self._state_manager
            .new_conversation()
        )

    # ==================================================================
    # Primary focus derivation
    # ==================================================================

    @staticmethod
    def _derive_focus(
        *,
        agent_result: AgentRunResult,
        followup_resolution: FollowUpResolution,
        state_before: InvestigationState,
    ) -> tuple[
        str | None,
        str | None,
    ]:
        """
        Determine primary conversational symbol focus.

        Priority:

        1. Agent deterministic resolved symbol.
        2. Follow-up deterministic resolved symbol.
        3. Existing focus retained by state manager.
        """

        resolved_symbol_id = (
            getattr(
                agent_result,
                "resolved_symbol_id",
                None,
            )
        )

        resolved_qualified_name = (
            getattr(
                agent_result,
                "resolved_qualified_name",
                None,
            )
        )

        if (
            resolved_symbol_id is not None
            or resolved_qualified_name is not None
        ):

            return (
                resolved_symbol_id,
                resolved_qualified_name,
            )

        if (
            followup_resolution
            .used_context
        ):

            return (
                followup_resolution
                .resolved_symbol_id,
                followup_resolution
                .resolved_qualified_name,
            )

        return (
            None,
            None,
        )

    # ==================================================================
    # Relationship-focus derivation
    # ==================================================================

    @staticmethod
    def _derive_relationship_focus(
        *,
        agent_result: AgentRunResult,
        followup_resolution: FollowUpResolution,
        state_before: InvestigationState,
    ) -> tuple[
        RelationshipFocusType | None,
        str | None,
        str | None,
        bool,
    ]:
        """
        Determine relationship focus using deterministic graph facts.

        Rules:

        CALLERS + exactly one fact:
            relationship focus = fact.source

        CALLEES + exactly one fact:
            relationship focus = fact.target

        CALLERS/CALLEES + zero or multiple facts:
            clear relationship focus

        A follow-up that explicitly references the current caller/callee
        preserves relationship focus so chains such as:

            who calls X?
            where is the caller defined?
            show me that caller's implementation

        remain deterministic.

        Ordinary unrelated turns clear relationship focus to prevent
        stale caller/callee references from leaking into later context.
        """

        intent = (
            getattr(
                agent_result.state,
                "intent",
                None,
            )
        )

        graph_facts = list(
            getattr(
                agent_result,
                "graph_facts",
                (),
            )
            or ()
        )

        # --------------------------------------------------------------
        # Structural graph result
        # --------------------------------------------------------------

        if (
            intent
            == AgentIntent.CALLERS
        ):

            if len(graph_facts) == 1:

                fact = (
                    graph_facts[0]
                )

                return (
                    RelationshipFocusType.CALLER,
                    fact.source_symbol_id,
                    fact.source_qualified_name,
                    False,
                )

            return (
                None,
                None,
                None,
                True,
            )

        if (
            intent
            == AgentIntent.CALLEES
        ):

            if len(graph_facts) == 1:

                fact = (
                    graph_facts[0]
                )

                return (
                    RelationshipFocusType.CALLEE,
                    fact.target_symbol_id,
                    fact.target_qualified_name,
                    False,
                )

            return (
                None,
                None,
                None,
                True,
            )

        # --------------------------------------------------------------
        # Explicit relation follow-up preserves relationship focus.
        # --------------------------------------------------------------

        if (
            followup_resolution.reference_kind
            in {
                FollowUpReferenceKind.CALLER_FOCUS,
                FollowUpReferenceKind.CALLEE_FOCUS,
            }
        ):

            return (
                state_before.relationship_focus_type,
                state_before.relationship_symbol_id,
                state_before.relationship_qualified_name,
                False,
            )

        # --------------------------------------------------------------
        # Any other completed non-graph turn invalidates the old
        # relation target so stale "the caller" references cannot
        # accidentally cross investigation subjects.
        # --------------------------------------------------------------

        return (
            None,
            None,
            None,
            True,
        )


class UnresolvedFollowUpReferenceError(
    ValueError
):
    """
    Raised when a conversational reference is detected but cannot
    be resolved deterministically from InvestigationState.
    """

    def __init__(
        self,
        *,
        query: str,
        reference_text: str | None,
    ) -> None:

        self.query = (
            query
        )

        self.reference_text = (
            reference_text
        )

        message = (
            "Cannot resolve follow-up reference"
        )

        if reference_text:

            message += (
                f" {reference_text!r}"
            )

        message += (
            " from the current investigation state."
        )

        super().__init__(
            message
        )