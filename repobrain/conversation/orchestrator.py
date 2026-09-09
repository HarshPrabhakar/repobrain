from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Protocol,
)

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
    FollowUpResolution,
    InvestigationState,
)


class AgentRunner(Protocol):
    """
    Minimal interface required from the existing RepoBrain
    deterministic agent orchestrator.

    The Phase 9 wrapper depends only on this callable contract
    rather than on the concrete Phase 7 implementation.
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
    Phase 9 stateful wrapper around RepoBrain's existing
    deterministic agent pipeline.

    Pipeline:

        user query
            ↓
        deterministic follow-up resolution
            ↓
        existing RepoBrain agent
            ↓
        derive deterministic focus
            ↓
        update InvestigationState

    This layer intentionally does NOT:

    - change hybrid retrieval
    - change graph traversal
    - change evidence assembly
    - call an LLM directly
    - mutate the repository
    - replace the Phase 7 orchestrator

    It only adds bounded multi-turn investigation state.
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
        """
        Return current investigation state.
        """

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
        Run one stateful RepoBrain investigation turn.

        The original user query is never sent directly to the
        existing agent when a deterministic conversational reference
        can be resolved.

        Example:

            "What does it call?"

        may become:

            "What does
            repobrain.ingestion.scanner.
            RepositoryScanner._calculate_sha256
            call?"
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
        # Do not silently send unresolved conversational references
        # into the existing agent.
        #
        # That would allow retrieval/LLM layers to guess what "it",
        # "that caller", etc. means.
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

        # --------------------------------------------------------------
        # Determine what repository symbol this turn should leave as
        # the current conversational focus.
        # --------------------------------------------------------------

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
        Clear investigation context while keeping the same
        conversation ID.
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
    # Focus derivation
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
        Determine the symbol that should become the conversational
        focus after this turn.

        Priority:

        1. Agent's deterministic resolved symbol.
        2. Follow-up symbol that was already resolved from state.
        3. Existing current focus.

        This avoids using retrieval ranking or LLM output to invent
        conversational focus.
        """

        # --------------------------------------------------------------
        # Existing Phase 7 result already resolved a symbol.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Follow-up itself resolved to a known symbol.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # No new deterministic focus.
        #
        # Returning None/None causes InvestigationStateManager to retain
        # the existing current focus.
        # --------------------------------------------------------------

        return (
            None,
            None,
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

        self.query = query

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