from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from repobrain.conversation.orchestrator import (
    StatefulAgentTurn,
    StatefulRepoBrainOrchestrator,
)

from repobrain.models.agent import (
    AgentRunResult,
)

from repobrain.models.conversation import (
    InvestigationState,
)


class GroundedAnswerProvider(
    Protocol
):
    """
    Minimal contract required from RepoBrain's existing
    GroundedAnswerGenerator.

    The real Phase 8 generator already exposes generate(result).
    """

    def generate(
        self,
        result: AgentRunResult,
    ) -> object:
        ...


@dataclass(
    frozen=True,
    slots=True,
)
class StatefulGroundedTurn:
    """
    Complete result of one stateful grounded RepoBrain turn.

    Contains:

    - stateful agent investigation
    - grounded answer
    - final state after grounding validation metadata is stored
    """

    agent_turn: StatefulAgentTurn

    answer: object

    state_after: InvestigationState

    @property
    def original_query(
        self,
    ) -> str:
        return (
            self.agent_turn
            .original_query
        )

    @property
    def resolved_query(
        self,
    ) -> str:
        return (
            self.agent_turn
            .resolved_query
        )


class StatefulGroundedAnsweringOrchestrator:
    """
    Phase 9.4 orchestration layer.

    Pipeline:

        User query
            ↓
        StatefulRepoBrainOrchestrator
            ↓
        AgentRunResult
            ↓
        GroundedAnswerGenerator
            ↓
        validated citation metadata
            ↓
        InvestigationStateManager

    The grounding result is taken from the existing Phase 8 answer
    generator.

    This class does not independently validate LLM citations.
    """

    def __init__(
        self,
        *,
        stateful_agent: StatefulRepoBrainOrchestrator,
        answer_generator: GroundedAnswerProvider,
    ) -> None:

        self._stateful_agent = (
            stateful_agent
        )

        self._answer_generator = (
            answer_generator
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def state(
        self,
    ) -> InvestigationState:
        return (
            self._stateful_agent
            .state
        )

    @property
    def stateful_agent(
        self,
    ) -> StatefulRepoBrainOrchestrator:
        return (
            self._stateful_agent
        )

    # ==================================================================
    # Main API
    # ==================================================================

    def run(
        self,
        query: str,
    ) -> StatefulGroundedTurn:
        """
        Execute one complete multi-turn grounded investigation.

        State semantics are transactional:

        - unresolved follow-up:
            no state change

        - agent failure:
            no completed turn

        - answer-generation failure:
            restore state from before the turn

        - successful answer:
            attach grounding metadata to the recorded turn
        """

        state_before = (
            self._stateful_agent
            .state
        )

        # --------------------------------------------------------------
        # Phase 9.3
        # --------------------------------------------------------------

        agent_turn = (
            self._stateful_agent
            .run(
                query
            )
        )

        # --------------------------------------------------------------
        # Phase 8
        # --------------------------------------------------------------

        try:

            answer = (
                self._answer_generator
                .generate(
                    agent_turn.agent_result
                )
            )

        except Exception:

            # ----------------------------------------------------------
            # The stateful agent already recorded the investigation.
            #
            # If answer generation fails, restore the previous immutable
            # state so we do not retain a half-completed conversation
            # turn.
            # ----------------------------------------------------------

            self._stateful_agent \
                .state_manager \
                .restore_state(
                    state_before
                )

            raise

        # --------------------------------------------------------------
        # Extract ONLY validation metadata already produced by Phase 8.
        # --------------------------------------------------------------

        grounded = (
            self._extract_grounded(
                answer
            )
        )

        citation_ids = (
            self._extract_valid_citation_ids(
                answer
            )
        )

        invalid_citation_ids = (
            self._extract_invalid_citation_ids(
                answer
            )
        )

        # --------------------------------------------------------------
        # Attach validated grounding metadata to the same turn.
        # --------------------------------------------------------------

        state_after = (
            self._stateful_agent
            .state_manager
            .update_last_turn_grounding(
                grounded=(
                    grounded
                ),
                citation_ids=(
                    citation_ids
                ),
                invalid_citation_ids=(
                    invalid_citation_ids
                ),
            )
        )

        return StatefulGroundedTurn(
            agent_turn=(
                agent_turn
            ),
            answer=(
                answer
            ),
            state_after=(
                state_after
            ),
        )

    # ==================================================================
    # Context controls
    # ==================================================================

    def clear(
        self,
    ) -> InvestigationState:
        return (
            self._stateful_agent
            .clear()
        )

    def new_conversation(
        self,
    ) -> InvestigationState:
        return (
            self._stateful_agent
            .new_conversation()
        )

    # ==================================================================
    # Grounding metadata extraction
    # ==================================================================

    @staticmethod
    def _extract_grounded(
        answer: object,
    ) -> bool:
        """
        Read the validated grounded flag from the Phase 8 result.
        """

        return bool(
            getattr(
                answer,
                "grounded",
                False,
            )
        )

    @classmethod
    def _extract_valid_citation_ids(
        cls,
        answer: object,
    ) -> tuple[str, ...]:
        """
        Extract validated citation IDs from GroundedAnswer.

        Current Phase 8 models expose citation objects.

        A small compatibility fallback is retained for fake providers
        and future model refactors.
        """

        citations = (
            getattr(
                answer,
                "citations",
                (),
            )
            or ()
        )

        result: list[str] = []

        seen: set[str] = set()

        for citation in citations:

            citation_id = (
                cls._citation_identifier(
                    citation
                )
            )

            if citation_id is None:
                continue

            if citation_id in seen:
                continue

            seen.add(
                citation_id
            )

            result.append(
                citation_id
            )

        return tuple(
            result
        )

    @classmethod
    def _extract_invalid_citation_ids(
        cls,
        answer: object,
    ) -> tuple[str, ...]:
        """
        Extract invalid citation IDs already identified by Phase 8.
        """

        values = (
            getattr(
                answer,
                "invalid_citations",
                None,
            )
        )

        if values is None:

            values = (
                getattr(
                    answer,
                    "invalid_citation_ids",
                    (),
                )
            )

        if values is None:
            return ()

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            if isinstance(
                value,
                str,
            ):

                citation_id = (
                    value.strip()
                )

            else:

                citation_id = (
                    cls._citation_identifier(
                        value
                    )
                    or ""
                )

            if not citation_id:
                continue

            if citation_id in seen:
                continue

            seen.add(
                citation_id
            )

            result.append(
                citation_id
            )

        return tuple(
            result
        )

    @staticmethod
    def _citation_identifier(
        citation: object,
    ) -> str | None:
        """
        Extract the citation namespace identifier.

        Primary expected field:

            citation_id

        Compatibility fallback:

            id
        """

        if isinstance(
            citation,
            str,
        ):

            normalized = (
                citation.strip()
            )

            return (
                normalized
                if normalized
                else None
            )

        value = (
            getattr(
                citation,
                "citation_id",
                None,
            )
        )

        if value is None:

            value = (
                getattr(
                    citation,
                    "id",
                    None,
                )
            )

        if value is None:
            return None

        normalized = (
            str(
                value
            )
            .strip()
        )

        if not normalized:
            return None

        return normalized