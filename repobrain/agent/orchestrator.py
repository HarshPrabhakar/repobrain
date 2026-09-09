from __future__ import annotations

from repobrain.agent.router import (
    DeterministicAgentRouter,
)

from repobrain.agent.tools import (
    AgentToolbox,
)

from repobrain.models.agent import (
    AgentAction,
    AgentIntent,
    AgentRunResult,
    AgentState,
    AgentStatus,
    AgentStep,
)


DEFAULT_MAX_AGENT_STEPS = 3

DEFAULT_AGENT_RETRIEVAL_TOP_K = 20


class RepoBrainAgentOrchestrator:
    """
    Phase 7 deterministic bounded repository investigation loop.

    Important:

    This is an orchestrator, not a truth source.

    Repository facts still come exclusively from deterministic
    RepoBrain tools.
    """

    def __init__(
        self,
        *,
        toolbox: AgentToolbox,
        router: DeterministicAgentRouter | None = None,
        max_steps: int = DEFAULT_MAX_AGENT_STEPS,
        retrieval_top_k: int = (
            DEFAULT_AGENT_RETRIEVAL_TOP_K
        ),
    ) -> None:

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be > 0."
            )

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be > 0."
            )

        self.toolbox = toolbox

        self.router = (
            router
            or DeterministicAgentRouter()
        )

        self.max_steps = max_steps

        self.retrieval_top_k = (
            retrieval_top_k
        )

    # =====================================================================
    # Public API
    # =====================================================================

    def run(
        self,
        query: str,
    ) -> AgentRunResult:

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        intent = self.router.classify(
            normalized_query
        )

        state = AgentState(
            query=normalized_query,
            intent=intent,
            status=(
                AgentStatus.RUNNING
            ),
            max_steps=(
                self.max_steps
            ),
        )

        graph_facts = []

        evidence = None

        resolved_symbol = None

        # ===============================================================
        # Graph-first investigations
        # ===============================================================

        if intent in {
            AgentIntent.CALLERS,
            AgentIntent.CALLEES,
        }:

            reference = (
                self.router
                .extract_symbol_reference(
                    normalized_query
                )
            )

            if reference:

                resolved_symbol = (
                    self.toolbox
                    .resolve_symbol(
                        reference
                    )
                )

            if resolved_symbol is not None:

                if (
                    intent
                    == AgentIntent.CALLERS
                ):

                    graph_facts = (
                        self.toolbox.callers(
                            resolved_symbol
                            .symbol_id
                        )
                    )

                    self._record_step(
                        state=state,
                        action=(
                            AgentAction.GRAPH_CALLERS
                        ),
                        query=reference
                        or normalized_query,
                        reason=(
                            "Question asks for callers "
                            "of an explicitly resolved symbol."
                        ),
                        observations=len(
                            graph_facts
                        ),
                    )

                else:

                    graph_facts = (
                        self.toolbox.callees(
                            resolved_symbol
                            .symbol_id
                        )
                    )

                    self._record_step(
                        state=state,
                        action=(
                            AgentAction.GRAPH_CALLEES
                        ),
                        query=reference
                        or normalized_query,
                        reason=(
                            "Question asks for callees "
                            "of an explicitly resolved symbol."
                        ),
                        observations=len(
                            graph_facts
                        ),
                    )

        # ===============================================================
        # Evidence retrieval
        #
        # Every investigation still receives source-backed evidence when
        # the step budget allows it.
        # ===============================================================

        if state.remaining_steps > 0:

            evidence = (
                self.toolbox
                .retrieve_evidence(
                    normalized_query,
                    retrieval_top_k=(
                        self.retrieval_top_k
                    ),
                )
            )

            self._record_step(
                state=state,
                action=(
                    AgentAction.HYBRID_EVIDENCE
                ),
                query=normalized_query,
                reason=(
                    self._retrieval_reason(
                        intent
                    )
                ),
                observations=len(
                    evidence.items
                ),
            )

        # ===============================================================
        # Completion
        # ===============================================================

        if (
            state.step_count
            >= state.max_steps
        ):
            state.status = (
                AgentStatus.MAX_STEPS_REACHED
            )

        else:
            state.status = (
                AgentStatus.COMPLETED
            )

        return AgentRunResult(
            state=state,
            evidence=evidence,
            graph_facts=(
                graph_facts
            ),
            resolved_symbol_id=(
                resolved_symbol.symbol_id
                if resolved_symbol
                is not None
                else None
            ),
            resolved_qualified_name=(
                resolved_symbol
                .qualified_name
                if resolved_symbol
                is not None
                else None
            ),
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @staticmethod
    def _record_step(
        *,
        state: AgentState,
        action: AgentAction,
        query: str,
        reason: str,
        observations: int,
    ) -> None:

        if state.remaining_steps <= 0:
            return

        state.steps.append(
            AgentStep(
                step_number=(
                    state.step_count
                    + 1
                ),
                action=action,
                query=query,
                reason=reason,
                observations=(
                    observations
                ),
            )
        )

    @staticmethod
    def _retrieval_reason(
        intent: AgentIntent,
    ) -> str:

        if (
            intent
            == AgentIntent.DOCUMENTATION
        ):
            return (
                "Documentation-oriented question "
                "requires repository evidence."
            )

        if (
            intent
            == AgentIntent.IMPLEMENTATION
        ):
            return (
                "Implementation question requires "
                "source-backed repository evidence."
            )

        if (
            intent
            == AgentIntent.DEFINITION
        ):
            return (
                "Definition question requires "
                "symbol/source evidence."
            )

        if intent in {
            AgentIntent.CALLERS,
            AgentIntent.CALLEES,
        }:
            return (
                "Graph fact is supplemented with "
                "source-backed evidence."
            )

        return (
            "General repository question requires "
            "hybrid evidence retrieval."
        )