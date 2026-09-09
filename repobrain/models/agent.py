from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from repobrain.models.evidence import (
    EvidenceBundle,
)

from repobrain.models.symbols import (
    RelationshipType,
)


class AgentIntent(str, Enum):
    """
    High-level deterministic investigation intent.
    """

    CALLERS = "CALLERS"
    CALLEES = "CALLEES"
    DEFINITION = "DEFINITION"
    IMPLEMENTATION = "IMPLEMENTATION"
    DOCUMENTATION = "DOCUMENTATION"
    GENERAL = "GENERAL"


class AgentAction(str, Enum):
    """
    Actions the Phase 7 orchestrator is allowed to execute.
    """

    HYBRID_EVIDENCE = "HYBRID_EVIDENCE"
    GRAPH_CALLERS = "GRAPH_CALLERS"
    GRAPH_CALLEES = "GRAPH_CALLEES"


class AgentStatus(str, Enum):
    """
    Lifecycle state for a bounded investigation.
    """

    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    FAILED = "FAILED"


class AgentGraphFact(BaseModel):
    """
    One deterministic graph fact discovered by the agent.
    """

    relationship_type: RelationshipType

    source_symbol_id: str
    source_qualified_name: str

    target_symbol_id: str
    target_qualified_name: str


class AgentStep(BaseModel):
    """
    One action performed during an investigation.
    """

    step_number: int = Field(
        ge=1,
    )

    action: AgentAction

    query: str

    reason: str

    observations: int = Field(
        ge=0,
        default=0,
    )


class AgentState(BaseModel):
    """
    Explicit bounded investigation state.
    """

    query: str

    intent: AgentIntent

    status: AgentStatus = (
        AgentStatus.READY
    )

    max_steps: int = Field(
        ge=1,
        default=3,
    )

    steps: list[
        AgentStep
    ] = Field(
        default_factory=list,
    )

    @property
    def step_count(
        self,
    ) -> int:
        return len(
            self.steps
        )

    @property
    def remaining_steps(
        self,
    ) -> int:
        return max(
            0,
            self.max_steps
            - self.step_count,
        )


class AgentRunResult(BaseModel):
    """
    Structured output of one Phase 7 investigation.

    Phase 7 intentionally returns evidence/facts rather than
    generating a natural-language repository answer.
    """

    state: AgentState

    evidence: EvidenceBundle | None = None

    graph_facts: list[
        AgentGraphFact
    ] = Field(
        default_factory=list,
    )

    resolved_symbol_id: str | None = None

    resolved_qualified_name: str | None = None