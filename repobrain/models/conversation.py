from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from repobrain.models.agent import (
    AgentIntent,
)


class RelationshipFocusType(StrEnum):
    """
    Deterministic relationship target currently available
    for conversational follow-up resolution.
    """

    CALLER = "CALLER"

    CALLEE = "CALLEE"


class ConversationTurn(BaseModel):
    """
    One completed RepoBrain investigation turn.

    The model stores only bounded deterministic state required
    for future follow-up resolution.

    It intentionally does not store:

    - full prompts
    - full evidence bundles
    - embeddings
    - arbitrary LLM memory
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    turn_number: int = Field(
        ge=1,
    )

    query: str = Field(
        min_length=1,
    )

    resolved_query: str | None = None

    intent: AgentIntent | None = None

    # ------------------------------------------------------------------
    # Primary symbol focus
    # ------------------------------------------------------------------

    focused_symbol_id: str | None = None

    focused_qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Deterministic relationship focus
    # ------------------------------------------------------------------

    relationship_focus_type: (
        RelationshipFocusType
        | None
    ) = None

    relationship_symbol_id: str | None = None

    relationship_qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Grounding metadata
    # ------------------------------------------------------------------

    grounded: bool | None = None

    citation_ids: tuple[str, ...] = ()

    invalid_citation_ids: tuple[str, ...] = ()


class InvestigationState(BaseModel):
    """
    Stateful context for one RepoBrain investigation session.

    This represents repository-investigation state rather than
    unrestricted conversational memory.

    Phase 9.6 adds a second bounded focus:

        relationship focus

    This is populated only from unambiguous deterministic
    repository graph results.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    conversation_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
    )

    turn_number: int = Field(
        default=0,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Current primary symbol focus
    # ------------------------------------------------------------------

    current_symbol_id: str | None = None

    current_qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Previous primary symbol focus
    # ------------------------------------------------------------------

    previous_symbol_id: str | None = None

    previous_qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Current deterministic relationship focus
    # ------------------------------------------------------------------

    relationship_focus_type: (
        RelationshipFocusType
        | None
    ) = None

    relationship_symbol_id: str | None = None

    relationship_qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Previous investigation metadata
    # ------------------------------------------------------------------

    previous_intent: AgentIntent | None = None

    last_query: str | None = None

    last_resolved_query: str | None = None

    # ------------------------------------------------------------------
    # Previous grounding metadata
    # ------------------------------------------------------------------

    last_citation_ids: tuple[str, ...] = ()

    last_invalid_citation_ids: tuple[str, ...] = ()

    last_grounded: bool | None = None

    # ------------------------------------------------------------------
    # Bounded history
    # ------------------------------------------------------------------

    history: tuple[
        ConversationTurn,
        ...,
    ] = ()

    @property
    def has_focus(
        self,
    ) -> bool:
        """
        Return True when the investigation currently has a
        known primary repository-symbol focus.
        """

        return (
            self.current_symbol_id
            is not None
            or self.current_qualified_name
            is not None
        )

    @property
    def has_relationship_focus(
        self,
    ) -> bool:
        """
        Return True when one unambiguous deterministic graph
        relationship target is available.
        """

        return (
            self.relationship_focus_type
            is not None
            and (
                self.relationship_symbol_id
                is not None
                or self.relationship_qualified_name
                is not None
            )
        )

    @property
    def current_turn(
        self,
    ) -> ConversationTurn | None:
        """
        Return the latest completed turn.
        """

        if not self.history:
            return None

        return self.history[-1]


class FollowUpReferenceKind(StrEnum):
    """
    Deterministic category of conversational reference.
    """

    NONE = "NONE"

    CURRENT_FOCUS = "CURRENT_FOCUS"

    PREVIOUS_FOCUS = "PREVIOUS_FOCUS"

    CALLER_FOCUS = "CALLER_FOCUS"

    CALLEE_FOCUS = "CALLEE_FOCUS"


class FollowUpResolution(BaseModel):
    """
    Result of deterministic conversational-reference resolution.

    original_query:
        User text exactly as supplied after whitespace normalization.

    resolved_query:
        Query that may safely be passed to the existing RepoBrain
        agent pipeline.

    used_context:
        Whether conversation state was actually used.

    reference_kind:
        Which deterministic state target was selected.

    reference_text:
        Exact conversational phrase that triggered resolution.

    resolved_symbol_id / resolved_qualified_name:
        Repository symbol used to replace the conversational reference.

    unresolved_reference:
        True when a context-dependent expression was detected but
        could not be safely resolved.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    original_query: str = Field(
        min_length=1,
    )

    resolved_query: str = Field(
        min_length=1,
    )

    used_context: bool = False

    reference_kind: FollowUpReferenceKind = (
        FollowUpReferenceKind.NONE
    )

    reference_text: str | None = None

    resolved_symbol_id: str | None = None

    resolved_qualified_name: str | None = None

    unresolved_reference: bool = False