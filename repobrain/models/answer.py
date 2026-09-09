from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from repobrain.models.symbols import (
    RelationshipType,
)


class AnswerCitationKind(str, Enum):
    """
    Type of deterministic evidence backing an answer citation.
    """

    SOURCE = "SOURCE"
    GRAPH = "GRAPH"


class AnswerCitation(BaseModel):
    """
    One validated citation used by a generated answer.

    SOURCE citations correspond to EvidenceItem objects:

        [E1]
        [E2]

    GRAPH citations correspond to AgentGraphFact objects:

        [G1]
        [G2]
    """

    citation_id: str

    citation_kind: AnswerCitationKind

    # ------------------------------------------------------------------
    # Source evidence fields
    # ------------------------------------------------------------------

    evidence_id: str | None = None

    relative_path: str | None = None

    start_line: int | None = None

    end_line: int | None = None

    qualified_name: str | None = None

    # ------------------------------------------------------------------
    # Graph evidence fields
    # ------------------------------------------------------------------

    relationship_type: (
        RelationshipType
        | None
    ) = None

    source_symbol_id: str | None = None

    source_qualified_name: str | None = None

    target_symbol_id: str | None = None

    target_qualified_name: str | None = None


class GroundedAnswer(BaseModel):
    """
    Final natural-language answer produced from RepoBrain evidence.
    """

    query: str

    answer_text: str

    model_name: str

    citations: list[
        AnswerCitation
    ] = Field(
        default_factory=list,
    )

    invalid_citation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    evidence_items_available: int = Field(
        ge=0,
        default=0,
    )

    graph_facts_available: int = Field(
        ge=0,
        default=0,
    )

    grounded: bool = False