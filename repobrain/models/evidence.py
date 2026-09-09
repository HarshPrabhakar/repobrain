from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from repobrain.models.hybrid import (
    RetrievalChannel,
)

from repobrain.models.symbols import (
    RelationshipType,
)


class EvidenceKind(str, Enum):
    """
    Type of evidence supplied to downstream reasoning.
    """

    SYMBOL = "SYMBOL"
    FILE = "FILE"
    DOCUMENTATION = "DOCUMENTATION"
    CONFIG = "CONFIG"
    TEST = "TEST"
    OTHER = "OTHER"


class EvidenceGraphRelation(BaseModel):
    """
    One structural relationship attached to an evidence item.
    """

    relationship_type: RelationshipType

    direction: str

    related_symbol_id: str

    related_qualified_name: str

    line_number: int | None = None


class EvidenceItem(BaseModel):
    """
    One bounded repository evidence item.

    This is the main object future LLM reasoning will consume.
    """

    evidence_id: str

    result_key: str

    file_id: str | None = None

    relative_path: str | None = None

    symbol_id: str | None = None

    qualified_name: str | None = None

    start_line: int | None = None

    end_line: int | None = None

    source_text: str

    evidence_kind: EvidenceKind

    fused_score: float = Field(
        ge=0.0,
    )

    retrieval_channels: list[
        RetrievalChannel
    ] = Field(
        default_factory=list,
    )

    graph_context: list[
        EvidenceGraphRelation
    ] = Field(
        default_factory=list,
    )

    estimated_tokens: int = Field(
        ge=0,
        default=0,
    )

    character_count: int = Field(
        ge=0,
        default=0,
    )


class EvidenceBundle(BaseModel):
    """
    Final bounded context assembled for one repository query.
    """

    query: str

    items: list[
        EvidenceItem
    ] = Field(
        default_factory=list,
    )

    total_characters: int = Field(
        ge=0,
        default=0,
    )

    estimated_tokens: int = Field(
        ge=0,
        default=0,
    )

    omitted_items: int = Field(
        ge=0,
        default=0,
    )

    max_items: int = Field(
        ge=1,
    )

    max_characters: int = Field(
        ge=1,
    )