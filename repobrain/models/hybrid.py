from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from repobrain.models.retrieval import (
    ChunkType,
)

from repobrain.models.symbols import (
    RelationshipType,
)


class RetrievalChannel(str, Enum):
    """
    Independent retrieval/evidence channels used by RepoBrain.
    """

    SYMBOL = "SYMBOL"
    BM25 = "BM25"
    SEMANTIC = "SEMANTIC"
    GRAPH = "GRAPH"


class RetrievalEvidence(BaseModel):
    """
    One piece of evidence contributing to a hybrid result.

    raw_score:
        Original score from the underlying retriever.

    contribution:
        RRF contribution used by the hybrid ranker.

    rank:
        Rank within that individual retrieval channel.
    """

    channel: RetrievalChannel

    rank: int = Field(
        ge=1,
    )

    raw_score: float | None = None

    contribution: float = Field(
        ge=0.0,
    )

    relationship_type: (
        RelationshipType
        | None
    ) = None

    hop_distance: int | None = Field(
        default=None,
        ge=1,
    )

    seed_symbol_id: str | None = None

    details: str | None = None


class GraphRetrievalEvidence(BaseModel):
    """
    Graph evidence produced from a previously retrieved seed symbol.

    Graph evidence does not pretend to be semantic similarity.
    """

    symbol_id: str

    qualified_name: str

    file_id: str

    relationship_type: RelationshipType

    hop_distance: int = Field(
        ge=1,
    )

    seed_symbol_id: str

    direction: str


class HybridSearchResult(BaseModel):
    """
    Unified RepoBrain retrieval result.

    A single result may be supported by several independent
    retrieval channels.
    """

    result_key: str

    symbol_id: str | None = None
    chunk_id: str | None = None

    file_id: str | None = None
    relative_path: str | None = None

    qualified_name: str | None = None

    chunk_type: ChunkType | None = None

    start_line: int | None = None
    end_line: int | None = None

    excerpt: str | None = None

    fused_score: float = Field(
        ge=0.0,
    )

    channels: list[
        RetrievalChannel
    ] = Field(
        default_factory=list,
    )

    evidence: list[
        RetrievalEvidence
    ] = Field(
        default_factory=list,
    )

class GraphRetrievalEvidence(BaseModel):

    symbol_id: str

    qualified_name: str

    file_id: str

    relationship_type: RelationshipType

    hop_distance: int = Field(
        ge=1,
    )

    seed_symbol_id: str

    direction: str