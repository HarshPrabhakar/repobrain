from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """
    Searchable repository chunk types.
    """

    MODULE = "MODULE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    FILE = "FILE"


class CodeChunk(BaseModel):
    """
    Searchable unit of repository content.

    Python source uses symbol-aware chunks.

    Non-Python source initially uses one whole-file chunk.
    """

    chunk_id: str

    repository_id: str
    file_id: str

    relative_path: str

    symbol_id: str | None = None

    qualified_name: str | None = None

    chunk_type: ChunkType

    language: str

    start_line: int = Field(
        ge=1
    )

    end_line: int = Field(
        ge=1
    )

    text: str

    content_hash: str | None = None


class LexicalSearchResult(BaseModel):
    """
    One Phase 3B lexical-search result.
    """

    chunk_id: str

    file_id: str
    relative_path: str

    symbol_id: str | None = None

    qualified_name: str | None = None

    chunk_type: ChunkType

    language: str

    start_line: int
    end_line: int

    score: float = Field(
        ge=0.0
    )

    matched_terms: list[str] = Field(
        default_factory=list
    )

    excerpt: str


class SemanticSearchResult(BaseModel):
    """
    One semantic-search result.

    score:
        Raw semantic similarity returned by FAISS.

    ranking_score:
        Deterministic Phase 3C.3 reranking score.

    The ranking score is not a probability.
    """

    chunk_id: str

    file_id: str
    relative_path: str

    symbol_id: str | None = None
    qualified_name: str | None = None

    chunk_type: ChunkType

    language: str

    start_line: int
    end_line: int

    # Raw vector similarity.
    score: float

    excerpt: str

    # Phase 3C.3 metadata.
    ranking_score: float | None = None

    query_intent: str | None = None

    source_kind: str | None = None

    ranking_reasons: list[str] = Field(
        default_factory=list
    )