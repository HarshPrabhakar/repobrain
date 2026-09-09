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
    One Phase 3C semantic-search result.

    Score is cosine-style vector similarity when normalized
    embeddings are used with FAISS inner-product search.
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

    score: float

    excerpt: str