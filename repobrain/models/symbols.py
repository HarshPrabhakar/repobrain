from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    MODULE = "MODULE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"


class RelationshipType(str, Enum):
    DEFINES = "DEFINES"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    DECORATED_BY = "DECORATED_BY"


class ResolutionType(str, Enum):
    """
    How confidently RepoBrain resolved a relationship.

    EXACT:
        Directly known from deterministic structure.

    INFERRED:
        Best-effort static resolution.

    UNRESOLVED:
        Target could not reliably be determined.
    """

    EXACT = "EXACT"
    INFERRED = "INFERRED"
    UNRESOLVED = "UNRESOLVED"


class CodeSymbol(BaseModel):
    """
    One structural Python symbol discovered from the AST.
    """

    symbol_id: str

    repository_id: str
    file_id: str

    name: str
    qualified_name: str
    fully_qualified_name: str

    symbol_type: SymbolType

    module: str

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    parent_symbol_id: str | None = None

    signature: str | None = None
    docstring: str | None = None

    is_async: bool = False

    decorators: list[str] = Field(
        default_factory=list
    )


class CodeRelationship(BaseModel):
    """
    Structural relationship extracted from Python source.

    target_symbol_id may be None when the referenced symbol has not yet
    been resolved to a repository-local CodeSymbol.
    """

    relationship_id: str

    repository_id: str
    file_id: str

    relationship_type: RelationshipType

    source_symbol_id: str

    target_symbol_id: str | None = None

    source_qualified_name: str
    target_qualified_name: str

    line_number: int | None = Field(
        default=None,
        ge=1,
    )

    resolution: ResolutionType

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ParseError(BaseModel):
    file_id: str
    relative_path: str

    error_type: str
    message: str

    line_number: int | None = None
    column: int | None = None


class PythonFileAnalysis(BaseModel):
    """
    Phase 2 result for one Python file.
    """

    file_id: str
    relative_path: str
    module: str

    symbols: list[CodeSymbol] = Field(
        default_factory=list
    )

    relationships: list[CodeRelationship] = Field(
        default_factory=list
    )


class PythonRepositoryAnalysis(BaseModel):
    """
    Aggregate Phase 2 result for an entire repository.
    """

    repository_id: str

    files_analyzed: int = 0

    symbols: list[CodeSymbol] = Field(
        default_factory=list
    )

    relationships: list[CodeRelationship] = Field(
        default_factory=list
    )

    errors: list[ParseError] = Field(
        default_factory=list
    )