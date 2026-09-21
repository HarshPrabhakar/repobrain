from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from enum import StrEnum
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RepositoryFingerprint(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    algorithm: str = Field(
        default="sha256-v1",
        min_length=1,
    )

    value: str = Field(
        min_length=64,
        max_length=64,
    )

    file_count: int = Field(
        ge=0,
    )

    total_size_bytes: int = Field(
        ge=0,
    )


class RepositoryRuntimeDiagnostics(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    files_discovered: int = Field(
        ge=0,
    )

    symbols: int = Field(
        ge=0,
    )

    relationships: int = Field(
        ge=0,
    )

    chunks: int = Field(
        ge=0,
    )

    bm25_documents: int = Field(
        ge=0,
    )

    graph_nodes: int = Field(
        ge=0,
    )

    graph_edges: int = Field(
        ge=0,
    )

    embedding_model: str = Field(
        min_length=1,
    )

    embedding_device: str = Field(
        min_length=1,
    )

    embedding_dimension: int = Field(
        ge=1,
    )

    llm_model: str = Field(
        min_length=1,
    )


class RepositoryRuntimeSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    runtime_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
    )

    repository_root: str = Field(
        min_length=1,
    )

    created_at: datetime = Field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            )
        )
    )

    conversation_count: int = Field(
        default=0,
        ge=0,
    )

    closed: bool = False

    fingerprint_algorithm: str | None = None

    fingerprint_value: str | None = None

    fingerprint_file_count: int | None = Field(
        default=None,
        ge=0,
    )


class ConversationSessionSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    session_id: str = Field(
        min_length=1,
    )

    runtime_id: str = Field(
        min_length=1,
    )

    repository_root: str = Field(
        min_length=1,
    )

    conversation_id: str = Field(
        min_length=1,
    )

    created_at: datetime

    last_accessed_at: datetime

    turn_number: int = Field(
        ge=0,
    )

    current_qualified_name: str | None = None

    relationship_focus_type: str | None = None

    relationship_qualified_name: str | None = None


class ApplicationHealthSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    ready: bool

    loaded_repositories: int = Field(
        ge=0,
    )

    active_sessions: int = Field(
        ge=0,
    )

    status: str = Field(
        min_length=1,
    )


class RepositoryDiagnosticsSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    runtime: RepositoryRuntimeSnapshot

    runtime_diagnostics: RepositoryRuntimeDiagnostics | None = None

    active_sessions: int = Field(
        ge=0,
    )


class SessionDiagnosticsSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    session: ConversationSessionSnapshot

    runtime_loaded: bool

    runtime_current: bool


class ApplicationDiagnosticsSnapshot(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    health: ApplicationHealthSnapshot

    repositories: tuple[
        RepositoryDiagnosticsSnapshot,
        ...,
    ]

    sessions: tuple[
        ConversationSessionSnapshot,
        ...,
    ]


# =============================================================================
# Phase 10.7 transport/API contracts
# =============================================================================


class ApplicationErrorCode(
    StrEnum
):
    INVALID_REQUEST = "INVALID_REQUEST"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_STALE = "SESSION_STALE"
    FOLLOW_UP_UNRESOLVED = "FOLLOW_UP_UNRESOLVED"
    REPOSITORY_NOT_FOUND = "REPOSITORY_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApplicationErrorResponse(
    BaseModel
):
    """
    Stable transport-safe error envelope.

    Clients should branch on code rather than parsing exception strings.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    code: ApplicationErrorCode

    message: str = Field(
        min_length=1,
    )

    retryable: bool = False


class OpenRepositoryRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    repository_root: str = Field(
        min_length=1,
    )


class CreateSessionRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    repository_root: str = Field(
        min_length=1,
    )


class AskRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    repository_root: str = Field(
        min_length=1,
    )

    session_id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
    )


class SessionCommandRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    repository_root: str = Field(
        min_length=1,
    )

    session_id: str = Field(
        min_length=1,
    )


class DeleteSessionRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    session_id: str = Field(
        min_length=1,
    )


class CloseRepositoryRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    repository_root: str = Field(
        min_length=1,
    )


class CitationResponse(
    BaseModel
):
    """
    Transport-safe citation representation.

    Fields are optional because source and graph citations carry
    different metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    citation_id: str = Field(
        min_length=1,
    )

    citation_kind: str | None = None

    relative_path: str | None = None

    start_line: int | None = Field(
        default=None,
        ge=1,
    )

    end_line: int | None = Field(
        default=None,
        ge=1,
    )

    qualified_name: str | None = None

    relationship_type: str | None = None

    source_qualified_name: str | None = None

    target_qualified_name: str | None = None


class InvestigationStateResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    conversation_id: str = Field(
        min_length=1,
    )

    turn_number: int = Field(
        ge=0,
    )

    current_qualified_name: str | None = None

    previous_qualified_name: str | None = None

    previous_intent: str | None = None

    relationship_focus_type: str | None = None

    relationship_qualified_name: str | None = None

    last_query: str | None = None

    last_resolved_query: str | None = None

    last_grounded: bool | None = None

    last_citation_ids: tuple[
        str,
        ...,
    ] = ()

    last_invalid_citation_ids: tuple[
        str,
        ...,
    ] = ()


class AskResponse(
    BaseModel
):
    """
    Stable answer DTO exposed outside the application core.
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

    answer_text: str

    model_name: str | None = None

    intent: str | None = None

    grounded: bool

    citations: tuple[
        CitationResponse,
        ...,
    ] = ()

    invalid_citation_ids: tuple[
        str,
        ...,
    ] = ()

    state: InvestigationStateResponse


class DeleteSessionResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    deleted: bool


class CloseRepositoryResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    closed: bool
