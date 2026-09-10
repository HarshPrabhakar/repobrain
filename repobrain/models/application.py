from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RepositoryFingerprint(
    BaseModel
):
    """
    Immutable content fingerprint for one repository snapshot.
    """

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


class RepositoryRuntimeSnapshot(
    BaseModel
):
    """
    Immutable diagnostic snapshot describing one loaded
    RepoBrain repository runtime.
    """

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
    """
    Immutable metadata for one application-level conversation session.

    The snapshot deliberately contains no prompts, evidence bundles,
    embeddings, or model-private state.
    """

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
