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


class RepositoryRuntimeSnapshot(
    BaseModel
):
    """
    Immutable diagnostic snapshot describing one loaded
    RepoBrain repository runtime.

    This model intentionally contains metadata only.

    It does not contain:

    - repository source
    - embeddings
    - indexes
    - graph objects
    - prompts
    - LLM state
    - conversation history
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