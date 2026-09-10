from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from threading import RLock
from uuid import uuid4

from repobrain.application.runtime import (
    RepositoryRuntime,
    RepositoryRuntimeClosedError,
)
from repobrain.conversation import (
    StatefulGroundedAnsweringOrchestrator,
)
from repobrain.models.application import (
    ConversationSessionSnapshot,
)


class ConversationSessionNotFoundError(
    KeyError
):
    """
    Raised when a requested application session does not exist.
    """


class ConversationSessionRuntimeMismatchError(
    RuntimeError
):
    """
    Raised when a session is accessed through the wrong runtime.
    """


@dataclass
class _ManagedConversationSession:
    """
    Internal mutable wrapper around one Phase-9 conversation.
    """

    session_id: str
    runtime_id: str
    repository_root: str
    conversation: StatefulGroundedAnsweringOrchestrator
    created_at: datetime
    last_accessed_at: datetime


class ConversationSessionManager:
    """
    Process-local manager for independent RepoBrain conversations.

    Phase 10.4 separates:

        repository runtime lifetime
        from
        conversation session lifetime

    One RepositoryRuntime may therefore serve many isolated sessions
    without rebuilding scanner/AST/retrieval/graph/LLM dependencies.

    The manager does not perform retrieval or answer generation itself.
    It only owns session identity, lifecycle, and routing.
    """

    def __init__(
        self,
    ) -> None:

        self._sessions: dict[
            str,
            _ManagedConversationSession,
        ] = {}

        self._lock = (
            RLock()
        )

    # ==================================================================
    # Session lifecycle
    # ==================================================================

    def create(
        self,
        runtime: RepositoryRuntime,
    ) -> str:
        """
        Create a fresh application session backed by one runtime.
        """

        if runtime.closed:

            raise (
                RepositoryRuntimeClosedError(
                    "Cannot create a conversation session "
                    "from a closed repository runtime."
                )
            )

        conversation = (
            runtime.create_conversation()
        )

        now = (
            datetime.now(
                timezone.utc
            )
        )

        session_id = (
            uuid4().hex
        )

        managed = (
            _ManagedConversationSession(
                session_id=(
                    session_id
                ),
                runtime_id=(
                    runtime.runtime_id
                ),
                repository_root=(
                    str(
                        runtime.repository_root
                    )
                ),
                conversation=(
                    conversation
                ),
                created_at=(
                    now
                ),
                last_accessed_at=(
                    now
                ),
            )
        )

        with self._lock:

            self._sessions[
                session_id
            ] = managed

        return session_id

    def get(
        self,
        session_id: str,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> StatefulGroundedAnsweringOrchestrator:
        """
        Return one managed conversation and mark it accessed.

        When runtime is supplied, the session must belong to that exact
        runtime. This prevents a stale or foreign session from being
        routed into a rebuilt repository runtime.
        """

        managed = (
            self._get_managed(
                session_id
            )
        )

        if (
            runtime is not None
            and managed.runtime_id
            != runtime.runtime_id
        ):

            raise (
                ConversationSessionRuntimeMismatchError(
                    "Conversation session belongs to a different "
                    "repository runtime."
                )
            )

        now = (
            datetime.now(
                timezone.utc
            )
        )

        with self._lock:

            managed.last_accessed_at = (
                now
            )

        return (
            managed.conversation
        )

    def delete(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete one session from the manager.

        Returns False when the session was already absent.
        """

        normalized_id = (
            self._normalize_session_id(
                session_id
            )
        )

        with self._lock:

            removed = (
                self._sessions.pop(
                    normalized_id,
                    None,
                )
            )

        return (
            removed is not None
        )

    def clear(
        self,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> int:
        """
        Delete sessions.

        With runtime=None:
            clear every session.

        With runtime supplied:
            clear only sessions belonging to that runtime.

        Returns the number of deleted sessions.
        """

        with self._lock:

            if runtime is None:

                count = (
                    len(
                        self._sessions
                    )
                )

                self._sessions.clear()

                return count

            target_runtime_id = (
                runtime.runtime_id
            )

            targets = [
                session_id
                for (
                    session_id,
                    managed,
                )
                in self._sessions.items()
                if (
                    managed.runtime_id
                    == target_runtime_id
                )
            ]

            for session_id in targets:

                self._sessions.pop(
                    session_id,
                    None,
                )

            return (
                len(
                    targets
                )
            )

    # ==================================================================
    # Conversation operations
    # ==================================================================

    def run(
        self,
        session_id: str,
        query: str,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> object:
        """
        Route one user query to the correct Phase-9 conversation.
        """

        conversation = (
            self.get(
                session_id,
                runtime=runtime,
            )
        )

        return (
            conversation.run(
                query
            )
        )

    def clear_context(
        self,
        session_id: str,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> object:
        """
        Clear Phase-9 investigation context for one session while
        preserving that conversation's ID.
        """

        conversation = (
            self.get(
                session_id,
                runtime=runtime,
            )
        )

        return (
            conversation.clear()
        )

    def new_conversation(
        self,
        session_id: str,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> object:
        """
        Reset one managed session to a fresh Phase-9 conversation ID
        while preserving the application-level session_id.
        """

        conversation = (
            self.get(
                session_id,
                runtime=runtime,
            )
        )

        return (
            conversation.new_conversation()
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def snapshot(
        self,
        session_id: str,
    ) -> ConversationSessionSnapshot:
        """
        Return immutable metadata for one managed session.
        """

        managed = (
            self._get_managed(
                session_id
            )
        )

        return (
            self._snapshot_from_managed(
                managed
            )
        )

    def snapshots(
        self,
        *,
        runtime: RepositoryRuntime | None = None,
    ) -> tuple[
        ConversationSessionSnapshot,
        ...,
    ]:
        """
        Return deterministic snapshots sorted by session ID.
        """

        with self._lock:

            managed_sessions = (
                tuple(
                    self._sessions.values()
                )
            )

        if runtime is not None:

            managed_sessions = tuple(
                managed
                for managed
                in managed_sessions
                if (
                    managed.runtime_id
                    == runtime.runtime_id
                )
            )

        snapshots = [
            self._snapshot_from_managed(
                managed
            )
            for managed
            in managed_sessions
        ]

        snapshots.sort(
            key=lambda item: (
                item.session_id
            )
        )

        return tuple(
            snapshots
        )

    def contains(
        self,
        session_id: str,
    ) -> bool:

        normalized_id = (
            self._normalize_session_id(
                session_id
            )
        )

        with self._lock:

            return (
                normalized_id
                in self._sessions
            )

    def __len__(
        self,
    ) -> int:

        with self._lock:

            return (
                len(
                    self._sessions
                )
            )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_managed(
        self,
        session_id: str,
    ) -> _ManagedConversationSession:

        normalized_id = (
            self._normalize_session_id(
                session_id
            )
        )

        with self._lock:

            managed = (
                self._sessions.get(
                    normalized_id
                )
            )

        if managed is None:

            raise (
                ConversationSessionNotFoundError(
                    f"Unknown conversation session: "
                    f"{normalized_id}"
                )
            )

        return managed

    @staticmethod
    def _snapshot_from_managed(
        managed: _ManagedConversationSession,
    ) -> ConversationSessionSnapshot:

        state = (
            managed.conversation.state
        )

        relationship_type = getattr(
            state,
            "relationship_focus_type",
            None,
        )

        relationship_value = (
            getattr(
                relationship_type,
                "value",
                None,
            )
            if relationship_type is not None
            else None
        )

        if (
            relationship_type is not None
            and relationship_value is None
        ):

            relationship_value = (
                str(
                    relationship_type
                )
            )

        return (
            ConversationSessionSnapshot(
                session_id=(
                    managed.session_id
                ),
                runtime_id=(
                    managed.runtime_id
                ),
                repository_root=(
                    managed.repository_root
                ),
                conversation_id=(
                    state.conversation_id
                ),
                created_at=(
                    managed.created_at
                ),
                last_accessed_at=(
                    managed.last_accessed_at
                ),
                turn_number=(
                    state.turn_number
                ),
                current_qualified_name=(
                    state.current_qualified_name
                ),
                relationship_focus_type=(
                    relationship_value
                ),
                relationship_qualified_name=(
                    state.relationship_qualified_name
                ),
            )
        )

    @staticmethod
    def _normalize_session_id(
        session_id: str,
    ) -> str:

        normalized = (
            session_id.strip()
        )

        if not normalized:

            raise ValueError(
                "session_id cannot be empty."
            )

        return normalized
