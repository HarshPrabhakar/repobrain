from __future__ import annotations

from pathlib import Path
from threading import RLock

from repobrain.application.builder import (
    RepositoryRuntimeBuilder,
)
from repobrain.application.diagnostics import (
    ApplicationDiagnosticsService,
)
from repobrain.application.fingerprint import (
    RepositoryFingerprinter,
)
from repobrain.application.registry import (
    RepositoryRuntimeRegistry,
)
from repobrain.application.runtime import (
    RepositoryRuntime,
)
from repobrain.application.sessions import (
    ConversationSessionManager,
    ConversationSessionNotFoundError,
)
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationHealthSnapshot,
    ConversationSessionSnapshot,
    RepositoryDiagnosticsSnapshot,
    RepositoryRuntimeSnapshot,
    SessionDiagnosticsSnapshot,
)
from repobrain.models.conversation import (
    InvestigationState,
)


class RepositorySessionStaleError(
    RuntimeError
):
    """
    Raised when a conversation session belongs to an older repository
    runtime that has been invalidated and rebuilt.

    A stale session is never silently rebound to a different runtime.
    """


class RepoBrainApplicationService:
    """
    Phase 10.5 application boundary for RepoBrain.

    This service is the single high-level interface intended for CLI,
    API, and future UI clients.

    Responsibilities:

    - open/cache repository runtimes
    - validate repository fingerprints before reuse
    - create independent conversation sessions
    - route questions to the correct session
    - expose investigation/session/runtime state
    - clear or reset conversations
    - close repositories safely

    It does not implement retrieval, graph analysis, or LLM reasoning.
    Those remain inside the lower RepoBrain layers.
    """

    def __init__(
        self,
        *,
        runtime_builder: RepositoryRuntimeBuilder | None = None,
        runtime_registry: RepositoryRuntimeRegistry | None = None,
        session_manager: ConversationSessionManager | None = None,
        fingerprinter: RepositoryFingerprinter | None = None,
    ) -> None:

        self._runtime_builder = (
            runtime_builder
            if runtime_builder is not None
            else RepositoryRuntimeBuilder()
        )

        self._fingerprinter = (
            fingerprinter
            if fingerprinter is not None
            else RepositoryFingerprinter()
        )

        self._runtime_registry = (
            runtime_registry
            if runtime_registry is not None
            else RepositoryRuntimeRegistry(
                runtime_factory=(
                    self._runtime_builder.build
                ),
                fingerprint_provider=(
                    self._fingerprinter.calculate
                ),
            )
        )

        self._session_manager = (
            session_manager
            if session_manager is not None
            else ConversationSessionManager()
        )

        self._diagnostics_service = (
            ApplicationDiagnosticsService(
                runtime_registry=(
                    self._runtime_registry
                ),
                session_manager=(
                    self._session_manager
                ),
            )
        )

        self._lock = (
            RLock()
        )

    # ==================================================================
    # Public components
    # ==================================================================

    @property
    def runtime_registry(
        self,
    ) -> RepositoryRuntimeRegistry:

        return (
            self._runtime_registry
        )

    @property
    def session_manager(
        self,
    ) -> ConversationSessionManager:

        return (
            self._session_manager
        )

    # ==================================================================
    # Diagnostics + health
    # ==================================================================

    def health(
        self,
    ) -> ApplicationHealthSnapshot:

        return (
            self._diagnostics_service
            .health()
        )

    def diagnostics(
        self,
    ) -> ApplicationDiagnosticsSnapshot:

        return (
            self._diagnostics_service
            .diagnostics()
        )

    def repository_diagnostics(
        self,
        repository_root: Path,
    ) -> RepositoryDiagnosticsSnapshot | None:

        return (
            self._diagnostics_service
            .repository_diagnostics(
                repository_root
            )
        )

    def session_diagnostics(
        self,
        session_id: str,
    ) -> SessionDiagnosticsSnapshot:

        return (
            self._diagnostics_service
            .session_diagnostics(
                session_id
            )
        )

    # ==================================================================
    # Repository lifecycle
    # ==================================================================

    def open_repository(
        self,
        repository_root: Path,
    ) -> RepositoryRuntimeSnapshot:
        """
        Open a repository or reuse its current cached runtime.

        Fingerprint validation is performed by the runtime registry.
        """

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        return (
            runtime.snapshot()
        )

    def close_repository(
        self,
        repository_root: Path,
    ) -> bool:
        """
        Close one loaded repository runtime.

        Sessions belonging to that exact runtime are deleted first.
        """

        with self._lock:

            runtime = (
                self._runtime_registry.get(
                    repository_root
                )
            )

            if runtime is None:
                return False

            self._session_manager.clear(
                runtime=runtime
            )

            return (
                self._runtime_registry.evict(
                    repository_root
                )
            )

    def close_all(
        self,
    ) -> None:
        """
        Close every session and repository runtime.
        """

        with self._lock:

            self._session_manager.clear()

            self._runtime_registry.clear()

    def repository_snapshots(
        self,
    ) -> tuple[
        RepositoryRuntimeSnapshot,
        ...,
    ]:

        return (
            self._runtime_registry
            .snapshots()
        )

    # ==================================================================
    # Session lifecycle
    # ==================================================================

    def create_session(
        self,
        repository_root: Path,
    ) -> ConversationSessionSnapshot:
        """
        Create a fresh application session for the current repository
        runtime.
        """

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        session_id = (
            self._session_manager.create(
                runtime
            )
        )

        return (
            self._session_manager.snapshot(
                session_id
            )
        )

    def delete_session(
        self,
        session_id: str,
    ) -> bool:

        return (
            self._session_manager.delete(
                session_id
            )
        )

    def session_snapshot(
        self,
        session_id: str,
    ) -> ConversationSessionSnapshot:

        return (
            self._session_manager.snapshot(
                session_id
            )
        )

    def session_snapshots(
        self,
        *,
        repository_root: Path | None = None,
    ) -> tuple[
        ConversationSessionSnapshot,
        ...,
    ]:
        """
        Return all sessions, optionally limited to the current runtime
        for one repository.
        """

        if repository_root is None:

            return (
                self._session_manager
                .snapshots()
            )

        runtime = (
            self._runtime_registry.get(
                repository_root
            )
        )

        if runtime is None:
            return ()

        return (
            self._session_manager
            .snapshots(
                runtime=runtime
            )
        )

    # ==================================================================
    # Query routing
    # ==================================================================

    def ask(
        self,
        repository_root: Path,
        session_id: str,
        query: str,
    ) -> object:
        """
        Ask one question in one repository session.

        The repository is fingerprint-validated before the question is
        routed.

        If repository content changed and the runtime was rebuilt, an
        existing session is considered stale and is NOT silently rebound.
        """

        normalized_query = (
            query.strip()
        )

        if not normalized_query:

            raise ValueError(
                "query cannot be empty."
            )

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        self._ensure_session_matches_runtime(
            session_id=(
                session_id
            ),
            runtime=(
                runtime
            ),
        )

        return (
            self._session_manager.run(
                session_id,
                normalized_query,
                runtime=runtime,
            )
        )

    # ==================================================================
    # Investigation state
    # ==================================================================

    def get_state(
        self,
        repository_root: Path,
        session_id: str,
    ) -> InvestigationState:
        """
        Return current immutable Phase-9 investigation state.
        """

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        self._ensure_session_matches_runtime(
            session_id=session_id,
            runtime=runtime,
        )

        conversation = (
            self._session_manager.get(
                session_id,
                runtime=runtime,
            )
        )

        return (
            conversation.state
        )

    def clear_session(
        self,
        repository_root: Path,
        session_id: str,
    ) -> ConversationSessionSnapshot:
        """
        Clear investigation context while preserving conversation_id.
        """

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        self._ensure_session_matches_runtime(
            session_id=session_id,
            runtime=runtime,
        )

        self._session_manager.clear_context(
            session_id,
            runtime=runtime,
        )

        return (
            self._session_manager.snapshot(
                session_id
            )
        )

    def new_conversation(
        self,
        repository_root: Path,
        session_id: str,
    ) -> ConversationSessionSnapshot:
        """
        Create a new Phase-9 conversation identity inside the same
        application session.
        """

        runtime = (
            self._runtime_registry
            .get_or_create(
                repository_root
            )
        )

        self._ensure_session_matches_runtime(
            session_id=session_id,
            runtime=runtime,
        )

        self._session_manager.new_conversation(
            session_id,
            runtime=runtime,
        )

        return (
            self._session_manager.snapshot(
                session_id
            )
        )

    # ==================================================================
    # Runtime/session safety
    # ==================================================================

    def _ensure_session_matches_runtime(
        self,
        *,
        session_id: str,
        runtime: RepositoryRuntime,
    ) -> None:
        """
        Reject sessions from an older or different runtime.

        This is intentionally explicit rather than allowing the
        ConversationSessionManager mismatch error to leak through the
        application boundary.
        """

        try:

            snapshot = (
                self._session_manager
                .snapshot(
                    session_id
                )
            )

        except ConversationSessionNotFoundError:
            raise

        if (
            snapshot.runtime_id
            != runtime.runtime_id
        ):

            raise (
                RepositorySessionStaleError(
                    "Conversation session is stale for the current "
                    "repository runtime. Create a new session."
                )
            )


