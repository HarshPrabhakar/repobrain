from __future__ import annotations

from pathlib import Path

from repobrain.application.registry import (
    RepositoryRuntimeRegistry,
)
from repobrain.application.sessions import (
    ConversationSessionManager,
)
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationHealthSnapshot,
    RepositoryDiagnosticsSnapshot,
    SessionDiagnosticsSnapshot,
)


class ApplicationDiagnosticsService:
    """
    Read-only diagnostics facade for RepoBrain's application layer.

    It does not perform retrieval, mutate repositories, or call the LLM.
    """

    def __init__(
        self,
        *,
        runtime_registry: RepositoryRuntimeRegistry,
        session_manager: ConversationSessionManager,
    ) -> None:

        self._runtime_registry = (
            runtime_registry
        )

        self._session_manager = (
            session_manager
        )

    def health(
        self,
    ) -> ApplicationHealthSnapshot:
        """
        Return lightweight process-local readiness information.
        """

        loaded_repositories = (
            len(
                self._runtime_registry
            )
        )

        active_sessions = (
            len(
                self._session_manager
            )
        )

        return (
            ApplicationHealthSnapshot(
                ready=True,
                loaded_repositories=(
                    loaded_repositories
                ),
                active_sessions=(
                    active_sessions
                ),
                status="ready",
            )
        )

    def diagnostics(
        self,
    ) -> ApplicationDiagnosticsSnapshot:
        """
        Return a complete deterministic application diagnostic snapshot.
        """

        repository_snapshots = (
            self._runtime_registry
            .snapshots()
        )

        repositories = []

        for runtime_snapshot in repository_snapshots:

            runtime = (
                self._runtime_registry.get(
                    Path(
                        runtime_snapshot.repository_root
                    )
                )
            )

            if runtime is None:
                continue

            active_sessions = (
                len(
                    self._session_manager
                    .snapshots(
                        runtime=runtime
                    )
                )
            )

            repositories.append(
                RepositoryDiagnosticsSnapshot(
                    runtime=(
                        runtime.snapshot()
                    ),
                    runtime_diagnostics=(
                        runtime.diagnostics
                    ),
                    active_sessions=(
                        active_sessions
                    ),
                )
            )

        repositories.sort(
            key=lambda item: (
                item.runtime.repository_root
                .casefold()
            )
        )

        sessions = (
            self._session_manager
            .snapshots()
        )

        return (
            ApplicationDiagnosticsSnapshot(
                health=(
                    self.health()
                ),
                repositories=(
                    tuple(
                        repositories
                    )
                ),
                sessions=(
                    sessions
                ),
            )
        )

    def repository_diagnostics(
        self,
        repository_root: Path,
    ) -> RepositoryDiagnosticsSnapshot | None:
        """
        Return diagnostics for one currently valid repository runtime.

        Fingerprint validation is performed by RepositoryRuntimeRegistry.get().
        """

        runtime = (
            self._runtime_registry.get(
                repository_root
            )
        )

        if runtime is None:
            return None

        active_sessions = (
            len(
                self._session_manager
                .snapshots(
                    runtime=runtime
                )
            )
        )

        return (
            RepositoryDiagnosticsSnapshot(
                runtime=(
                    runtime.snapshot()
                ),
                runtime_diagnostics=(
                    runtime.diagnostics
                ),
                active_sessions=(
                    active_sessions
                ),
            )
        )

    def session_diagnostics(
        self,
        session_id: str,
    ) -> SessionDiagnosticsSnapshot:
        """
        Return diagnostics for one application conversation session.

        runtime_current is True only when the session is still attached
        to the currently valid runtime for its repository.
        """

        session = (
            self._session_manager.snapshot(
                session_id
            )
        )

        runtime = (
            self._runtime_registry.get(
                Path(
                    session.repository_root
                )
            )
        )

        runtime_loaded = (
            runtime is not None
        )

        runtime_current = (
            runtime is not None
            and runtime.runtime_id
            == session.runtime_id
        )

        return (
            SessionDiagnosticsSnapshot(
                session=session,
                runtime_loaded=(
                    runtime_loaded
                ),
                runtime_current=(
                    runtime_current
                ),
            )
        )
