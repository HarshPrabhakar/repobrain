from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Protocol

from repobrain.conversation import (
    StatefulGroundedAnsweringOrchestrator,
    StatefulRepoBrainOrchestrator,
)

from repobrain.models.application import (
    RepositoryFingerprint,
    RepositoryRuntimeSnapshot,
)


class AgentRunner(
    Protocol
):
    """
    Minimal interface required from the existing
    deterministic RepoBrain agent.
    """

    def __call__(
        self,
        query: str,
    ) -> object:
        ...


class AnswerGenerator(
    Protocol
):
    """
    Minimal interface required from RepoBrain's grounded
    answer generator.
    """

    def generate(
        self,
        result: object,
    ) -> object:
        ...


class RepositoryRuntimeClosedError(
    RuntimeError
):
    """
    Raised when code attempts to create a conversation
    from a runtime that has already been closed.
    """


class RepositoryRuntime:
    """
    Long-lived application runtime for one repository.

    Repository lifetime and conversation lifetime are deliberately
    separate.

    Phase 10.3 additionally attaches the repository content fingerprint
    that was valid when this runtime was built.
    """

    def __init__(
        self,
        *,
        repository_root: Path,
        agent_runner: AgentRunner,
        answer_generator: AnswerGenerator,
        fingerprint: RepositoryFingerprint | None = None,
    ) -> None:

        normalized_root = (
            repository_root
            .expanduser()
            .resolve()
        )

        if not normalized_root.exists():

            raise ValueError(
                "repository_root does not exist: "
                f"{normalized_root}"
            )

        if not normalized_root.is_dir():

            raise ValueError(
                "repository_root is not a directory: "
                f"{normalized_root}"
            )

        self._repository_root = (
            normalized_root
        )

        self._agent_runner = (
            agent_runner
        )

        self._answer_generator = (
            answer_generator
        )

        self._fingerprint = (
            fingerprint
        )

        self._runtime_id = (
            RepositoryRuntimeSnapshot(
                repository_root=(
                    str(
                        normalized_root
                    )
                )
            )
            .runtime_id
        )

        self._created_snapshot = (
            RepositoryRuntimeSnapshot(
                runtime_id=(
                    self._runtime_id
                ),
                repository_root=(
                    str(
                        normalized_root
                    )
                ),
                fingerprint_algorithm=(
                    fingerprint.algorithm
                    if fingerprint is not None
                    else None
                ),
                fingerprint_value=(
                    fingerprint.value
                    if fingerprint is not None
                    else None
                ),
                fingerprint_file_count=(
                    fingerprint.file_count
                    if fingerprint is not None
                    else None
                ),
            )
        )

        self._conversation_count = 0

        self._closed = False

        self._lock = (
            RLock()
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def runtime_id(
        self,
    ) -> str:

        return (
            self._runtime_id
        )

    @property
    def repository_root(
        self,
    ) -> Path:

        return (
            self._repository_root
        )

    @property
    def fingerprint(
        self,
    ) -> RepositoryFingerprint | None:
        """
        Fingerprint that was valid when this runtime was built.
        """

        return (
            self._fingerprint
        )

    @property
    def conversation_count(
        self,
    ) -> int:

        with self._lock:

            return (
                self._conversation_count
            )

    @property
    def closed(
        self,
    ) -> bool:

        with self._lock:

            return (
                self._closed
            )

    # ==================================================================
    # Conversation construction
    # ==================================================================

    def create_conversation(
        self,
    ) -> StatefulGroundedAnsweringOrchestrator:
        """
        Create one independent Phase-9 conversation while reusing this
        runtime's repository intelligence.
        """

        with self._lock:

            if self._closed:

                raise (
                    RepositoryRuntimeClosedError(
                        "Cannot create a conversation "
                        "from a closed repository runtime."
                    )
                )

            stateful_agent = (
                StatefulRepoBrainOrchestrator(
                    agent_runner=(
                        self._agent_runner
                    )
                )
            )

            conversation = (
                StatefulGroundedAnsweringOrchestrator(
                    stateful_agent=(
                        stateful_agent
                    ),
                    answer_generator=(
                        self._answer_generator
                    ),
                )
            )

            self._conversation_count += 1

            return conversation

    # ==================================================================
    # Runtime lifecycle
    # ==================================================================

    def close(
        self,
    ) -> None:

        with self._lock:

            self._closed = True

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def snapshot(
        self,
    ) -> RepositoryRuntimeSnapshot:

        with self._lock:

            fingerprint = (
                self._fingerprint
            )

            return (
                RepositoryRuntimeSnapshot(
                    runtime_id=(
                        self._runtime_id
                    ),
                    repository_root=(
                        str(
                            self._repository_root
                        )
                    ),
                    created_at=(
                        self._created_snapshot
                        .created_at
                    ),
                    conversation_count=(
                        self._conversation_count
                    ),
                    closed=(
                        self._closed
                    ),
                    fingerprint_algorithm=(
                        fingerprint.algorithm
                        if fingerprint is not None
                        else None
                    ),
                    fingerprint_value=(
                        fingerprint.value
                        if fingerprint is not None
                        else None
                    ),
                    fingerprint_file_count=(
                        fingerprint.file_count
                        if fingerprint is not None
                        else None
                    ),
                )
            )
