from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import (
    Callable,
    Protocol,
)

from repobrain.conversation import (
    StatefulGroundedAnsweringOrchestrator,
    StatefulRepoBrainOrchestrator,
)

from repobrain.models.application import (
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

    The runtime owns the expensive, reusable repository
    intelligence dependencies indirectly through:

        agent_runner
        answer_generator

    New conversations receive fresh Phase-9 conversation
    state while reusing those shared dependencies.

    This is the key separation introduced by Phase 10:

        repository lifetime != conversation lifetime

    The runtime does not mutate repository files.
    """

    def __init__(
        self,
        *,
        repository_root: Path,
        agent_runner: AgentRunner,
        answer_generator: AnswerGenerator,
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
        """
        Stable ID for the lifetime of this loaded runtime.
        """

        return (
            self._runtime_id
        )

    @property
    def repository_root(
        self,
    ) -> Path:
        """
        Canonical repository root owned by this runtime.
        """

        return (
            self._repository_root
        )

    @property
    def conversation_count(
        self,
    ) -> int:
        """
        Number of conversations created from this runtime.
        """

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
        Create one independent Phase-9 conversation.

        Expensive repository intelligence is reused through the
        shared agent runner and answer generator.

        Conversation state itself is fresh.
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
        """
        Mark this repository runtime as closed.

        Phase 10.1 does not destroy model/index objects manually;
        ownership cleanup will be expanded when persistent caching
        and lifecycle resources are introduced.
        """

        with self._lock:

            self._closed = True

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def snapshot(
        self,
    ) -> RepositoryRuntimeSnapshot:
        """
        Return immutable runtime metadata.
        """

        with self._lock:

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
                )
            )