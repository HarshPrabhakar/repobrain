from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import (
    Callable,
)

from repobrain.application.runtime import (
    RepositoryRuntime,
)

from repobrain.models.application import (
    RepositoryRuntimeSnapshot,
)


RepositoryRuntimeFactory = Callable[
    [Path],
    RepositoryRuntime,
]


class RepositoryRuntimeRegistry:
    """
    Process-local registry of loaded repository runtimes.

    Exactly one runtime is retained per canonical repository path.

    This gives RepoBrain its first application-level cache:

        same repository
            -> same RepositoryRuntime

        different repository
            -> different RepositoryRuntime

    Conversation state is NOT cached here.
    Each runtime creates independent conversations.
    """

    def __init__(
        self,
        *,
        runtime_factory: RepositoryRuntimeFactory,
    ) -> None:

        self._runtime_factory = (
            runtime_factory
        )

        self._runtimes: dict[
            str,
            RepositoryRuntime,
        ] = {}

        self._lock = (
            RLock()
        )

    # ==================================================================
    # Repository lifecycle
    # ==================================================================

    def get_or_create(
        self,
        repository_root: Path,
    ) -> RepositoryRuntime:
        """
        Return the cached runtime for a repository or build it once.

        Path comparison is case-insensitive so the registry behaves
        safely on the user's Windows environment.
        """

        normalized_root = (
            self._normalize_root(
                repository_root
            )
        )

        key = (
            self._repository_key(
                normalized_root
            )
        )

        with self._lock:

            existing = (
                self._runtimes.get(
                    key
                )
            )

            if (
                existing is not None
                and not existing.closed
            ):

                return existing

            runtime = (
                self._runtime_factory(
                    normalized_root
                )
            )

            if (
                runtime.repository_root
                != normalized_root
            ):

                raise RuntimeError(
                    "Runtime factory returned a runtime "
                    "for a different repository."
                )

            self._runtimes[
                key
            ] = runtime

            return runtime

    def get(
        self,
        repository_root: Path,
    ) -> RepositoryRuntime | None:
        """
        Return a currently loaded runtime without creating one.
        """

        normalized_root = (
            self._normalize_root(
                repository_root
            )
        )

        key = (
            self._repository_key(
                normalized_root
            )
        )

        with self._lock:

            runtime = (
                self._runtimes.get(
                    key
                )
            )

            if (
                runtime is None
                or runtime.closed
            ):
                return None

            return runtime

    def contains(
        self,
        repository_root: Path,
    ) -> bool:

        return (
            self.get(
                repository_root
            )
            is not None
        )

    def evict(
        self,
        repository_root: Path,
    ) -> bool:
        """
        Remove and close one loaded repository runtime.

        Returns True when a runtime existed.
        """

        normalized_root = (
            self._normalize_root(
                repository_root
            )
        )

        key = (
            self._repository_key(
                normalized_root
            )
        )

        with self._lock:

            runtime = (
                self._runtimes.pop(
                    key,
                    None,
                )
            )

        if runtime is None:
            return False

        runtime.close()

        return True

    def clear(
        self,
    ) -> None:
        """
        Close and remove every loaded runtime.
        """

        with self._lock:

            runtimes = tuple(
                self._runtimes.values()
            )

            self._runtimes.clear()

        for runtime in runtimes:

            runtime.close()

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def snapshots(
        self,
    ) -> tuple[
        RepositoryRuntimeSnapshot,
        ...,
    ]:
        """
        Return deterministic snapshots of loaded runtimes.
        """

        with self._lock:

            runtimes = tuple(
                self._runtimes.values()
            )

        active = [
            runtime.snapshot()
            for runtime in runtimes
            if not runtime.closed
        ]

        active.sort(
            key=lambda item: (
                item.repository_root
                .casefold()
            )
        )

        return tuple(
            active
        )

    def __len__(
        self,
    ) -> int:

        with self._lock:

            return sum(
                1
                for runtime
                in self._runtimes.values()
                if not runtime.closed
            )

    # ==================================================================
    # Path helpers
    # ==================================================================

    @staticmethod
    def _normalize_root(
        repository_root: Path,
    ) -> Path:

        normalized = (
            repository_root
            .expanduser()
            .resolve()
        )

        if not normalized.exists():

            raise ValueError(
                "repository_root does not exist: "
                f"{normalized}"
            )

        if not normalized.is_dir():

            raise ValueError(
                "repository_root is not a directory: "
                f"{normalized}"
            )

        return normalized

    @staticmethod
    def _repository_key(
        repository_root: Path,
    ) -> str:
        """
        Case-insensitive canonical cache key.

        This intentionally matches Windows repository path behavior.
        """

        return (
            str(
                repository_root
            )
            .casefold()
        )