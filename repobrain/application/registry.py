from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Callable

from repobrain.application.runtime import (
    RepositoryRuntime,
)

from repobrain.models.application import (
    RepositoryFingerprint,
    RepositoryRuntimeSnapshot,
)


RepositoryRuntimeFactory = Callable[
    [Path],
    RepositoryRuntime,
]

RepositoryFingerprintProvider = Callable[
    [Path],
    RepositoryFingerprint,
]


class RepositoryRuntimeRegistry:
    """
    Process-local registry of loaded repository runtimes.

    Phase 10.3 adds content-aware invalidation.

    When a fingerprint provider is configured:

        unchanged repository
            -> reuse current runtime

        changed repository
            -> close stale runtime
            -> rebuild runtime

    Without a fingerprint provider, Phase 10.1 path-only reuse remains
    available for backward compatibility and isolated tests.
    """

    def __init__(
        self,
        *,
        runtime_factory: RepositoryRuntimeFactory,
        fingerprint_provider: (
            RepositoryFingerprintProvider
            | None
        ) = None,
    ) -> None:

        self._runtime_factory = (
            runtime_factory
        )

        self._fingerprint_provider = (
            fingerprint_provider
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
        Return a valid cached runtime or build a new one.

        If fingerprint validation is enabled, stale runtimes are never
        returned.
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

                if self._runtime_is_current(
                    existing,
                    normalized_root,
                ):

                    return existing

                self._runtimes.pop(
                    key,
                    None,
                )

                existing.close()

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

            if (
                self._fingerprint_provider
                is not None
                and runtime.fingerprint
                is None
            ):

                runtime.close()

                raise RuntimeError(
                    "Fingerprint-aware registry requires "
                    "runtime_factory to return a runtime "
                    "with a repository fingerprint."
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
        Return a loaded runtime without creating one.

        When fingerprint validation is configured, get() also refuses
        stale runtimes and evicts them.
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

            if not self._runtime_is_current(
                runtime,
                normalized_root,
            ):

                self._runtimes.pop(
                    key,
                    None,
                )

                runtime.close()

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
    # Fingerprint validation
    # ==================================================================

    def _runtime_is_current(
        self,
        runtime: RepositoryRuntime,
        repository_root: Path,
    ) -> bool:
        """
        Return True only when the runtime fingerprint still matches the
        current repository content.

        A runtime with no fingerprint is considered stale when the
        registry is operating in fingerprint-aware mode.
        """

        provider = (
            self._fingerprint_provider
        )

        if provider is None:
            return True

        runtime_fingerprint = (
            runtime.fingerprint
        )

        if runtime_fingerprint is None:
            return False

        current_fingerprint = (
            provider(
                repository_root
            )
        )

        return (
            current_fingerprint
            == runtime_fingerprint
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

        return (
            str(
                repository_root
            )
            .casefold()
        )
