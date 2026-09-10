from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from repobrain.application import (
    RepositoryRuntime,
    RepositoryRuntimeRegistry,
)
from repobrain.models.application import (
    RepositoryFingerprint,
)


class FakeAgentRunner:

    def __call__(
        self,
        query: str,
    ) -> object:

        raise AssertionError(
            "Agent should not run."
        )


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> object:

        raise AssertionError(
            "Answer generator should not run."
        )


def file_fingerprint(
    repository_root: Path,
) -> RepositoryFingerprint:

    content = (
        repository_root
        .joinpath(
            "tracked.txt"
        )
        .read_bytes()
    )

    digest = (
        hashlib.sha256(
            content
        )
        .hexdigest()
    )

    return RepositoryFingerprint(
        value=digest,
        file_count=1,
        total_size_bytes=len(content),
    )


class FingerprintedRuntimeFactory:

    def __init__(
        self,
    ) -> None:

        self.calls = 0

    def __call__(
        self,
        repository_root: Path,
    ) -> RepositoryRuntime:

        self.calls += 1

        return RepositoryRuntime(
            repository_root=repository_root,
            agent_runner=(
                FakeAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
            fingerprint=(
                file_fingerprint(
                    repository_root
                )
            ),
        )


def test_unchanged_repository_reuses_runtime(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "tracked.txt"
    ).write_text(
        "version-one",
        encoding="utf-8",
    )

    factory = (
        FingerprintedRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                file_fingerprint
            ),
        )
    )

    first = (
        registry.get_or_create(
            tmp_path
        )
    )

    second = (
        registry.get_or_create(
            tmp_path
        )
    )

    assert (
        first is second
    )

    assert factory.calls == 1


def test_changed_repository_rebuilds_runtime(
    tmp_path: Path,
) -> None:

    tracked = (
        tmp_path
        / "tracked.txt"
    )

    tracked.write_text(
        "version-one",
        encoding="utf-8",
    )

    factory = (
        FingerprintedRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                file_fingerprint
            ),
        )
    )

    first = (
        registry.get_or_create(
            tmp_path
        )
    )

    tracked.write_text(
        "version-two",
        encoding="utf-8",
    )

    second = (
        registry.get_or_create(
            tmp_path
        )
    )

    assert (
        second is not first
    )

    assert (
        first.closed
        is True
    )

    assert (
        second.closed
        is False
    )

    assert factory.calls == 2


def test_get_evicts_stale_runtime(
    tmp_path: Path,
) -> None:

    tracked = (
        tmp_path
        / "tracked.txt"
    )

    tracked.write_text(
        "before",
        encoding="utf-8",
    )

    factory = (
        FingerprintedRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                file_fingerprint
            ),
        )
    )

    runtime = (
        registry.get_or_create(
            tmp_path
        )
    )

    tracked.write_text(
        "after",
        encoding="utf-8",
    )

    result = (
        registry.get(
            tmp_path
        )
    )

    assert result is None

    assert (
        runtime.closed
        is True
    )

    assert len(registry) == 0


def test_fingerprint_aware_registry_rejects_unfingerprinted_runtime(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "tracked.txt"
    ).write_text(
        "value",
        encoding="utf-8",
    )

    def factory(
        repository_root: Path,
    ) -> RepositoryRuntime:

        return RepositoryRuntime(
            repository_root=repository_root,
            agent_runner=(
                FakeAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
        )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                file_fingerprint
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "requires runtime_factory "
            "to return a runtime "
            "with a repository fingerprint"
        ),
    ):

        registry.get_or_create(
            tmp_path
        )


def test_runtime_snapshot_exposes_build_fingerprint(
    tmp_path: Path,
) -> None:

    tracked = (
        tmp_path
        / "tracked.txt"
    )

    tracked.write_text(
        "snapshot",
        encoding="utf-8",
    )

    fingerprint = (
        file_fingerprint(
            tmp_path
        )
    )

    runtime = RepositoryRuntime(
        repository_root=tmp_path,
        agent_runner=(
            FakeAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
        fingerprint=(
            fingerprint
        ),
    )

    snapshot = (
        runtime.snapshot()
    )

    assert (
        snapshot.fingerprint_algorithm
        == fingerprint.algorithm
    )

    assert (
        snapshot.fingerprint_value
        == fingerprint.value
    )

    assert (
        snapshot.fingerprint_file_count
        == fingerprint.file_count
    )
