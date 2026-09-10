from __future__ import annotations

from repobrain.application import (
    RepositoryRuntime,
    RepositoryRuntimeRegistry,
)


class FakeAgentRunner:

    def __call__(
        self,
        query: str,
    ) -> object:

        raise AssertionError(
            "Agent should not run in registry tests."
        )


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> object:

        raise AssertionError(
            "Answer generator should not run "
            "in registry tests."
        )


class RecordingRuntimeFactory:

    def __init__(
        self,
    ) -> None:

        self.calls = []

    def __call__(
        self,
        repository_root,
    ) -> RepositoryRuntime:

        self.calls.append(
            repository_root
        )

        return RepositoryRuntime(
            repository_root=(
                repository_root
            ),
            agent_runner=(
                FakeAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
        )


def test_same_repository_reuses_runtime(
    tmp_path,
) -> None:

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=(
                factory
            )
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

    assert (
        len(factory.calls)
        == 1
    )

    assert len(registry) == 1


def test_different_repositories_get_different_runtimes(
    tmp_path,
) -> None:

    repository_one = (
        tmp_path
        / "repo-one"
    )

    repository_two = (
        tmp_path
        / "repo-two"
    )

    repository_one.mkdir()

    repository_two.mkdir()

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    first = (
        registry.get_or_create(
            repository_one
        )
    )

    second = (
        registry.get_or_create(
            repository_two
        )
    )

    assert (
        first is not second
    )

    assert (
        first.runtime_id
        != second.runtime_id
    )

    assert (
        len(factory.calls)
        == 2
    )

    assert len(registry) == 2


def test_get_does_not_create_runtime(
    tmp_path,
) -> None:

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    result = (
        registry.get(
            tmp_path
        )
    )

    assert result is None

    assert (
        factory.calls
        == []
    )


def test_contains_reports_loaded_repository(
    tmp_path,
) -> None:

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    assert (
        registry.contains(
            tmp_path
        )
        is False
    )

    registry.get_or_create(
        tmp_path
    )

    assert (
        registry.contains(
            tmp_path
        )
        is True
    )


def test_evict_closes_runtime(
    tmp_path,
) -> None:

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    runtime = (
        registry.get_or_create(
            tmp_path
        )
    )

    removed = (
        registry.evict(
            tmp_path
        )
    )

    assert removed is True

    assert (
        runtime.closed
        is True
    )

    assert (
        registry.contains(
            tmp_path
        )
        is False
    )

    assert len(registry) == 0


def test_get_or_create_rebuilds_after_eviction(
    tmp_path,
) -> None:

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    first = (
        registry.get_or_create(
            tmp_path
        )
    )

    registry.evict(
        tmp_path
    )

    second = (
        registry.get_or_create(
            tmp_path
        )
    )

    assert (
        first is not second
    )

    assert (
        first.runtime_id
        != second.runtime_id
    )

    assert (
        len(factory.calls)
        == 2
    )


def test_clear_closes_every_runtime(
    tmp_path,
) -> None:

    repository_one = (
        tmp_path
        / "one"
    )

    repository_two = (
        tmp_path
        / "two"
    )

    repository_one.mkdir()

    repository_two.mkdir()

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    first = (
        registry.get_or_create(
            repository_one
        )
    )

    second = (
        registry.get_or_create(
            repository_two
        )
    )

    registry.clear()

    assert (
        first.closed
        is True
    )

    assert (
        second.closed
        is True
    )

    assert len(registry) == 0


def test_snapshots_are_deterministic(
    tmp_path,
) -> None:

    repository_b = (
        tmp_path
        / "b-repository"
    )

    repository_a = (
        tmp_path
        / "a-repository"
    )

    repository_b.mkdir()

    repository_a.mkdir()

    factory = (
        RecordingRuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory
        )
    )

    registry.get_or_create(
        repository_b
    )

    registry.get_or_create(
        repository_a
    )

    snapshots = (
        registry.snapshots()
    )

    assert len(
        snapshots
    ) == 2

    assert (
        snapshots[0]
        .repository_root
        .casefold()
        <
        snapshots[1]
        .repository_root
        .casefold()
    )