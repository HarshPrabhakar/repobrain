from __future__ import annotations

from dataclasses import dataclass

import pytest

from repobrain.application import (
    RepositoryRuntime,
    RepositoryRuntimeClosedError,
)

from repobrain.models.agent import (
    AgentIntent,
)


@dataclass
class FakeAgentState:
    intent: AgentIntent


@dataclass
class FakeAgentResult:
    state: FakeAgentState

    resolved_symbol_id: str | None = None

    resolved_qualified_name: str | None = None

    graph_facts: tuple[
        object,
        ...,
    ] = ()


class RecordingAgentRunner:

    def __init__(
        self,
    ) -> None:

        self.queries: list[
            str
        ] = []

    def __call__(
        self,
        query: str,
    ) -> FakeAgentResult:

        self.queries.append(
            query
        )

        return FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> object:

        return result


def test_runtime_normalizes_repository_root(
    tmp_path,
) -> None:

    runtime = RepositoryRuntime(
        repository_root=(
            tmp_path
        ),
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )

    assert (
        runtime.repository_root
        == tmp_path.resolve()
    )


def test_runtime_starts_open(
    tmp_path,
) -> None:

    runtime = RepositoryRuntime(
        repository_root=tmp_path,
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )

    assert (
        runtime.closed
        is False
    )

    assert (
        runtime.conversation_count
        == 0
    )


def test_create_conversation_returns_fresh_state(
    tmp_path,
) -> None:

    runtime = RepositoryRuntime(
        repository_root=tmp_path,
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )

    first = (
        runtime.create_conversation()
    )

    second = (
        runtime.create_conversation()
    )

    assert (
        first is not second
    )

    assert (
        first.state
        .conversation_id
        != second.state
        .conversation_id
    )

    assert (
        first.state.turn_number
        == 0
    )

    assert (
        second.state.turn_number
        == 0
    )

    assert (
        runtime.conversation_count
        == 2
    )


def test_runtime_snapshot_tracks_conversations(
    tmp_path,
) -> None:

    runtime = RepositoryRuntime(
        repository_root=tmp_path,
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )

    runtime.create_conversation()

    runtime.create_conversation()

    snapshot = (
        runtime.snapshot()
    )

    assert (
        snapshot.runtime_id
        == runtime.runtime_id
    )

    assert (
        snapshot.repository_root
        == str(
            tmp_path.resolve()
        )
    )

    assert (
        snapshot.conversation_count
        == 2
    )

    assert (
        snapshot.closed
        is False
    )


def test_closed_runtime_rejects_new_conversations(
    tmp_path,
) -> None:

    runtime = RepositoryRuntime(
        repository_root=tmp_path,
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )

    runtime.close()

    assert (
        runtime.closed
        is True
    )

    with pytest.raises(
        RepositoryRuntimeClosedError,
        match=(
            "closed repository runtime"
        ),
    ):

        runtime.create_conversation()


def test_missing_repository_is_rejected(
    tmp_path,
) -> None:

    missing = (
        tmp_path
        / "missing"
    )

    with pytest.raises(
        ValueError,
        match=(
            "repository_root does not exist"
        ),
    ):

        RepositoryRuntime(
            repository_root=(
                missing
            ),
            agent_runner=(
                RecordingAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
        )


def test_repository_file_is_rejected(
    tmp_path,
) -> None:

    file_path = (
        tmp_path
        / "not-a-repository.txt"
    )

    file_path.write_text(
        "test",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "repository_root is not a directory"
        ),
    ):

        RepositoryRuntime(
            repository_root=(
                file_path
            ),
            agent_runner=(
                RecordingAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
        )