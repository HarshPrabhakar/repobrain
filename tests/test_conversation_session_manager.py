from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from repobrain.application import (
    ConversationSessionManager,
    ConversationSessionNotFoundError,
    ConversationSessionRuntimeMismatchError,
    RepositoryRuntime,
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
    graph_facts: tuple[object, ...] = ()


class RecordingAgentRunner:

    def __init__(
        self,
    ) -> None:

        self.queries: list[str] = []

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


class FakeAnswer:

    answer_text = "ok"
    citations = ()
    invalid_citation_ids = ()
    grounded = True


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> FakeAnswer:

        return FakeAnswer()


def make_runtime(
    root: Path,
) -> RepositoryRuntime:

    return RepositoryRuntime(
        repository_root=root,
        agent_runner=(
            RecordingAgentRunner()
        ),
        answer_generator=(
            FakeAnswerGenerator()
        ),
    )


def test_create_session_returns_unique_ids(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    first = (
        manager.create(
            runtime
        )
    )

    second = (
        manager.create(
            runtime
        )
    )

    assert first != second

    assert len(manager) == 2

    assert (
        runtime.conversation_count
        == 2
    )


def test_sessions_have_independent_conversation_ids(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    first = (
        manager.create(
            runtime
        )
    )

    second = (
        manager.create(
            runtime
        )
    )

    first_snapshot = (
        manager.snapshot(
            first
        )
    )

    second_snapshot = (
        manager.snapshot(
            second
        )
    )

    assert (
        first_snapshot.conversation_id
        != second_snapshot.conversation_id
    )


def test_get_returns_same_conversation_object(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime
        )
    )

    first = (
        manager.get(
            session_id
        )
    )

    second = (
        manager.get(
            session_id
        )
    )

    assert first is second


def test_unknown_session_is_rejected() -> None:

    manager = (
        ConversationSessionManager()
    )

    with pytest.raises(
        ConversationSessionNotFoundError,
        match="Unknown conversation session",
    ):

        manager.get(
            "missing"
        )


def test_empty_session_id_is_rejected() -> None:

    manager = (
        ConversationSessionManager()
    )

    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):

        manager.get(
            "   "
        )


def test_session_runtime_mismatch_is_rejected(
    tmp_path: Path,
) -> None:

    first_root = (
        tmp_path
        / "one"
    )

    second_root = (
        tmp_path
        / "two"
    )

    first_root.mkdir()
    second_root.mkdir()

    runtime_one = (
        make_runtime(
            first_root
        )
    )

    runtime_two = (
        make_runtime(
            second_root
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime_one
        )
    )

    with pytest.raises(
        ConversationSessionRuntimeMismatchError,
        match="different repository runtime",
    ):

        manager.get(
            session_id,
            runtime=runtime_two,
        )


def test_delete_session(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime
        )
    )

    assert (
        manager.delete(
            session_id
        )
        is True
    )

    assert (
        manager.contains(
            session_id
        )
        is False
    )

    assert (
        manager.delete(
            session_id
        )
        is False
    )


def test_clear_all_sessions(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    manager.create(
        runtime
    )

    manager.create(
        runtime
    )

    count = (
        manager.clear()
    )

    assert count == 2

    assert len(manager) == 0


def test_clear_only_sessions_for_one_runtime(
    tmp_path: Path,
) -> None:

    first_root = (
        tmp_path
        / "one"
    )

    second_root = (
        tmp_path
        / "two"
    )

    first_root.mkdir()
    second_root.mkdir()

    runtime_one = (
        make_runtime(
            first_root
        )
    )

    runtime_two = (
        make_runtime(
            second_root
        )
    )

    manager = (
        ConversationSessionManager()
    )

    first_session = (
        manager.create(
            runtime_one
        )
    )

    second_session = (
        manager.create(
            runtime_two
        )
    )

    removed = (
        manager.clear(
            runtime=runtime_one
        )
    )

    assert removed == 1

    assert (
        manager.contains(
            first_session
        )
        is False
    )

    assert (
        manager.contains(
            second_session
        )
        is True
    )


def test_new_conversation_preserves_session_id(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime
        )
    )

    before = (
        manager.snapshot(
            session_id
        )
    )

    manager.new_conversation(
        session_id
    )

    after = (
        manager.snapshot(
            session_id
        )
    )

    assert (
        before.session_id
        == after.session_id
        == session_id
    )

    assert (
        before.conversation_id
        != after.conversation_id
    )

    assert (
        after.turn_number
        == 0
    )


def test_clear_context_preserves_conversation_id(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime
        )
    )

    before = (
        manager.snapshot(
            session_id
        )
    )

    manager.clear_context(
        session_id
    )

    after = (
        manager.snapshot(
            session_id
        )
    )

    assert (
        before.conversation_id
        == after.conversation_id
    )


def test_snapshots_are_sorted_by_session_id(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    manager.create(
        runtime
    )

    manager.create(
        runtime
    )

    snapshots = (
        manager.snapshots()
    )

    assert len(
        snapshots
    ) == 2

    ids = [
        item.session_id
        for item in snapshots
    ]

    assert ids == sorted(
        ids
    )


def test_snapshot_contains_runtime_identity(
    tmp_path: Path,
) -> None:

    runtime = (
        make_runtime(
            tmp_path
        )
    )

    manager = (
        ConversationSessionManager()
    )

    session_id = (
        manager.create(
            runtime
        )
    )

    snapshot = (
        manager.snapshot(
            session_id
        )
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
        snapshot.turn_number
        == 0
    )
