from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import pytest

from repobrain.application import (
    ConversationSessionManager,
    RepoBrainApplicationService,
    RepositoryRuntime,
    RepositoryRuntimeRegistry,
    RepositorySessionStaleError,
)
from repobrain.models.application import RepositoryFingerprint
from repobrain.models.agent import AgentIntent


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
    def __init__(self) -> None:
        self.queries: list[str] = []

    def __call__(self, query: str) -> FakeAgentResult:
        self.queries.append(query)
        return FakeAgentResult(
            state=FakeAgentState(intent=AgentIntent.GENERAL)
        )


class FakeAnswer:
    answer_text = "ok"
    citations = ()
    invalid_citation_ids = ()
    grounded = True


class FakeAnswerGenerator:
    def generate(self, result: object) -> FakeAnswer:
        return FakeAnswer()


def fingerprint_for(root: Path) -> RepositoryFingerprint:
    value = (root / "marker.txt").read_text(encoding="utf-8")
    payload = value.encode("utf-8")
    return RepositoryFingerprint(
        value=hashlib.sha256(payload).hexdigest(),
        file_count=1,
        total_size_bytes=len(payload),
    )


class RuntimeFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, root: Path) -> RepositoryRuntime:
        self.calls += 1
        return RepositoryRuntime(
            repository_root=root,
            agent_runner=RecordingAgentRunner(),
            answer_generator=FakeAnswerGenerator(),
            fingerprint=fingerprint_for(root),
        )


def make_service(root: Path):
    factory = RuntimeFactory()
    registry = RepositoryRuntimeRegistry(
        runtime_factory=factory,
        fingerprint_provider=fingerprint_for,
    )
    service = RepoBrainApplicationService(
        runtime_registry=registry,
        session_manager=ConversationSessionManager(),
    )
    return service, factory


def test_open_repository_reuses_current_runtime(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, factory = make_service(tmp_path)

    first = service.open_repository(tmp_path)
    second = service.open_repository(tmp_path)

    assert first.runtime_id == second.runtime_id
    assert factory.calls == 1


def test_create_session_opens_repository(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)

    assert session.repository_root == str(tmp_path.resolve())
    assert session.turn_number == 0


def test_two_sessions_are_independent(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    first = service.create_session(tmp_path)
    second = service.create_session(tmp_path)

    assert first.session_id != second.session_id
    assert first.conversation_id != second.conversation_id
    assert first.runtime_id == second.runtime_id


def test_get_state_returns_phase9_state(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)
    state = service.get_state(tmp_path, session.session_id)

    assert state.conversation_id == session.conversation_id
    assert state.turn_number == 0


def test_clear_session_preserves_conversation_id(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)
    cleared = service.clear_session(tmp_path, session.session_id)

    assert cleared.session_id == session.session_id
    assert cleared.conversation_id == session.conversation_id


def test_new_conversation_preserves_session_id(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)
    reset = service.new_conversation(tmp_path, session.session_id)

    assert reset.session_id == session.session_id
    assert reset.conversation_id != session.conversation_id


def test_delete_session(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)

    assert service.delete_session(session.session_id) is True
    assert service.delete_session(session.session_id) is False


def test_close_repository_deletes_its_sessions(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)

    assert service.close_repository(tmp_path) is True
    assert service.session_manager.contains(session.session_id) is False
    assert len(service.runtime_registry) == 0


def test_changed_repository_makes_old_session_stale(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("one", encoding="utf-8")

    service, factory = make_service(tmp_path)
    session = service.create_session(tmp_path)
    old_runtime_id = session.runtime_id

    marker.write_text("two", encoding="utf-8")
    new_runtime = service.open_repository(tmp_path)

    assert new_runtime.runtime_id != old_runtime_id
    assert factory.calls == 2

    with pytest.raises(RepositorySessionStaleError, match="session is stale"):
        service.get_state(tmp_path, session.session_id)


def test_empty_query_is_rejected_before_routing(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.ask(tmp_path, session.session_id, "   ")


def test_repository_and_session_snapshots(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    session = service.create_session(tmp_path)

    repositories = service.repository_snapshots()
    sessions = service.session_snapshots(repository_root=tmp_path)

    assert len(repositories) == 1
    assert len(sessions) == 1
    assert sessions[0].session_id == session.session_id


def test_close_all_clears_everything(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("one", encoding="utf-8")
    service, _ = make_service(tmp_path)

    service.create_session(tmp_path)
    service.close_all()

    assert len(service.runtime_registry) == 0
    assert len(service.session_manager) == 0
