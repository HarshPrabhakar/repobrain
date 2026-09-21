from __future__ import annotations

import hashlib
from pathlib import Path

from repobrain.application import (
    ConversationSessionManager,
    RepoBrainApplicationService,
    RepoBrainTransport,
    RepositoryRuntime,
    RepositoryRuntimeRegistry,
)
from repobrain.models.agent import (
    AgentIntent,
)
from repobrain.models.application import (
    ApplicationErrorCode,
    AskRequest,
    AskResponse,
    CreateSessionRequest,
    OpenRepositoryRequest,
    CloseRepositoryRequest,
    RepositoryFingerprint,
    SessionCommandRequest,
)


class FakeAgentState:

    def __init__(
        self,
        intent: AgentIntent,
    ) -> None:

        self.intent = intent


class FakeAgentResult:

    def __init__(
        self,
        *,
        query: str,
    ) -> None:

        self.state = (
            FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )

        self.resolved_symbol_id = None
        self.resolved_qualified_name = None
        self.graph_facts = ()
        self.query = query


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
            query=query
        )


class FakeCitation:

    citation_id = "E1"
    citation_kind = (
        type(
            "CitationKind",
            (),
            {
                "value": "SOURCE",
            },
        )()
    )
    relative_path = "repobrain/example.py"
    start_line = 1
    end_line = 5
    qualified_name = "repobrain.example.fn"
    relationship_type = None
    source_qualified_name = None
    target_qualified_name = None


class FakeAnswer:

    answer_text = (
        "Grounded application answer [E1]"
    )

    model_name = (
        "fake-local-model"
    )

    grounded = True

    citations = (
        FakeCitation(),
    )

    invalid_citation_ids = ()


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> FakeAnswer:

        return FakeAnswer()


def fingerprint_for(
    repository_root: Path,
) -> RepositoryFingerprint:

    marker = (
        repository_root
        / "marker.txt"
    )

    content = (
        marker.read_bytes()
    )

    return RepositoryFingerprint(
        value=(
            hashlib.sha256(
                content
            )
            .hexdigest()
        ),
        file_count=1,
        total_size_bytes=(
            len(
                content
            )
        ),
    )


class RuntimeFactory:

    def __init__(
        self,
    ) -> None:

        self.calls = 0
        self.runners: list[
            RecordingAgentRunner
        ] = []

    def __call__(
        self,
        repository_root: Path,
    ) -> RepositoryRuntime:

        self.calls += 1

        runner = (
            RecordingAgentRunner()
        )

        self.runners.append(
            runner
        )

        return RepositoryRuntime(
            repository_root=(
                repository_root
            ),
            agent_runner=(
                runner
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
            fingerprint=(
                fingerprint_for(
                    repository_root
                )
            ),
        )


def build_stack(
    repository_root: Path,
) -> tuple[
    RepoBrainTransport,
    RepoBrainApplicationService,
    RuntimeFactory,
]:

    factory = (
        RuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=(
                factory
            ),
            fingerprint_provider=(
                fingerprint_for
            ),
        )
    )

    service = (
        RepoBrainApplicationService(
            runtime_registry=(
                registry
            ),
            session_manager=(
                ConversationSessionManager()
            ),
        )
    )

    transport = (
        RepoBrainTransport(
            application=(
                service
            )
        )
    )

    return (
        transport,
        service,
        factory,
    )


def test_phase10_end_to_end_repository_session_query_flow(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "version-one",
        encoding="utf-8",
    )

    (
        transport,
        service,
        factory,
    ) = build_stack(
        tmp_path
    )

    runtime = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    answer = (
        transport.ask(
            AskRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    session.session_id
                ),
                query=(
                    "explain the repository"
                ),
            )
        )
    )

    state = (
        transport.get_state(
            SessionCommandRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    session.session_id
                ),
            )
        )
    )

    health = (
        transport.health()
    )

    diagnostics = (
        transport.diagnostics()
    )

    assert (
        factory.calls
        == 1
    )

    assert (
        runtime.runtime_id
        == session.runtime_id
    )

    assert isinstance(
        answer,
        AskResponse,
    )

    assert (
        answer.answer_text
        == "Grounded application answer [E1]"
    )

    assert (
        answer.grounded
        is True
    )

    assert (
        answer.citations[0]
        .citation_id
        == "E1"
    )

    assert (
        health.ready
        is True
    )

    assert (
        health.loaded_repositories
        == 1
    )

    assert (
        health.active_sessions
        == 1
    )

    assert (
        len(
            diagnostics.repositories
        )
        == 1
    )

    assert (
        len(
            diagnostics.sessions
        )
        == 1
    )

    assert (
        state.conversation_id
        == session.conversation_id
    )


def test_phase10_end_to_end_runtime_reuse(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "stable",
        encoding="utf-8",
    )

    (
        transport,
        _,
        factory,
    ) = build_stack(
        tmp_path
    )

    first = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    second = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    assert (
        first.runtime_id
        == second.runtime_id
    )

    assert (
        factory.calls
        == 1
    )


def test_phase10_end_to_end_changed_repository_invalidates_old_session(
    tmp_path: Path,
) -> None:

    marker = (
        tmp_path
        / "marker.txt"
    )

    marker.write_text(
        "version-one",
        encoding="utf-8",
    )

    (
        transport,
        _,
        factory,
    ) = build_stack(
        tmp_path
    )

    session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    old_runtime_id = (
        session.runtime_id
    )

    marker.write_text(
        "version-two",
        encoding="utf-8",
    )

    rebuilt = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    result = (
        transport.ask(
            AskRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    session.session_id
                ),
                query="continue",
            )
        )
    )

    assert (
        rebuilt.runtime_id
        != old_runtime_id
    )

    assert (
        factory.calls
        == 2
    )

    assert (
        result.code
        == ApplicationErrorCode.SESSION_STALE
    )

    assert (
        result.retryable
        is False
    )


def test_phase10_end_to_end_new_session_after_rebuild_is_valid(
    tmp_path: Path,
) -> None:

    marker = (
        tmp_path
        / "marker.txt"
    )

    marker.write_text(
        "version-one",
        encoding="utf-8",
    )

    (
        transport,
        _,
        factory,
    ) = build_stack(
        tmp_path
    )

    old_session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    marker.write_text(
        "version-two",
        encoding="utf-8",
    )

    rebuilt = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    new_session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    answer = (
        transport.ask(
            AskRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    new_session.session_id
                ),
                query="fresh question",
            )
        )
    )

    assert (
        old_session.runtime_id
        != rebuilt.runtime_id
    )

    assert (
        new_session.runtime_id
        == rebuilt.runtime_id
    )

    assert (
        factory.calls
        == 2
    )

    assert isinstance(
        answer,
        AskResponse,
    )


def test_phase10_end_to_end_clear_and_new_conversation_semantics(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "stable",
        encoding="utf-8",
    )

    (
        transport,
        _,
        _,
    ) = build_stack(
        tmp_path
    )

    session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    cleared = (
        transport.clear_session(
            SessionCommandRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    session.session_id
                ),
            )
        )
    )

    reset = (
        transport.new_conversation(
            SessionCommandRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                ),
                session_id=(
                    session.session_id
                ),
            )
        )
    )

    assert (
        cleared.session_id
        == session.session_id
    )

    assert (
        cleared.conversation_id
        == session.conversation_id
    )

    assert (
        reset.session_id
        == session.session_id
    )

    assert (
        reset.conversation_id
        != session.conversation_id
    )


def test_phase10_end_to_end_close_repository_cleans_runtime_and_sessions(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "stable",
        encoding="utf-8",
    )

    (
        transport,
        service,
        _,
    ) = build_stack(
        tmp_path
    )

    session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    result = (
        transport.close_repository(
            CloseRepositoryRequest(
                repository_root=(
                    str(
                        tmp_path
                    )
                )
            )
        )
    )

    assert (
        result.closed
        is True
    )

    assert (
        len(
            service.runtime_registry
        )
        == 0
    )

    assert (
        service.session_manager
        .contains(
            session.session_id
        )
        is False
    )
