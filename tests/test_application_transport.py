from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from types import SimpleNamespace

from repobrain.application import (
    RepoBrainTransport,
)
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationErrorCode,
    ApplicationHealthSnapshot,
    AskRequest,
    AskResponse,
    CloseRepositoryRequest,
    ConversationSessionSnapshot,
    CreateSessionRequest,
    DeleteSessionRequest,
    OpenRepositoryRequest,
    RepositoryRuntimeSnapshot,
    SessionCommandRequest,
)


class FakeApplication:
    """
    Minimal fake for verifying transport contracts without loading
    repository indexes, embedding models, or Ollama.
    """

    def __init__(
        self,
    ) -> None:

        now = (
            datetime.now(
                timezone.utc
            )
        )

        self.runtime = (
            RepositoryRuntimeSnapshot(
                runtime_id="runtime-1",
                repository_root="D:/repo",
                created_at=now,
                conversation_count=1,
                closed=False,
            )
        )

        self.session = (
            ConversationSessionSnapshot(
                session_id="session-1",
                runtime_id="runtime-1",
                repository_root="D:/repo",
                conversation_id="conversation-1",
                created_at=now,
                last_accessed_at=now,
                turn_number=0,
            )
        )

    def open_repository(
        self,
        repository_root,
    ):
        return self.runtime

    def close_repository(
        self,
        repository_root,
    ):
        return True

    def create_session(
        self,
        repository_root,
    ):
        return self.session

    def delete_session(
        self,
        session_id,
    ):
        return True

    def health(
        self,
    ):
        return ApplicationHealthSnapshot(
            ready=True,
            loaded_repositories=1,
            active_sessions=1,
            status="ready",
        )

    def diagnostics(
        self,
    ):
        return ApplicationDiagnosticsSnapshot(
            health=self.health(),
            repositories=(),
            sessions=(
                self.session,
            ),
        )

    def repository_diagnostics(
        self,
        repository_root,
    ):
        return None

    def session_diagnostics(
        self,
        session_id,
    ):
        from repobrain.models.application import (
            SessionDiagnosticsSnapshot,
        )

        return SessionDiagnosticsSnapshot(
            session=self.session,
            runtime_loaded=True,
            runtime_current=True,
        )

    def ask(
        self,
        repository_root,
        session_id,
        query,
    ):
        citation = (
            SimpleNamespace(
                citation_id="E1",
                citation_kind=(
                    SimpleNamespace(
                        value="SOURCE"
                    )
                ),
                relative_path=(
                    "repobrain/example.py"
                ),
                start_line=10,
                end_line=20,
                qualified_name=(
                    "repobrain.example.fn"
                ),
                relationship_type=None,
                source_qualified_name=None,
                target_qualified_name=None,
            )
        )

        answer = (
            SimpleNamespace(
                answer_text="Example answer [E1]",
                model_name="qwen2.5-coder:14b",
                grounded=True,
                citations=(
                    citation,
                ),
                invalid_citation_ids=(),
            )
        )

        result_state = (
            SimpleNamespace(
                intent=(
                    SimpleNamespace(
                        value="DEFINITION"
                    )
                )
            )
        )

        agent_result = (
            SimpleNamespace(
                state=result_state
            )
        )

        agent_turn = (
            SimpleNamespace(
                original_query=query,
                resolved_query=query,
                agent_result=agent_result,
            )
        )

        state_after = (
            SimpleNamespace(
                conversation_id="conversation-1",
                turn_number=1,
                current_qualified_name=(
                    "repobrain.example.fn"
                ),
                previous_qualified_name=None,
                previous_intent=(
                    SimpleNamespace(
                        value="DEFINITION"
                    )
                ),
                relationship_focus_type=None,
                relationship_qualified_name=None,
                last_query=query,
                last_resolved_query=query,
                last_grounded=True,
                last_citation_ids=(
                    "E1",
                ),
                last_invalid_citation_ids=(),
            )
        )

        return (
            SimpleNamespace(
                answer=answer,
                agent_turn=agent_turn,
                state_after=state_after,
            )
        )

    def get_state(
        self,
        repository_root,
        session_id,
    ):
        return (
            SimpleNamespace(
                conversation_id="conversation-1",
                turn_number=0,
                current_qualified_name=None,
                previous_qualified_name=None,
                previous_intent=None,
                relationship_focus_type=None,
                relationship_qualified_name=None,
                last_query=None,
                last_resolved_query=None,
                last_grounded=None,
                last_citation_ids=(),
                last_invalid_citation_ids=(),
            )
        )

    def clear_session(
        self,
        repository_root,
        session_id,
    ):
        return self.session

    def new_conversation(
        self,
        repository_root,
        session_id,
    ):
        return self.session


def make_transport() -> RepoBrainTransport:

    return RepoBrainTransport(
        application=FakeApplication()
    )


def test_open_repository_returns_transport_safe_model() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root="D:/repo"
            )
        )
    )

    assert isinstance(
        result,
        RepositoryRuntimeSnapshot,
    )

    assert (
        result.runtime_id
        == "runtime-1"
    )


def test_create_session_returns_snapshot() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.create_session(
            CreateSessionRequest(
                repository_root="D:/repo"
            )
        )
    )

    assert isinstance(
        result,
        ConversationSessionSnapshot,
    )

    assert (
        result.session_id
        == "session-1"
    )


def test_ask_serializes_grounded_turn() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.ask(
            AskRequest(
                repository_root="D:/repo",
                session_id="session-1",
                query="where is fn defined?",
            )
        )
    )

    assert isinstance(
        result,
        AskResponse,
    )

    assert (
        result.answer_text
        == "Example answer [E1]"
    )

    assert (
        result.intent
        == "DEFINITION"
    )

    assert (
        result.grounded
        is True
    )

    assert (
        result.citations[0]
        .citation_id
        == "E1"
    )

    assert (
        result.citations[0]
        .relative_path
        == "repobrain/example.py"
    )

    assert (
        result.state.turn_number
        == 1
    )


def test_get_state_returns_transport_state() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.get_state(
            SessionCommandRequest(
                repository_root="D:/repo",
                session_id="session-1",
            )
        )
    )

    assert (
        result.conversation_id
        == "conversation-1"
    )

    assert (
        result.turn_number
        == 0
    )


def test_delete_session_returns_response_model() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.delete_session(
            DeleteSessionRequest(
                session_id="session-1"
            )
        )
    )

    assert result.deleted is True


def test_close_repository_returns_response_model() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.close_repository(
            CloseRepositoryRequest(
                repository_root="D:/repo"
            )
        )
    )

    assert result.closed is True


def test_health_and_diagnostics_are_direct_typed_contracts() -> None:

    transport = (
        make_transport()
    )

    health = (
        transport.health()
    )

    diagnostics = (
        transport.diagnostics()
    )

    assert (
        health.ready
        is True
    )

    assert (
        diagnostics.health.status
        == "ready"
    )


def test_session_diagnostics_is_transport_safe() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.session_diagnostics(
            DeleteSessionRequest(
                session_id="session-1"
            )
        )
    )

    assert (
        result.runtime_loaded
        is True
    )

    assert (
        result.runtime_current
        is True
    )


def test_invalid_request_becomes_stable_error_response() -> None:

    class InvalidApplication(
        FakeApplication
    ):

        def open_repository(
            self,
            repository_root,
        ):
            raise ValueError(
                "bad repository"
            )

    transport = (
        RepoBrainTransport(
            application=(
                InvalidApplication()
            )
        )
    )

    result = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root="D:/repo"
            )
        )
    )

    assert (
        result.code
        == ApplicationErrorCode.INVALID_REQUEST
    )

    assert (
        result.retryable
        is False
    )


def test_internal_failure_becomes_retryable_error() -> None:

    class BrokenApplication(
        FakeApplication
    ):

        def open_repository(
            self,
            repository_root,
        ):
            raise RuntimeError(
                "unexpected"
            )

    transport = (
        RepoBrainTransport(
            application=(
                BrokenApplication()
            )
        )
    )

    result = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root="D:/repo"
            )
        )
    )

    assert (
        result.code
        == ApplicationErrorCode.INTERNAL_ERROR
    )

    assert (
        result.retryable
        is True
    )


def test_transport_models_serialize_to_json_compatible_data() -> None:

    transport = (
        make_transport()
    )

    result = (
        transport.ask(
            AskRequest(
                repository_root="D:/repo",
                session_id="session-1",
                query="where is fn defined?",
            )
        )
    )

    payload = (
        result.model_dump(
            mode="json"
        )
    )

    assert (
        payload["answer_text"]
        == "Example answer [E1]"
    )

    assert (
        payload["citations"][0]
        ["citation_id"]
        == "E1"
    )

    assert (
        payload["state"]
        ["conversation_id"]
        == "conversation-1"
    )
