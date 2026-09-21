from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from types import SimpleNamespace

from fastapi.testclient import (
    TestClient,
)

from repobrain.api import (
    create_app,
)
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationHealthSnapshot,
    ConversationSessionSnapshot,
    RepositoryRuntimeSnapshot,
)


class FakeApplication:

    def __init__(
        self,
    ) -> None:

        now = datetime.now(
            timezone.utc
        )

        self.close_all_calls = 0

        self.runtime = RepositoryRuntimeSnapshot(
            runtime_id="runtime-1",
            repository_root="D:/repo",
            created_at=now,
            conversation_count=1,
            closed=False,
            fingerprint_algorithm="sha256-v1",
            fingerprint_value="a" * 64,
            fingerprint_file_count=10,
        )

        self.session = ConversationSessionSnapshot(
            session_id="session-1",
            runtime_id="runtime-1",
            repository_root="D:/repo",
            conversation_id="conversation-1",
            created_at=now,
            last_accessed_at=now,
            turn_number=0,
        )

    def close_all(
        self,
    ) -> None:

        self.close_all_calls += 1

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
            sessions=(self.session,),
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

    def repository_diagnostics(
        self,
        repository_root,
    ):

        return None

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

    def get_state(
        self,
        repository_root,
        session_id,
    ):

        return SimpleNamespace(
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

        return self.session.model_copy(
            update={
                "conversation_id": "conversation-2",
            }
        )

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

        citation = SimpleNamespace(
            citation_id="E1",
            citation_kind=SimpleNamespace(
                value="SOURCE"
            ),
            relative_path="repobrain/example.py",
            start_line=10,
            end_line=20,
            qualified_name="repobrain.example.fn",
            relationship_type=None,
            source_qualified_name=None,
            target_qualified_name=None,
        )

        answer = SimpleNamespace(
            answer_text="Grounded answer [E1]",
            model_name="deterministic-graph",
            grounded=True,
            citations=(citation,),
            invalid_citation_ids=(),
        )

        agent_turn = SimpleNamespace(
            original_query=query,
            resolved_query=query,
            agent_result=SimpleNamespace(
                state=SimpleNamespace(
                    intent=SimpleNamespace(
                        value="GENERAL"
                    )
                )
            ),
        )

        state_after = SimpleNamespace(
            conversation_id="conversation-1",
            turn_number=1,
            current_qualified_name="repobrain.example.fn",
            previous_qualified_name=None,
            previous_intent=SimpleNamespace(
                value="GENERAL"
            ),
            relationship_focus_type=None,
            relationship_qualified_name=None,
            last_query=query,
            last_resolved_query=query,
            last_grounded=True,
            last_citation_ids=("E1",),
            last_invalid_citation_ids=(),
        )

        return SimpleNamespace(
            answer=answer,
            agent_turn=agent_turn,
            state_after=state_after,
        )


def make_client():

    app = create_app(
        application=FakeApplication()
    )

    return TestClient(
        app
    )


def test_repository_open_endpoint() -> None:

    with make_client() as client:

        response = client.post(
            "/repositories/open",
            json={
                "repository_root": "D:/repo",
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["runtime_id"] == "runtime-1"

    assert payload["repository_root"] == "D:/repo"


def test_session_create_endpoint() -> None:

    with make_client() as client:

        response = client.post(
            "/sessions",
            json={
                "repository_root": "D:/repo",
            },
        )

    assert response.status_code == 200

    assert (
        response.json()["session_id"]
        == "session-1"
    )


def test_query_ask_endpoint() -> None:

    with make_client() as client:

        response = client.post(
            "/query/ask",
            json={
                "repository_root": "D:/repo",
                "session_id": "session-1",
                "query": "what does this function do?",
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["answer_text"]
        == "Grounded answer [E1]"
    )

    assert payload["grounded"] is True

    assert (
        payload["citations"][0]["citation_id"]
        == "E1"
    )


def test_session_state_clear_and_new_endpoints() -> None:

    with make_client() as client:

        state = client.post(
            "/sessions/state",
            json={
                "repository_root": "D:/repo",
                "session_id": "session-1",
            },
        )

        cleared = client.post(
            "/sessions/clear",
            json={
                "repository_root": "D:/repo",
                "session_id": "session-1",
            },
        )

        new = client.post(
            "/sessions/new",
            json={
                "repository_root": "D:/repo",
                "session_id": "session-1",
            },
        )

    assert state.status_code == 200

    assert cleared.status_code == 200

    assert new.status_code == 200

    assert (
        new.json()["conversation_id"]
        == "conversation-2"
    )


def test_repository_close_endpoint() -> None:

    with make_client() as client:

        response = client.post(
            "/repositories/close",
            json={
                "repository_root": "D:/repo",
            },
        )

    assert response.status_code == 200

    assert (
        response.json()["closed"]
        is True
    )


def test_dashboard_contains_repository_and_chat_controls() -> None:

    with make_client() as client:

        response = client.get(
            "/"
        )

    assert response.status_code == 200

    body = response.text

    assert "Local repository path" in body
    assert "Open repository" in body
    assert "Start session" in body
    assert "Ask RepoBrain" in body
    assert "/repositories/open" in body
    assert "/sessions" in body
    assert "/query/ask" in body


def test_new_endpoints_are_visible_in_openapi() -> None:

    with make_client() as client:

        payload = client.get(
            "/openapi.json"
        ).json()

    paths = payload["paths"]

    assert "/repositories/open" in paths
    assert "/repositories/close" in paths
    assert "/repositories/diagnostics" in paths
    assert "/sessions" in paths
    assert "/sessions/state" in paths
    assert "/sessions/clear" in paths
    assert "/sessions/new" in paths
    assert "/query/ask" in paths
