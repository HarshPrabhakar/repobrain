from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

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
)


class FakeApplication:

    def __init__(
        self,
    ) -> None:

        self.close_all_calls = 0

        now = (
            datetime.now(
                timezone.utc
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

    def close_all(
        self,
    ) -> None:

        self.close_all_calls += 1

    def health(
        self,
    ) -> ApplicationHealthSnapshot:

        return (
            ApplicationHealthSnapshot(
                ready=True,
                loaded_repositories=1,
                active_sessions=1,
                status="ready",
            )
        )

    def diagnostics(
        self,
    ) -> ApplicationDiagnosticsSnapshot:

        return (
            ApplicationDiagnosticsSnapshot(
                health=(
                    self.health()
                ),
                repositories=(),
                sessions=(
                    self.session,
                ),
            )
        )


def test_root_serves_dashboard_html() -> None:

    app = (
        create_app(
            application=(
                FakeApplication()
            )
        )
    )

    with TestClient(
        app
    ) as client:

        response = (
            client.get(
                "/"
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        "text/html"
        in response.headers[
            "content-type"
        ]
    )

    body = (
        response.text
    )

    assert (
        "RepoBrain"
        in body
    )

    assert (
        "Loaded repositories"
        in body
    )

    assert (
        "Active sessions"
        in body
    )

    assert (
        "/health"
        in body
    )

    assert (
        "/diagnostics"
        in body
    )


def test_dashboard_keeps_machine_endpoints_available() -> None:

    app = (
        create_app(
            application=(
                FakeApplication()
            )
        )
    )

    with TestClient(
        app
    ) as client:

        health = (
            client.get(
                "/health"
            )
        )

        diagnostics = (
            client.get(
                "/diagnostics"
            )
        )

    assert (
        health.status_code
        == 200
    )

    assert (
        diagnostics.status_code
        == 200
    )


def test_root_ui_is_not_added_to_openapi_schema() -> None:

    app = (
        create_app(
            application=(
                FakeApplication()
            )
        )
    )

    with TestClient(
        app
    ) as client:

        payload = (
            client.get(
                "/openapi.json"
            )
            .json()
        )

    assert (
        "/"
        not in payload["paths"]
    )

    assert (
        "/health"
        in payload["paths"]
    )

    assert (
        "/diagnostics"
        in payload["paths"]
    )
