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


def test_root_reports_11_2_dashboard() -> None:

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

    assert (
        "RepoBrain"
        in response.text
    )

    assert (
        "API version"
        in response.text
    )

    assert (
        "11.2"
        in response.text
    )


def test_health_endpoint_returns_typed_health_snapshot() -> None:

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
                "/health"
            )
        )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["ready"]
        is True
    )

    assert (
        payload["loaded_repositories"]
        == 1
    )

    assert (
        payload["active_sessions"]
        == 1
    )

    assert (
        payload["status"]
        == "ready"
    )


def test_diagnostics_endpoint_returns_application_snapshot() -> None:

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
                "/diagnostics"
            )
        )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["health"]["ready"]
        is True
    )

    assert (
        payload["health"]["loaded_repositories"]
        == 1
    )

    assert (
        len(
            payload["sessions"]
        )
        == 1
    )

    assert (
        payload["sessions"][0]["session_id"]
        == "session-1"
    )


def test_health_and_diagnostics_are_present_in_openapi() -> None:

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
                "/openapi.json"
            )
        )

    assert (
        response.status_code
        == 200
    )

    paths = (
        response.json()
        ["paths"]
    )

    assert (
        "/health"
        in paths
    )

    assert (
        "/diagnostics"
        in paths
    )

    # Dashboard is human-facing UI, not an API operation.
    assert (
        "/"
        not in paths
    )


def test_health_does_not_close_application_until_shutdown() -> None:

    application = (
        FakeApplication()
    )

    app = (
        create_app(
            application=application
        )
    )

    with TestClient(
        app
    ) as client:

        response = (
            client.get(
                "/health"
            )
        )

        assert (
            response.status_code
            == 200
        )

        assert (
            application.close_all_calls
            == 0
        )

    assert (
        application.close_all_calls
        == 1
    )
