from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from repobrain.api import (
    create_app,
)


class FakeApplication:
    """
    Minimal Phase-11 application fake.

    These tests validate application construction and lifecycle only.
    """

    def __init__(
        self,
    ) -> None:

        self.close_all_calls = 0

    def close_all(
        self,
    ) -> None:

        self.close_all_calls += 1


def test_create_app_returns_repobrain_metadata() -> None:

    application = (
        FakeApplication()
    )

    app = (
        create_app(
            application=application
        )
    )

    assert (
        app.title
        == "RepoBrain API"
    )

    assert (
        app.version
        == "11.2"
    )


def test_root_endpoint_serves_dashboard_html() -> None:

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
        "Loaded repositories"
        in response.text
    )


def test_application_and_transport_are_attached_to_app_state() -> None:

    application = (
        FakeApplication()
    )

    app = (
        create_app(
            application=application
        )
    )

    assert (
        app.state.repobrain_service
        is application
    )

    assert (
        app.state.repobrain_transport
        is not None
    )


def test_shutdown_closes_application_resources_once() -> None:

    application = (
        FakeApplication()
    )

    app = (
        create_app(
            application=application
        )
    )

    assert (
        application.close_all_calls
        == 0
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
            application.close_all_calls
            == 0
        )

    assert (
        application.close_all_calls
        == 1
    )


def test_openapi_document_is_generated() -> None:

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
                "/openapi.json"
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
        payload["info"]["title"]
        == "RepoBrain API"
    )

    # The dashboard root is intentionally hidden from the API schema.
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
