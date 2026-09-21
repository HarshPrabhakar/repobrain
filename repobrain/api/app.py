from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from repobrain.application import (
    RepoBrainApplicationService,
    RepoBrainTransport,
)
from repobrain.api.routes import (
    create_query_router,
    create_repository_router,
    create_session_router,
)
from repobrain.api.ui import dashboard_response
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationHealthSnapshot,
)


def create_app(
    *,
    application: RepoBrainApplicationService | None = None,
) -> FastAPI:

    service = (
        application
        if application is not None
        else RepoBrainApplicationService()
    )

    transport = RepoBrainTransport(
        application=service
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:

        app.state.repobrain_service = service
        app.state.repobrain_transport = transport

        yield

        service.close_all()

    app = FastAPI(
        title="RepoBrain API",
        description=(
            "Local repository intelligence API backed by RepoBrain."
        ),
        version="11.2",
        lifespan=lifespan,
    )

    app.state.repobrain_service = service
    app.state.repobrain_transport = transport

    @app.get(
        "/",
        tags=["ui"],
        include_in_schema=False,
    )
    def root():
        return dashboard_response()

    @app.get(
        "/health",
        response_model=ApplicationHealthSnapshot,
        tags=["system"],
    )
    def health() -> ApplicationHealthSnapshot:
        return transport.health()

    @app.get(
        "/diagnostics",
        response_model=ApplicationDiagnosticsSnapshot,
        tags=["system"],
    )
    def diagnostics() -> ApplicationDiagnosticsSnapshot:
        return transport.diagnostics()

    app.include_router(
        create_repository_router(
            transport=transport
        )
    )

    app.include_router(
        create_session_router(
            transport=transport
        )
    )

    app.include_router(
        create_query_router(
            transport=transport
        )
    )

    return app


app = create_app()
