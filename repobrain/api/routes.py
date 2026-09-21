from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from repobrain.application import RepoBrainTransport
from repobrain.models.application import (
    ApplicationErrorCode,
    ApplicationErrorResponse,
    AskRequest,
    AskResponse,
    CloseRepositoryRequest,
    CloseRepositoryResponse,
    ConversationSessionSnapshot,
    CreateSessionRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    InvestigationStateResponse,
    OpenRepositoryRequest,
    RepositoryDiagnosticsSnapshot,
    RepositoryRuntimeSnapshot,
    SessionCommandRequest,
)


def _error_status(error: ApplicationErrorResponse) -> int:
    mapping = {
        ApplicationErrorCode.INVALID_REQUEST: status.HTTP_400_BAD_REQUEST,
        ApplicationErrorCode.SESSION_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ApplicationErrorCode.SESSION_STALE: status.HTTP_409_CONFLICT,
        ApplicationErrorCode.FOLLOW_UP_UNRESOLVED: status.HTTP_409_CONFLICT,
        ApplicationErrorCode.REPOSITORY_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ApplicationErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    return mapping[error.code]


def _to_http(result: object):
    if isinstance(result, ApplicationErrorResponse):
        return JSONResponse(
            status_code=_error_status(result),
            content=result.model_dump(mode="json"),
        )
    return result


def create_repository_router(*, transport: RepoBrainTransport) -> APIRouter:
    router = APIRouter(prefix="/repositories", tags=["repositories"])

    @router.post(
        "/open",
        response_model=RepositoryRuntimeSnapshot | ApplicationErrorResponse,
    )
    def open_repository(request: OpenRepositoryRequest):
        return _to_http(transport.open_repository(request))

    @router.post(
        "/close",
        response_model=CloseRepositoryResponse | ApplicationErrorResponse,
    )
    def close_repository(request: CloseRepositoryRequest):
        return _to_http(transport.close_repository(request))

    @router.post(
        "/diagnostics",
        response_model=RepositoryDiagnosticsSnapshot | None | ApplicationErrorResponse,
    )
    def repository_diagnostics(request: OpenRepositoryRequest):
        return _to_http(transport.repository_diagnostics(request))

    return router


def create_session_router(*, transport: RepoBrainTransport) -> APIRouter:
    router = APIRouter(prefix="/sessions", tags=["sessions"])

    @router.post(
        "",
        response_model=ConversationSessionSnapshot | ApplicationErrorResponse,
    )
    def create_session(request: CreateSessionRequest):
        return _to_http(transport.create_session(request))

    @router.post(
        "/state",
        response_model=InvestigationStateResponse | ApplicationErrorResponse,
    )
    def get_state(request: SessionCommandRequest):
        return _to_http(transport.get_state(request))

    @router.post(
        "/clear",
        response_model=ConversationSessionSnapshot | ApplicationErrorResponse,
    )
    def clear_session(request: SessionCommandRequest):
        return _to_http(transport.clear_session(request))

    @router.post(
        "/new",
        response_model=ConversationSessionSnapshot | ApplicationErrorResponse,
    )
    def new_conversation(request: SessionCommandRequest):
        return _to_http(transport.new_conversation(request))

    @router.delete(
        "",
        response_model=DeleteSessionResponse | ApplicationErrorResponse,
    )
    def delete_session(request: DeleteSessionRequest):
        return _to_http(transport.delete_session(request))

    return router


def create_query_router(*, transport: RepoBrainTransport) -> APIRouter:
    router = APIRouter(prefix="/query", tags=["query"])

    @router.post(
        "/ask",
        response_model=AskResponse | ApplicationErrorResponse,
    )
    def ask(request: AskRequest):
        return _to_http(transport.ask(request))

    return router
