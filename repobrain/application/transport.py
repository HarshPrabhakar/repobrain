from __future__ import annotations

from pathlib import Path

from repobrain.application.service import (
    RepoBrainApplicationService,
    RepositorySessionStaleError,
)
from repobrain.application.sessions import (
    ConversationSessionNotFoundError,
)
from repobrain.conversation import (
    UnresolvedFollowUpReferenceError,
)
from repobrain.models.application import (
    ApplicationDiagnosticsSnapshot,
    ApplicationErrorCode,
    ApplicationErrorResponse,
    ApplicationHealthSnapshot,
    AskRequest,
    AskResponse,
    CitationResponse,
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
    SessionDiagnosticsSnapshot,
)


class RepoBrainTransport:
    """
    Phase 10.7 transport-safe facade.

    This class deliberately exposes Pydantic request/response models
    instead of internal runtime/orchestrator/result objects.

    A future FastAPI, desktop UI, or RPC adapter can call this layer
    without depending on RepoBrain's internal Python object graph.
    """

    def __init__(
        self,
        *,
        application: RepoBrainApplicationService,
    ) -> None:

        self._application = (
            application
        )

    # ==================================================================
    # Repository operations
    # ==================================================================

    def open_repository(
        self,
        request: OpenRepositoryRequest,
    ) -> (
        RepositoryRuntimeSnapshot
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .open_repository(
                    Path(
                        request.repository_root
                    )
                )
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except FileNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.REPOSITORY_NOT_FOUND,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    def close_repository(
        self,
        request: CloseRepositoryRequest,
    ) -> (
        CloseRepositoryResponse
        | ApplicationErrorResponse
    ):

        try:

            closed = (
                self._application
                .close_repository(
                    Path(
                        request.repository_root
                    )
                )
            )

            return (
                CloseRepositoryResponse(
                    closed=(
                        closed
                    )
                )
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    # ==================================================================
    # Session operations
    # ==================================================================

    def create_session(
        self,
        request: CreateSessionRequest,
    ) -> (
        ConversationSessionSnapshot
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .create_session(
                    Path(
                        request.repository_root
                    )
                )
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    def delete_session(
        self,
        request: DeleteSessionRequest,
    ) -> (
        DeleteSessionResponse
        | ApplicationErrorResponse
    ):

        try:

            return (
                DeleteSessionResponse(
                    deleted=(
                        self._application
                        .delete_session(
                            request.session_id
                        )
                    )
                )
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    # ==================================================================
    # Query
    # ==================================================================

    def ask(
        self,
        request: AskRequest,
    ) -> (
        AskResponse
        | ApplicationErrorResponse
    ):

        try:

            turn = (
                self._application
                .ask(
                    Path(
                        request.repository_root
                    ),
                    request.session_id,
                    request.query,
                )
            )

            return (
                self._serialize_turn(
                    turn
                )
            )

        except ConversationSessionNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_NOT_FOUND,
                str(
                    exc
                ),
            )

        except RepositorySessionStaleError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_STALE,
                str(
                    exc
                ),
            )

        except UnresolvedFollowUpReferenceError as exc:

            reference_text = getattr(
                exc,
                "reference_text",
                None,
            )

            message = (
                "Follow-up reference could not be resolved "
                "from the current investigation state."
            )

            if reference_text:

                message = (
                    f"{message} "
                    f"Reference: {reference_text!r}."
                )

            return self._error(
                ApplicationErrorCode.FOLLOW_UP_UNRESOLVED,
                message,
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    # ==================================================================
    # Conversation commands
    # ==================================================================

    def get_state(
        self,
        request: SessionCommandRequest,
    ) -> (
        InvestigationStateResponse
        | ApplicationErrorResponse
    ):

        try:

            state = (
                self._application
                .get_state(
                    Path(
                        request.repository_root
                    ),
                    request.session_id,
                )
            )

            return (
                self._serialize_state(
                    state
                )
            )

        except ConversationSessionNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_NOT_FOUND,
                str(
                    exc
                ),
            )

        except RepositorySessionStaleError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_STALE,
                str(
                    exc
                ),
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    def clear_session(
        self,
        request: SessionCommandRequest,
    ) -> (
        ConversationSessionSnapshot
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .clear_session(
                    Path(
                        request.repository_root
                    ),
                    request.session_id,
                )
            )

        except ConversationSessionNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_NOT_FOUND,
                str(
                    exc
                ),
            )

        except RepositorySessionStaleError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_STALE,
                str(
                    exc
                ),
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    def new_conversation(
        self,
        request: SessionCommandRequest,
    ) -> (
        ConversationSessionSnapshot
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .new_conversation(
                    Path(
                        request.repository_root
                    ),
                    request.session_id,
                )
            )

        except ConversationSessionNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_NOT_FOUND,
                str(
                    exc
                ),
            )

        except RepositorySessionStaleError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_STALE,
                str(
                    exc
                ),
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def health(
        self,
    ) -> ApplicationHealthSnapshot:

        return (
            self._application
            .health()
        )

    def diagnostics(
        self,
    ) -> ApplicationDiagnosticsSnapshot:

        return (
            self._application
            .diagnostics()
        )

    def repository_diagnostics(
        self,
        request: OpenRepositoryRequest,
    ) -> (
        RepositoryDiagnosticsSnapshot
        | None
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .repository_diagnostics(
                    Path(
                        request.repository_root
                    )
                )
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    def session_diagnostics(
        self,
        request: DeleteSessionRequest,
    ) -> (
        SessionDiagnosticsSnapshot
        | ApplicationErrorResponse
    ):

        try:

            return (
                self._application
                .session_diagnostics(
                    request.session_id
                )
            )

        except ConversationSessionNotFoundError as exc:

            return self._error(
                ApplicationErrorCode.SESSION_NOT_FOUND,
                str(
                    exc
                ),
            )

        except ValueError as exc:

            return self._error(
                ApplicationErrorCode.INVALID_REQUEST,
                str(
                    exc
                ),
            )

        except Exception as exc:

            return self._error(
                ApplicationErrorCode.INTERNAL_ERROR,
                str(
                    exc
                ),
                retryable=True,
            )

    # ==================================================================
    # Serialization
    # ==================================================================

    def _serialize_turn(
        self,
        turn: object,
    ) -> AskResponse:
        """
        Convert the existing grounded Phase-9 turn into a stable DTO.
        """

        answer = (
            turn.answer
        )

        agent_turn = (
            turn.agent_turn
        )

        state_after = (
            turn.state_after
        )

        agent_result = (
            agent_turn.agent_result
        )

        result_state = getattr(
            agent_result,
            "state",
            None,
        )

        intent = (
            getattr(
                result_state,
                "intent",
                None,
            )
            if result_state is not None
            else None
        )

        answer_text = getattr(
            answer,
            "answer_text",
            None,
        )

        if answer_text is None:

            answer_text = getattr(
                answer,
                "text",
                None,
            )

        if answer_text is None:

            answer_text = str(
                answer
            )

        model_name = getattr(
            answer,
            "model_name",
            None,
        )

        grounded = getattr(
            answer,
            "grounded",
            None,
        )

        if grounded is None:

            grounded = (
                state_after.last_grounded
            )

        raw_citations = getattr(
            answer,
            "citations",
            (),
        )

        citations = tuple(
            self._serialize_citation(
                citation
            )
            for citation
            in raw_citations
        )

        invalid_citation_ids = tuple(
            str(
                value
            )
            for value
            in getattr(
                answer,
                "invalid_citation_ids",
                (),
            )
        )

        return (
            AskResponse(
                original_query=(
                    agent_turn.original_query
                ),
                resolved_query=(
                    agent_turn.resolved_query
                ),
                answer_text=(
                    str(
                        answer_text
                    )
                ),
                model_name=(
                    str(
                        model_name
                    )
                    if model_name is not None
                    else None
                ),
                intent=(
                    self._enum_value(
                        intent
                    )
                ),
                grounded=(
                    bool(
                        grounded
                    )
                ),
                citations=(
                    citations
                ),
                invalid_citation_ids=(
                    invalid_citation_ids
                ),
                state=(
                    self._serialize_state(
                        state_after
                    )
                ),
            )
        )

    def _serialize_citation(
        self,
        citation: object,
    ) -> CitationResponse:

        citation_id = getattr(
            citation,
            "citation_id",
            None,
        )

        if citation_id is None:

            raise RuntimeError(
                "Answer citation is missing citation_id."
            )

        return (
            CitationResponse(
                citation_id=(
                    str(
                        citation_id
                    )
                ),
                citation_kind=(
                    self._enum_value(
                        getattr(
                            citation,
                            "citation_kind",
                            None,
                        )
                    )
                ),
                relative_path=(
                    getattr(
                        citation,
                        "relative_path",
                        None,
                    )
                ),
                start_line=(
                    getattr(
                        citation,
                        "start_line",
                        None,
                    )
                ),
                end_line=(
                    getattr(
                        citation,
                        "end_line",
                        None,
                    )
                ),
                qualified_name=(
                    getattr(
                        citation,
                        "qualified_name",
                        None,
                    )
                ),
                relationship_type=(
                    self._enum_value(
                        getattr(
                            citation,
                            "relationship_type",
                            None,
                        )
                    )
                ),
                source_qualified_name=(
                    getattr(
                        citation,
                        "source_qualified_name",
                        None,
                    )
                ),
                target_qualified_name=(
                    getattr(
                        citation,
                        "target_qualified_name",
                        None,
                    )
                ),
            )
        )

    def _serialize_state(
        self,
        state: object,
    ) -> InvestigationStateResponse:

        return (
            InvestigationStateResponse(
                conversation_id=(
                    str(
                        state.conversation_id
                    )
                ),
                turn_number=(
                    int(
                        state.turn_number
                    )
                ),
                current_qualified_name=(
                    state.current_qualified_name
                ),
                previous_qualified_name=(
                    state.previous_qualified_name
                ),
                previous_intent=(
                    self._enum_value(
                        state.previous_intent
                    )
                ),
                relationship_focus_type=(
                    self._enum_value(
                        state.relationship_focus_type
                    )
                ),
                relationship_qualified_name=(
                    state.relationship_qualified_name
                ),
                last_query=(
                    state.last_query
                ),
                last_resolved_query=(
                    state.last_resolved_query
                ),
                last_grounded=(
                    state.last_grounded
                ),
                last_citation_ids=(
                    tuple(
                        str(
                            value
                        )
                        for value
                        in state.last_citation_ids
                    )
                ),
                last_invalid_citation_ids=(
                    tuple(
                        str(
                            value
                        )
                        for value
                        in state.last_invalid_citation_ids
                    )
                ),
            )
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _enum_value(
        value: object,
    ) -> str | None:

        if value is None:
            return None

        enum_value = getattr(
            value,
            "value",
            None,
        )

        if enum_value is not None:
            return str(
                enum_value
            )

        return str(
            value
        )

    @staticmethod
    def _error(
        code: ApplicationErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> ApplicationErrorResponse:

        return (
            ApplicationErrorResponse(
                code=(
                    code
                ),
                message=(
                    message
                    if message.strip()
                    else code.value
                ),
                retryable=(
                    retryable
                ),
            )
        )
