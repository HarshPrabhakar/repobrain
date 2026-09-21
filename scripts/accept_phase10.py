from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.application import (
    RepoBrainApplicationService,
    RepoBrainTransport,
    RepositoryRuntimeBuilder,
)
from repobrain.models.application import (
    AskRequest,
    AskResponse,
    CreateSessionRequest,
    OpenRepositoryRequest,
    SessionCommandRequest,
)


SEPARATOR = "=" * 100


def print_header(
    title: str,
) -> None:

    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def print_progress(
    step: int,
    total: int,
    message: str,
    details: dict[str, object],
) -> None:

    if not details:

        print(
            f"[{step}/{total}] {message}"
        )

        return

    for key, value in details.items():

        label = (
            key
            .replace(
                "_",
                " ",
            )
            .title()
        )

        print(
            f"        {label:<20}: {value}"
        )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Live Phase 10 end-to-end acceptance for RepoBrain."
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help=(
            "Repository root to validate."
        ),
    )

    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
    )

    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
    )

    return parser.parse_args()


def main() -> int:

    args = (
        parse_args()
    )

    repository_root = (
        args.repository
        .expanduser()
        .resolve()
    )

    print_header(
        "REPOBRAIN PHASE 10 LIVE ACCEPTANCE"
    )

    print(
        f"Repository : {repository_root}"
    )

    builder = (
        RepositoryRuntimeBuilder(
            ollama_model=(
                args.model
            ),
            ollama_host=(
                args.ollama_host
            ),
            progress_callback=(
                print_progress
            ),
        )
    )

    service = (
        RepoBrainApplicationService(
            runtime_builder=(
                builder
            )
        )
    )

    transport = (
        RepoBrainTransport(
            application=(
                service
            )
        )
    )

    # ------------------------------------------------------------------
    # 1. Open repository
    # ------------------------------------------------------------------

    print_header(
        "1. OPEN REPOSITORY"
    )

    runtime = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                )
            )
        )
    )

    if not hasattr(
        runtime,
        "runtime_id",
    ):

        print(
            "FAILED: repository did not open."
        )

        print(
            runtime
        )

        return 1

    print(
        f"Runtime ID : {runtime.runtime_id}"
    )

    # ------------------------------------------------------------------
    # 2. Create session
    # ------------------------------------------------------------------

    print_header(
        "2. CREATE SESSION"
    )

    session = (
        transport.create_session(
            CreateSessionRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                )
            )
        )
    )

    if not hasattr(
        session,
        "session_id",
    ):

        print(
            "FAILED: session was not created."
        )

        print(
            session
        )

        return 1

    print(
        f"Session ID      : {session.session_id}"
    )

    print(
        f"Conversation ID : {session.conversation_id}"
    )

    print(
        f"Runtime reused  : "
        f"{session.runtime_id == runtime.runtime_id}"
    )

    # ------------------------------------------------------------------
    # 3. Deterministic structural question
    # ------------------------------------------------------------------

    print_header(
        "3. STRUCTURAL QUERY"
    )

    query = (
        "who calls _calculate_sha256?"
    )

    print(
        f"Query : {query}"
    )

    answer = (
        transport.ask(
            AskRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                ),
                session_id=(
                    session.session_id
                ),
                query=(
                    query
                ),
            )
        )
    )

    if not isinstance(
        answer,
        AskResponse,
    ):

        print(
            "FAILED: structural query returned an error."
        )

        print(
            answer
        )

        return 1

    print()
    print(
        answer.answer_text
    )

    print()
    print(
        f"Intent     : {answer.intent}"
    )

    print(
        f"Grounded   : {answer.grounded}"
    )

    print(
        f"Model      : {answer.model_name}"
    )

    print(
        f"Turn       : {answer.state.turn_number}"
    )

    if not answer.grounded:

        print(
            "FAILED: answer was not grounded."
        )

        return 1

    # ------------------------------------------------------------------
    # 4. Verify state through transport boundary
    # ------------------------------------------------------------------

    print_header(
        "4. VERIFY SESSION STATE"
    )

    state = (
        transport.get_state(
            SessionCommandRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                ),
                session_id=(
                    session.session_id
                ),
            )
        )
    )

    if not hasattr(
        state,
        "conversation_id",
    ):

        print(
            "FAILED: state request returned an error."
        )

        print(
            state
        )

        return 1

    print(
        f"Conversation ID     : {state.conversation_id}"
    )

    print(
        f"Turn number         : {state.turn_number}"
    )

    print(
        f"Current focus       : "
        f"{state.current_qualified_name or '(none)'}"
    )

    print(
        f"Relationship type   : "
        f"{state.relationship_focus_type or '(none)'}"
    )

    print(
        f"Relationship focus  : "
        f"{state.relationship_qualified_name or '(none)'}"
    )

    # ------------------------------------------------------------------
    # 5. Health + diagnostics
    # ------------------------------------------------------------------

    print_header(
        "5. HEALTH + DIAGNOSTICS"
    )

    health = (
        transport.health()
    )

    diagnostics = (
        transport.diagnostics()
    )

    print(
        f"Ready               : {health.ready}"
    )

    print(
        f"Status              : {health.status}"
    )

    print(
        f"Loaded repositories : "
        f"{health.loaded_repositories}"
    )

    print(
        f"Active sessions     : "
        f"{health.active_sessions}"
    )

    print(
        f"Diagnostic runtimes : "
        f"{len(diagnostics.repositories)}"
    )

    print(
        f"Diagnostic sessions : "
        f"{len(diagnostics.sessions)}"
    )

    if not health.ready:

        print(
            "FAILED: application health is not ready."
        )

        return 1

    if (
        health.loaded_repositories
        != 1
    ):

        print(
            "FAILED: expected exactly one loaded repository."
        )

        return 1

    if (
        health.active_sessions
        != 1
    ):

        print(
            "FAILED: expected exactly one active session."
        )

        return 1

    # ------------------------------------------------------------------
    # 6. Runtime reuse
    # ------------------------------------------------------------------

    print_header(
        "6. VERIFY RUNTIME REUSE"
    )

    runtime_again = (
        transport.open_repository(
            OpenRepositoryRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                )
            )
        )
    )

    if not hasattr(
        runtime_again,
        "runtime_id",
    ):

        print(
            "FAILED: second repository open returned an error."
        )

        print(
            runtime_again
        )

        return 1

    reused = (
        runtime_again.runtime_id
        == runtime.runtime_id
    )

    print(
        f"Runtime reused : {reused}"
    )

    if not reused:

        print(
            "FAILED: unchanged repository did not reuse runtime."
        )

        return 1

    # ------------------------------------------------------------------
    # 7. New conversation semantics
    # ------------------------------------------------------------------

    print_header(
        "7. VERIFY NEW CONVERSATION"
    )

    reset = (
        transport.new_conversation(
            SessionCommandRequest(
                repository_root=(
                    str(
                        repository_root
                    )
                ),
                session_id=(
                    session.session_id
                ),
            )
        )
    )

    if not hasattr(
        reset,
        "conversation_id",
    ):

        print(
            "FAILED: new conversation returned an error."
        )

        print(
            reset
        )

        return 1

    print(
        f"Session preserved      : "
        f"{reset.session_id == session.session_id}"
    )

    print(
        f"Conversation changed   : "
        f"{reset.conversation_id != session.conversation_id}"
    )

    if (
        reset.session_id
        != session.session_id
    ):

        print(
            "FAILED: application session ID changed."
        )

        return 1

    if (
        reset.conversation_id
        == session.conversation_id
    ):

        print(
            "FAILED: Phase-9 conversation ID did not change."
        )

        return 1

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print_header(
        "PHASE 10 ACCEPTANCE PASSED"
    )

    print(
        "Repository runtime      : PASS"
    )

    print(
        "Fingerprint-aware reuse : PASS"
    )

    print(
        "Session management      : PASS"
    )

    print(
        "Transport boundary      : PASS"
    )

    print(
        "Grounded query          : PASS"
    )

    print(
        "State routing           : PASS"
    )

    print(
        "Health + diagnostics    : PASS"
    )

    service.close_all()

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
