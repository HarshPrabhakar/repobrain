from __future__ import annotations

import hashlib
from pathlib import Path

from repobrain.application import (
    ConversationSessionManager,
    RepoBrainApplicationService,
    RepositoryRuntime,
    RepositoryRuntimeRegistry,
)
from repobrain.models.application import (
    RepositoryFingerprint,
    RepositoryRuntimeDiagnostics,
)


class FakeAgentRunner:

    def __call__(
        self,
        query: str,
    ) -> object:

        raise AssertionError(
            "Agent must not run during diagnostics tests."
        )


class FakeAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> object:

        raise AssertionError(
            "Answer generator must not run during diagnostics tests."
        )


def fingerprint_for(
    root: Path,
) -> RepositoryFingerprint:

    payload = (
        root
        .joinpath(
            "marker.txt"
        )
        .read_bytes()
    )

    return RepositoryFingerprint(
        value=(
            hashlib.sha256(
                payload
            )
            .hexdigest()
        ),
        file_count=1,
        total_size_bytes=(
            len(
                payload
            )
        ),
    )


def diagnostics_for() -> RepositoryRuntimeDiagnostics:

    return RepositoryRuntimeDiagnostics(
        files_discovered=104,
        symbols=894,
        relationships=5428,
        chunks=900,
        bm25_documents=900,
        graph_nodes=894,
        graph_edges=2182,
        embedding_model=(
            "nomic-ai/CodeRankEmbed"
        ),
        embedding_device="cuda",
        embedding_dimension=768,
        llm_model=(
            "qwen2.5-coder:14b"
        ),
    )


class RuntimeFactory:

    def __init__(
        self,
    ) -> None:

        self.calls = 0

    def __call__(
        self,
        root: Path,
    ) -> RepositoryRuntime:

        self.calls += 1

        return RepositoryRuntime(
            repository_root=root,
            agent_runner=(
                FakeAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
            fingerprint=(
                fingerprint_for(
                    root
                )
            ),
            diagnostics=(
                diagnostics_for()
            ),
        )


def make_service(
    root: Path,
) -> RepoBrainApplicationService:

    factory = (
        RuntimeFactory()
    )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                fingerprint_for
            ),
        )
    )

    return RepoBrainApplicationService(
        runtime_registry=registry,
        session_manager=(
            ConversationSessionManager()
        ),
    )


def test_health_is_ready_with_no_loaded_repository(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    health = (
        service.health()
    )

    assert health.ready is True

    assert (
        health.status
        == "ready"
    )

    assert (
        health.loaded_repositories
        == 0
    )

    assert (
        health.active_sessions
        == 0
    )


def test_health_counts_loaded_repository_and_session(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    service.create_session(
        tmp_path
    )

    health = (
        service.health()
    )

    assert (
        health.loaded_repositories
        == 1
    )

    assert (
        health.active_sessions
        == 1
    )


def test_repository_diagnostics_exposes_build_metadata(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    session = (
        service.create_session(
            tmp_path
        )
    )

    result = (
        service.repository_diagnostics(
            tmp_path
        )
    )

    assert result is not None

    assert (
        result.runtime.runtime_id
        == session.runtime_id
    )

    assert (
        result.active_sessions
        == 1
    )

    assert (
        result.runtime_diagnostics
        is not None
    )

    assert (
        result.runtime_diagnostics
        .embedding_model
        == "nomic-ai/CodeRankEmbed"
    )

    assert (
        result.runtime_diagnostics
        .embedding_device
        == "cuda"
    )

    assert (
        result.runtime_diagnostics
        .embedding_dimension
        == 768
    )

    assert (
        result.runtime_diagnostics
        .graph_nodes
        == 894
    )

    assert (
        result.runtime_diagnostics
        .graph_edges
        == 2182
    )


def test_repository_diagnostics_returns_none_when_not_loaded(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    result = (
        service.repository_diagnostics(
            tmp_path
        )
    )

    assert result is None


def test_session_diagnostics_reports_current_runtime(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    session = (
        service.create_session(
            tmp_path
        )
    )

    result = (
        service.session_diagnostics(
            session.session_id
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

    assert (
        result.session.session_id
        == session.session_id
    )


def test_session_diagnostics_detects_rebuilt_runtime(
    tmp_path: Path,
) -> None:

    marker = (
        tmp_path
        / "marker.txt"
    )

    marker.write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    session = (
        service.create_session(
            tmp_path
        )
    )

    marker.write_text(
        "two",
        encoding="utf-8",
    )

    service.open_repository(
        tmp_path
    )

    result = (
        service.session_diagnostics(
            session.session_id
        )
    )

    assert (
        result.runtime_loaded
        is True
    )

    assert (
        result.runtime_current
        is False
    )


def test_full_diagnostics_contains_repository_and_sessions(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    service = (
        make_service(
            tmp_path
        )
    )

    first = (
        service.create_session(
            tmp_path
        )
    )

    second = (
        service.create_session(
            tmp_path
        )
    )

    result = (
        service.diagnostics()
    )

    assert (
        result.health.ready
        is True
    )

    assert (
        result.health.loaded_repositories
        == 1
    )

    assert (
        result.health.active_sessions
        == 2
    )

    assert (
        len(
            result.repositories
        )
        == 1
    )

    assert (
        result.repositories[0]
        .active_sessions
        == 2
    )

    session_ids = {
        item.session_id
        for item
        in result.sessions
    }

    assert session_ids == {
        first.session_id,
        second.session_id,
    }


def test_runtime_without_extended_diagnostics_is_supported(
    tmp_path: Path,
) -> None:

    (
        tmp_path
        / "marker.txt"
    ).write_text(
        "one",
        encoding="utf-8",
    )

    def factory(
        root: Path,
    ) -> RepositoryRuntime:

        return RepositoryRuntime(
            repository_root=root,
            agent_runner=(
                FakeAgentRunner()
            ),
            answer_generator=(
                FakeAnswerGenerator()
            ),
            fingerprint=(
                fingerprint_for(
                    root
                )
            ),
        )

    registry = (
        RepositoryRuntimeRegistry(
            runtime_factory=factory,
            fingerprint_provider=(
                fingerprint_for
            ),
        )
    )

    service = (
        RepoBrainApplicationService(
            runtime_registry=registry,
            session_manager=(
                ConversationSessionManager()
            ),
        )
    )

    service.open_repository(
        tmp_path
    )

    result = (
        service.repository_diagnostics(
            tmp_path
        )
    )

    assert result is not None

    assert (
        result.runtime_diagnostics
        is None
    )
