from __future__ import annotations

from pathlib import Path

import pytest

from repobrain.application import (
    RepositoryRuntimeBuilder,
)


def test_builder_defaults() -> None:

    builder = (
        RepositoryRuntimeBuilder()
    )

    assert (
        builder.ollama_host
        == "http://localhost:11434"
    )

    assert (
        builder.ollama_model
        == "qwen2.5-coder:14b"
    )

    assert (
        builder.temperature
        == 0.0
    )

    assert builder.max_steps == 4

    assert (
        builder.retrieval_top_k
        == 20
    )

    assert (
        builder.max_evidence_items
        == 8
    )

    assert (
        builder.max_characters
        == 24_000
    )

    assert (
        builder.max_graph_relations
        == 20
    )


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "message",
    ),
    [
        (
            "max_steps",
            0,
            "max_steps must be >= 1",
        ),
        (
            "retrieval_top_k",
            0,
            "retrieval_top_k must be >= 1",
        ),
        (
            "max_evidence_items",
            0,
            "max_evidence_items must be >= 1",
        ),
        (
            "max_characters",
            0,
            "max_characters must be >= 1",
        ),
        (
            "max_graph_relations",
            0,
            "max_graph_relations must be >= 1",
        ),
    ],
)
def test_invalid_builder_limits_are_rejected(
    keyword: str,
    value: int,
    message: str,
) -> None:

    kwargs = {
        keyword: value,
    }

    with pytest.raises(
        ValueError,
        match=message,
    ):

        RepositoryRuntimeBuilder(
            **kwargs
        )


def test_missing_repository_is_rejected(
    tmp_path: Path,
) -> None:

    builder = (
        RepositoryRuntimeBuilder()
    )

    missing = (
        tmp_path
        / "missing"
    )

    with pytest.raises(
        ValueError,
        match=(
            "repository_root does not exist"
        ),
    ):

        builder.build(
            missing
        )


def test_repository_file_is_rejected(
    tmp_path: Path,
) -> None:

    builder = (
        RepositoryRuntimeBuilder()
    )

    file_path = (
        tmp_path
        / "file.txt"
    )

    file_path.write_text(
        "test",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "repository_root is not a directory"
        ),
    ):

        builder.build(
            file_path
        )


def test_progress_callback_contract() -> None:

    events: list[
        tuple[
            int,
            int,
            str,
            dict[str, object],
        ]
    ] = []

    def callback(
        step: int,
        total: int,
        message: str,
        details: dict[str, object],
    ) -> None:

        events.append(
            (
                step,
                total,
                message,
                details,
            )
        )

    builder = RepositoryRuntimeBuilder(
        progress_callback=callback
    )

    builder._progress(
        3,
        "Resolving repository symbols...",
        resolved_symbols=42,
    )

    assert events == [
        (
            3,
            12,
            "Resolving repository symbols...",
            {
                "resolved_symbols": 42,
            },
        )
    ]
