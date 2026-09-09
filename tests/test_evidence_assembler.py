from __future__ import annotations

from types import SimpleNamespace

from repobrain.evidence import (
    EvidenceAssembler,
)

from repobrain.models.evidence import (
    EvidenceKind,
)

from repobrain.models.hybrid import (
    HybridSearchResult,
    RetrievalChannel,
)


def make_chunk(
    *,
    chunk_id: str,
    symbol_id: str | None,
    path: str,
    start_line: int,
    end_line: int,
    source: str,
):

    return SimpleNamespace(
        chunk_id=chunk_id,
        symbol_id=symbol_id,
        file_id=(
            f"file_{chunk_id}"
        ),
        relative_path=path,
        qualified_name=(
            f"app.{symbol_id}"
            if symbol_id
            else None
        ),
        start_line=start_line,
        end_line=end_line,
        source=source,
    )


def make_result(
    *,
    symbol_id: str | None,
    chunk_id: str | None,
    path: str,
    start_line: int,
    end_line: int,
    score: float = 0.1,
):

    return HybridSearchResult(
        result_key=(
            f"symbol:{symbol_id}"
            if symbol_id
            else f"chunk:{chunk_id}"
        ),
        symbol_id=symbol_id,
        chunk_id=chunk_id,
        file_id=(
            f"file_{chunk_id}"
            if chunk_id
            else None
        ),
        relative_path=path,
        qualified_name=(
            f"app.{symbol_id}"
            if symbol_id
            else None
        ),
        start_line=start_line,
        end_line=end_line,
        fused_score=score,
        channels=[
            RetrievalChannel.SEMANTIC,
        ],
        evidence=[],
    )


def test_assembler_recovers_chunk_source() -> None:

    chunk = make_chunk(
        chunk_id="chunk_a",
        symbol_id="a",
        path="app/a.py",
        start_line=1,
        end_line=3,
        source="def a():\n    return 1",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk,
        ]
    )

    bundle = assembler.assemble(
        query="where is a",
        results=[
            make_result(
                symbol_id="a",
                chunk_id="chunk_a",
                path="app/a.py",
                start_line=1,
                end_line=3,
            )
        ],
    )

    assert len(bundle.items) == 1

    assert (
        "def a"
        in bundle.items[0].source_text
    )


def test_symbol_id_used_to_find_chunk() -> None:

    chunk = make_chunk(
        chunk_id="chunk_a",
        symbol_id="a",
        path="app/a.py",
        start_line=1,
        end_line=3,
        source="def a(): pass",
    )

    result = make_result(
        symbol_id="a",
        chunk_id=None,
        path="app/a.py",
        start_line=1,
        end_line=3,
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk,
        ]
    )

    bundle = assembler.assemble(
        query="a",
        results=[
            result,
        ],
    )

    assert len(bundle.items) == 1


def test_duplicate_symbol_removed() -> None:

    chunk = make_chunk(
        chunk_id="chunk_a",
        symbol_id="a",
        path="app/a.py",
        start_line=1,
        end_line=3,
        source="def a(): pass",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk,
        ]
    )

    result = make_result(
        symbol_id="a",
        chunk_id="chunk_a",
        path="app/a.py",
        start_line=1,
        end_line=3,
    )

    bundle = assembler.assemble(
        query="a",
        results=[
            result,
            result,
        ],
    )

    assert len(bundle.items) == 1


def test_overlapping_source_removed() -> None:

    chunk_a = make_chunk(
        chunk_id="a",
        symbol_id=None,
        path="app/a.py",
        start_line=1,
        end_line=20,
        source="first",
    )

    chunk_b = make_chunk(
        chunk_id="b",
        symbol_id=None,
        path="app/a.py",
        start_line=10,
        end_line=30,
        source="second",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk_a,
            chunk_b,
        ]
    )

    bundle = assembler.assemble(
        query="example",
        results=[
            make_result(
                symbol_id=None,
                chunk_id="a",
                path="app/a.py",
                start_line=1,
                end_line=20,
            ),
            make_result(
                symbol_id=None,
                chunk_id="b",
                path="app/a.py",
                start_line=10,
                end_line=30,
            ),
        ],
    )

    assert len(bundle.items) == 1


def test_non_overlapping_source_kept() -> None:

    chunk_a = make_chunk(
        chunk_id="a",
        symbol_id=None,
        path="app/a.py",
        start_line=1,
        end_line=5,
        source="first",
    )

    chunk_b = make_chunk(
        chunk_id="b",
        symbol_id=None,
        path="app/a.py",
        start_line=10,
        end_line=15,
        source="second",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk_a,
            chunk_b,
        ]
    )

    bundle = assembler.assemble(
        query="example",
        results=[
            make_result(
                symbol_id=None,
                chunk_id="a",
                path="app/a.py",
                start_line=1,
                end_line=5,
            ),
            make_result(
                symbol_id=None,
                chunk_id="b",
                path="app/a.py",
                start_line=10,
                end_line=15,
            ),
        ],
    )

    assert len(bundle.items) == 2


def test_test_file_classification() -> None:

    chunk = make_chunk(
        chunk_id="test",
        symbol_id="test_a",
        path="tests/test_a.py",
        start_line=1,
        end_line=5,
        source="def test_a(): pass",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk,
        ]
    )

    bundle = assembler.assemble(
        query="test",
        results=[
            make_result(
                symbol_id="test_a",
                chunk_id="test",
                path="tests/test_a.py",
                start_line=1,
                end_line=5,
            )
        ],
    )

    assert (
        bundle.items[0].evidence_kind
        == EvidenceKind.TEST
    )


def test_documentation_classification() -> None:

    chunk = make_chunk(
        chunk_id="readme",
        symbol_id=None,
        path="README.md",
        start_line=1,
        end_line=10,
        source="# RepoBrain",
    )

    assembler = EvidenceAssembler(
        chunks=[
            chunk,
        ]
    )

    bundle = assembler.assemble(
        query="project",
        results=[
            make_result(
                symbol_id=None,
                chunk_id="readme",
                path="README.md",
                start_line=1,
                end_line=10,
            )
        ],
    )

    assert (
        bundle.items[0].evidence_kind
        == EvidenceKind.DOCUMENTATION
    )


def test_empty_query_rejected() -> None:

    assembler = EvidenceAssembler(
        chunks=[]
    )

    try:
        assembler.assemble(
            query="   ",
            results=[],
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError."
    )