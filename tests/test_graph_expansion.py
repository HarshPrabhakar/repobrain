from __future__ import annotations

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    RelationshipType,
    ResolutionType,
    SymbolType,
)

from repobrain.retrieval.graph_expansion import (
    GraphEvidenceExpander,
)


def symbol(
    symbol_id: str,
) -> CodeSymbol:

    return CodeSymbol(
        symbol_id=symbol_id,
        repository_id="repo",
        file_id=f"file_{symbol_id}",
        name=symbol_id,
        qualified_name=(
            f"app.{symbol_id}"
        ),
        fully_qualified_name=(
            f"app.{symbol_id}"
        ),
        symbol_type=(
            SymbolType.FUNCTION
        ),
        module="app",
        start_line=1,
        end_line=10,
        parent_symbol_id=None,
        signature=None,
        docstring=None,
        is_async=False,
        decorators=[],
    )


def call(
    relationship_id: str,
    source: CodeSymbol,
    target: CodeSymbol,
) -> CodeRelationship:

    return CodeRelationship(
        relationship_id=(
            relationship_id
        ),
        repository_id="repo",
        file_id=source.file_id,
        relationship_type=(
            RelationshipType.CALLS
        ),
        source_symbol_id=(
            source.symbol_id
        ),
        target_symbol_id=(
            target.symbol_id
        ),
        source_qualified_name=(
            source.qualified_name
        ),
        target_qualified_name=(
            target.qualified_name
        ),
        line_number=1,
        resolution=(
            ResolutionType.EXACT
        ),
        metadata={},
    )


def test_graph_expansion_finds_neighbor() -> None:

    a = symbol("a")
    b = symbol("b")

    graph = (
        RepositoryKnowledgeGraph(
            symbols=[
                a,
                b,
            ],
            relationships=[
                call(
                    "ab",
                    a,
                    b,
                )
            ],
        )
    )

    results = (
        GraphEvidenceExpander(
            graph
        ).expand(
            seed_symbol_ids=[
                "a",
            ],
            max_depth=1,
        )
    )

    assert len(results) == 1

    assert (
        results[0].symbol_id
        == "b"
    )


def test_graph_expansion_respects_depth() -> None:

    a = symbol("a")
    b = symbol("b")
    c = symbol("c")

    graph = (
        RepositoryKnowledgeGraph(
            symbols=[
                a,
                b,
                c,
            ],
            relationships=[
                call(
                    "ab",
                    a,
                    b,
                ),
                call(
                    "bc",
                    b,
                    c,
                ),
            ],
        )
    )

    results = (
        GraphEvidenceExpander(
            graph
        ).expand(
            seed_symbol_ids=[
                "a",
            ],
            max_depth=1,
        )
    )

    ids = {
        result.symbol_id
        for result in results
    }

    assert "b" in ids
    assert "c" not in ids


def test_zero_depth_returns_empty() -> None:

    graph = (
        RepositoryKnowledgeGraph(
            symbols=[],
            relationships=[],
        )
    )

    assert (
        GraphEvidenceExpander(
            graph
        ).expand(
            seed_symbol_ids=[],
            max_depth=0,
        )
        == []
    )