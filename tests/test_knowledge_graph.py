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


def make_symbol(
    symbol_id: str,
    name: str,
    *,
    symbol_type: SymbolType = SymbolType.FUNCTION,
) -> CodeSymbol:

    return CodeSymbol(
        symbol_id=symbol_id,

        repository_id="repo_test",

        file_id=(
            f"file_{symbol_id}"
        ),

        name=name,

        qualified_name=(
            f"app.{name}"
        ),

        fully_qualified_name=(
            f"app.{name}"
        ),

        symbol_type=(
            symbol_type
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


def make_relationship(
    relationship_id: str,
    *,
    relationship_type: RelationshipType,
    source: CodeSymbol,
    target: CodeSymbol | None,
) -> CodeRelationship:

    return CodeRelationship(
        relationship_id=(
            relationship_id
        ),

        repository_id="repo_test",

        file_id=(
            source.file_id
        ),

        relationship_type=(
            relationship_type
        ),

        source_symbol_id=(
            source.symbol_id
        ),

        target_symbol_id=(
            target.symbol_id
            if target is not None
            else None
        ),

        source_qualified_name=(
            source.qualified_name
        ),

        target_qualified_name=(
            target.qualified_name
            if target is not None
            else "external.unknown"
        ),

        line_number=5,

        resolution=(
            ResolutionType.EXACT
            if target is not None
            else ResolutionType.UNRESOLVED
        ),

        metadata={},
    )


def test_graph_creates_nodes() -> None:

    a = make_symbol(
        "a",
        "a",
    )

    b = make_symbol(
        "b",
        "b",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            a,
            b,
        ],
        relationships=[],
    )

    assert len(graph) == 2

    assert (
        graph.get_node("a")
        is not None
    )


def test_graph_creates_resolved_edge() -> None:

    caller = make_symbol(
        "caller",
        "caller",
    )

    callee = make_symbol(
        "callee",
        "callee",
    )

    relationship = make_relationship(
        "rel_call",
        relationship_type=(
            RelationshipType.CALLS
        ),
        source=caller,
        target=callee,
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            caller,
            callee,
        ],
        relationships=[
            relationship,
        ],
    )

    stats = graph.stats()

    assert stats.edges == 1

    assert (
        stats.edges_by_type[
            "CALLS"
        ]
        == 1
    )


def test_unresolved_relationship_not_added_as_edge() -> None:

    caller = make_symbol(
        "caller",
        "caller",
    )

    relationship = make_relationship(
        "rel_external",
        relationship_type=(
            RelationshipType.CALLS
        ),
        source=caller,
        target=None,
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            caller,
        ],
        relationships=[
            relationship,
        ],
    )

    assert graph.stats().edges == 0

    assert (
        graph.stats()
        .unresolved_relationships
        == 1
    )


def test_callees() -> None:

    caller = make_symbol(
        "caller",
        "caller",
    )

    callee = make_symbol(
        "callee",
        "callee",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            caller,
            callee,
        ],

        relationships=[
            make_relationship(
                "rel",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=caller,
                target=callee,
            )
        ],
    )

    results = graph.callees(
        caller.symbol_id
    )

    assert [
        node.symbol_id
        for node in results
    ] == [
        "callee"
    ]


def test_callers() -> None:

    caller = make_symbol(
        "caller",
        "caller",
    )

    callee = make_symbol(
        "callee",
        "callee",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            caller,
            callee,
        ],

        relationships=[
            make_relationship(
                "rel",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=caller,
                target=callee,
            )
        ],
    )

    results = graph.callers(
        callee.symbol_id
    )

    assert [
        node.symbol_id
        for node in results
    ] == [
        "caller"
    ]


def test_parents() -> None:

    child = make_symbol(
        "child",
        "Child",
        symbol_type=(
            SymbolType.CLASS
        ),
    )

    parent = make_symbol(
        "parent",
        "Parent",
        symbol_type=(
            SymbolType.CLASS
        ),
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            child,
            parent,
        ],

        relationships=[
            make_relationship(
                "inherit",
                relationship_type=(
                    RelationshipType.INHERITS
                ),
                source=child,
                target=parent,
            )
        ],
    )

    assert (
        graph.parents(
            child.symbol_id
        )[0].symbol_id
        == parent.symbol_id
    )


def test_children() -> None:

    child = make_symbol(
        "child",
        "Child",
        symbol_type=(
            SymbolType.CLASS
        ),
    )

    parent = make_symbol(
        "parent",
        "Parent",
        symbol_type=(
            SymbolType.CLASS
        ),
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            child,
            parent,
        ],

        relationships=[
            make_relationship(
                "inherit",
                relationship_type=(
                    RelationshipType.INHERITS
                ),
                source=child,
                target=parent,
            )
        ],
    )

    assert (
        graph.children(
            parent.symbol_id
        )[0].symbol_id
        == child.symbol_id
    )


def test_defined_symbols() -> None:

    module = make_symbol(
        "module",
        "module",
        symbol_type=(
            SymbolType.MODULE
        ),
    )

    function = make_symbol(
        "function",
        "function",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            module,
            function,
        ],

        relationships=[
            make_relationship(
                "defines",
                relationship_type=(
                    RelationshipType.DEFINES
                ),
                source=module,
                target=function,
            )
        ],
    )

    assert (
        graph.defined_symbols(
            module.symbol_id
        )[0].symbol_id
        == function.symbol_id
    )


def test_neighbors_returns_direction() -> None:

    caller = make_symbol(
        "caller",
        "caller",
    )

    callee = make_symbol(
        "callee",
        "callee",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            caller,
            callee,
        ],

        relationships=[
            make_relationship(
                "call",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=caller,
                target=callee,
            )
        ],
    )

    outgoing = graph.neighbors(
        caller.symbol_id,
        direction="outgoing",
    )

    incoming = graph.neighbors(
        callee.symbol_id,
        direction="incoming",
    )

    assert (
        outgoing[0].direction
        == "outgoing"
    )

    assert (
        incoming[0].direction
        == "incoming"
    )


def test_neighbors_can_filter_relationship_type() -> None:

    source = make_symbol(
        "source",
        "source",
    )

    called = make_symbol(
        "called",
        "called",
    )

    defined = make_symbol(
        "defined",
        "defined",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            source,
            called,
            defined,
        ],

        relationships=[
            make_relationship(
                "call",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=source,
                target=called,
            ),

            make_relationship(
                "define",
                relationship_type=(
                    RelationshipType.DEFINES
                ),
                source=source,
                target=defined,
            ),
        ],
    )

    neighbors = graph.neighbors(
        source.symbol_id,
        direction="outgoing",
        relationship_type=(
            RelationshipType.CALLS
        ),
    )

    assert len(neighbors) == 1

    assert (
        neighbors[0]
        .symbol.symbol_id
        == called.symbol_id
    )


def test_traverse_depth_zero() -> None:

    a = make_symbol(
        "a",
        "a",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            a,
        ],
        relationships=[],
    )

    results = graph.traverse(
        a.symbol_id,
        max_depth=0,
    )

    assert len(results) == 1

    assert results[0].depth == 0


def test_traverse_multiple_hops() -> None:

    a = make_symbol(
        "a",
        "a",
    )

    b = make_symbol(
        "b",
        "b",
    )

    c = make_symbol(
        "c",
        "c",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            a,
            b,
            c,
        ],

        relationships=[
            make_relationship(
                "ab",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=a,
                target=b,
            ),

            make_relationship(
                "bc",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=b,
                target=c,
            ),
        ],
    )

    results = graph.traverse(
        a.symbol_id,
        max_depth=2,
        direction="outgoing",
        relationship_types={
            RelationshipType.CALLS,
        },
    )

    by_id = {
        step.symbol.symbol_id: (
            step.depth
        )
        for step
        in results
    }

    assert by_id == {
        "a": 0,
        "b": 1,
        "c": 2,
    }


def test_traverse_handles_cycles() -> None:

    a = make_symbol(
        "a",
        "a",
    )

    b = make_symbol(
        "b",
        "b",
    )

    graph = RepositoryKnowledgeGraph(
        symbols=[
            a,
            b,
        ],

        relationships=[
            make_relationship(
                "ab",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=a,
                target=b,
            ),

            make_relationship(
                "ba",
                relationship_type=(
                    RelationshipType.CALLS
                ),
                source=b,
                target=a,
            ),
        ],
    )

    results = graph.traverse(
        a.symbol_id,
        max_depth=10,
    )

    assert len(results) == 2


def test_unknown_symbol_returns_empty_traversal() -> None:

    graph = RepositoryKnowledgeGraph(
        symbols=[],
        relationships=[],
    )

    assert (
        graph.traverse(
            "does_not_exist"
        )
        == []
    )