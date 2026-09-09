from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.indexing import (
    PythonSymbolResolver,
)

from repobrain.ingestion import (
    RepositoryScanner,
)

from repobrain.parsing import (
    PythonRepositoryAnalyzer,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 4 "
            "knowledge graph inspector"
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Repository path",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=(
            "Qualified name or short symbol "
            "name to inspect"
        ),
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help=(
            "Traversal depth when --symbol "
            "is supplied"
        ),
    )

    return parser.parse_args()


def find_symbol(
    *,
    query: str,
    symbols,
):

    normalized = (
        query.strip().casefold()
    )

    exact = [
        symbol
        for symbol in symbols
        if (
            symbol.qualified_name
            .casefold()
            == normalized
            or symbol.name
            .casefold()
            == normalized
        )
    ]

    if len(exact) == 1:
        return exact[0]

    suffix = [
        symbol
        for symbol in symbols
        if (
            symbol.qualified_name
            .casefold()
            .endswith(
                f".{normalized}"
            )
        )
    ]

    if len(suffix) == 1:
        return suffix[0]

    return None


def main() -> int:

    args = parse_args()

    print()
    print(
        "=" * 94
    )
    print(
        "REPOBRAIN"
    )
    print(
        "Phase 4 - Code Knowledge Graph"
    )
    print(
        "=" * 94
    )
    print()

    # =========================================================
    # Phase 1
    # =========================================================

    print(
        "[1/4] Scanning repository..."
    )

    scanner = (
        RepositoryScanner()
    )

    scan_result = scanner.scan(
        args.repository
    )

    # =========================================================
    # Phase 2
    # =========================================================

    print(
        "[2/4] Extracting Python AST intelligence..."
    )

    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = analyzer.analyze(
        scan_result
    )

    # =========================================================
    # Phase 2.5
    # =========================================================

    print(
        "[3/4] Resolving repository symbols..."
    )

    resolver = (
        PythonSymbolResolver()
    )

    resolved_analysis, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    # =========================================================
    # Phase 4
    # =========================================================

    print(
        "[4/4] Building knowledge graph..."
    )

    graph = (
        RepositoryKnowledgeGraph(
            symbols=(
                resolved_analysis.symbols
            ),

            relationships=(
                resolved_analysis.relationships
            ),
        )
    )

    stats = graph.stats()

    print()
    print(
        f"Files                    : "
        f"{len(scan_result.files)}"
    )

    print(
        f"Symbols                  : "
        f"{len(resolved_analysis.symbols)}"
    )

    print(
        f"Relationships            : "
        f"{len(resolved_analysis.relationships)}"
    )

    print(
        f"Graph nodes              : "
        f"{stats.nodes}"
    )

    print(
        f"Graph edges              : "
        f"{stats.edges}"
    )

    print(
        f"Unresolved/non-graph rel : "
        f"{stats.unresolved_relationships}"
    )

    print()
    print(
        "Edges by type"
    )
    print(
        "-" * 94
    )

    for (
        relationship_type,
        count,
    ) in stats.edges_by_type.items():

        print(
            f"{relationship_type:<20} "
            f"{count}"
        )

    if args.symbol is None:

        print()
        print(
            "=" * 94
        )
        print(
            "Knowledge graph built successfully."
        )
        print(
            "=" * 94
        )

        return 0

    symbol = find_symbol(
        query=args.symbol,
        symbols=(
            resolved_analysis.symbols
        ),
    )

    if symbol is None:

        print()
        print(
            "Symbol not found or ambiguous:"
        )
        print(
            args.symbol
        )

        return 1

    print()
    print(
        "=" * 94
    )

    print(
        "SYMBOL"
    )

    print(
        "=" * 94
    )

    print(
        f"Name      : "
        f"{symbol.name}"
    )

    print(
        f"Qualified : "
        f"{symbol.qualified_name}"
    )

    print(
        f"Type      : "
        f"{symbol.symbol_type.value}"
    )

    print()

    # =========================================================
    # Callers
    # =========================================================

    callers = graph.callers(
        symbol.symbol_id
    )

    print(
        "CALLERS"
    )

    print(
        "-" * 94
    )

    if callers:

        for node in callers:

            print(
                f"- {node.qualified_name}"
            )

    else:

        print(
            "(none)"
        )

    print()

    # =========================================================
    # Callees
    # =========================================================

    callees = graph.callees(
        symbol.symbol_id
    )

    print(
        "CALLEES"
    )

    print(
        "-" * 94
    )

    if callees:

        for node in callees:

            print(
                f"- {node.qualified_name}"
            )

    else:

        print(
            "(none)"
        )

    print()

    # =========================================================
    # Traversal
    # =========================================================

    traversal = graph.traverse(
        symbol.symbol_id,
        max_depth=args.depth,
    )

    print(
        f"NEIGHBORHOOD "
        f"(depth={args.depth})"
    )

    print(
        "-" * 94
    )

    for step in traversal:

        if step.depth == 0:

            print(
                f"[0] "
                f"{step.symbol.qualified_name}"
            )

            continue

        relationship = (
            step.via_relationship
        )

        relation_name = (
            relationship.relationship_type.value
            if relationship is not None
            else "UNKNOWN"
        )

        print(
            f"[{step.depth}] "
            f"{step.direction:<8} "
            f"{relation_name:<14} "
            f"{step.symbol.qualified_name}"
        )

    print()

    print(
        "=" * 94
    )

    print(
        "Graph inspection completed successfully."
    )

    print(
        "=" * 94
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )