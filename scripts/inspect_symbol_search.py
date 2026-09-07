from __future__ import annotations

import argparse

from pathlib import Path

from repobrain.indexing import (
    PythonSymbolResolver,
    SymbolIndex,
)

from repobrain.ingestion import (
    RepositoryScanner,
)

from repobrain.parsing import (
    PythonRepositoryAnalyzer,
)

from repobrain.retrieval import (
    SymbolSearchEngine,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 3A "
            "symbol-search inspector"
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Repository path",
    )

    parser.add_argument(
        "query",
        type=str,
        help="Symbol search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    # Phase 1
    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        args.repository
    )

    # Phase 2
    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = analyzer.analyze(
        scan_result
    )

    # Phase 2.5
    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    # Phase 3A
    index = SymbolIndex(
        resolved.symbols
    )

    search = SymbolSearchEngine(
        index
    )

    results = search.search(
        args.query,
        top_k=args.top_k,
    )

    print()
    print("=" * 88)
    print("REPOBRAIN")
    print(
        "Phase 3A - Symbol Search"
    )
    print("=" * 88)

    print()
    print(
        f"Query      : {args.query}"
    )

    print(
        f"Symbols    : "
        f"{len(resolved.symbols)}"
    )

    print(
        f"Results    : "
        f"{len(results)}"
    )

    print()
    print("Search Results")
    print("-" * 88)

    if not results:

        print(
            "No matching repository symbols."
        )

    else:

        for index_number, result in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"[{index_number}] "
                f"{result.symbol_type.value}"
            )

            print(
                f"    Symbol : "
                f"{result.qualified_name}"
            )

            print(
                f"    Module : "
                f"{result.module}"
            )

            print(
                f"    Lines  : "
                f"{result.start_line}-"
                f"{result.end_line}"
            )

            print(
                f"    Score  : "
                f"{result.score:.3f}"
            )

            print(
                f"    Match  : "
                f"{result.match_type}"
            )

            print(
                f"    Reason : "
                f"{result.reason}"
            )

    print()
    print("=" * 88)
    print(
        "Symbol search completed successfully."
    )
    print("=" * 88)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )