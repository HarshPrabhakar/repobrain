from __future__ import annotations

import argparse

from pathlib import Path

from repobrain.indexing import (
    PythonSymbolResolver,
)

from repobrain.ingestion import (
    RepositoryScanner,
)

from repobrain.parsing import (
    PythonRepositoryAnalyzer,
)

from repobrain.retrieval import (
    BM25Index,
    RepositoryChunkBuilder,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 3B "
            "BM25 lexical code search"
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
        help="Lexical search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
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
    resolver = (
        PythonSymbolResolver()
    )

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    # Phase 3B chunking
    chunk_builder = (
        RepositoryChunkBuilder()
    )

    chunks = chunk_builder.build(
        scan_result=scan_result,
        analysis=resolved,
    )

    # BM25
    index = BM25Index(
        chunks
    )

    results = index.search(
        args.query,
        top_k=args.top_k,
        language=args.language,
    )

    print()
    print("=" * 92)
    print("REPOBRAIN")
    print(
        "Phase 3B - BM25 Lexical Search"
    )
    print("=" * 92)

    print()
    print(
        f"Query       : {args.query}"
    )

    print(
        f"Files       : "
        f"{scan_result.metadata.indexed_files}"
    )

    print(
        f"Chunks      : "
        f"{len(chunks)}"
    )

    print(
        f"Results     : "
        f"{len(results)}"
    )

    print()
    print("Search Results")
    print("-" * 92)

    if not results:

        print(
            "No lexical matches found."
        )

    else:

        for number, result in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"[{number}] "
                f"{result.chunk_type.value}"
            )

            print(
                f"    File    : "
                f"{result.relative_path}"
            )

            if result.qualified_name:

                print(
                    f"    Symbol  : "
                    f"{result.qualified_name}"
                )

            print(
                f"    Lines   : "
                f"{result.start_line}-"
                f"{result.end_line}"
            )

            print(
                f"    Language: "
                f"{result.language}"
            )

            print(
                f"    Score   : "
                f"{result.score:.4f}"
            )

            print(
                f"    Terms   : "
                f"{', '.join(result.matched_terms)}"
            )

            print(
                f"    Excerpt : "
                f"{result.excerpt}"
            )

    print()
    print("=" * 92)
    print(
        "Lexical search completed successfully."
    )
    print("=" * 92)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )