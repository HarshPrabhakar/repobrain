from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from repobrain.ingestion import RepositoryScanner
from repobrain.parsing import PythonRepositoryAnalyzer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 2 Python AST inspector"
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Repository path",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        args.repository
    )

    analyzer = PythonRepositoryAnalyzer()

    result = analyzer.analyze(
        scan_result
    )

    print()
    print("=" * 76)
    print("REPOBRAIN")
    print("Phase 2 - Python AST Intelligence")
    print("=" * 76)

    print()
    print(
        f"Repository ID  : "
        f"{result.repository_id}"
    )

    print(
        f"Files analyzed : "
        f"{result.files_analyzed}"
    )

    print(
        f"Symbols        : "
        f"{len(result.symbols)}"
    )

    print(
        f"Relationships  : "
        f"{len(result.relationships)}"
    )

    symbol_counts = Counter(
        symbol.symbol_type.value
        for symbol in result.symbols
    )

    relationship_counts = Counter(
        relation.relationship_type.value
        for relation in result.relationships
    )

    print()
    print("Symbols")
    print("-" * 76)

    for symbol_type, count in sorted(
        symbol_counts.items()
    ):
        print(
            f"{symbol_type:<24}"
            f"{count:>8}"
        )

    print()
    print("Relationships")
    print("-" * 76)

    for relation_type, count in sorted(
        relationship_counts.items()
    ):
        print(
            f"{relation_type:<24}"
            f"{count:>8}"
        )

    print()
    print("Discovered symbols")
    print("-" * 76)

    for symbol in sorted(
        result.symbols,
        key=lambda item: (
            item.file_id,
            item.start_line,
            item.qualified_name,
        ),
    ):

        print(
            f"{symbol.symbol_type.value:<10} "
            f"{symbol.fully_qualified_name:<55} "
            f"L{symbol.start_line}-{symbol.end_line}"
        )

    print()
    print("Relationships")
    print("-" * 76)

    for relation in result.relationships:

        print(
            f"{relation.relationship_type.value:<14} "
            f"{relation.source_qualified_name} "
            f"-> "
            f"{relation.target_qualified_name} "
            f"[{relation.resolution.value}]"
        )

    if result.errors:

        print()
        print("Parse errors")
        print("-" * 76)

        for error in result.errors:

            print(
                f"{error.relative_path}: "
                f"{error.error_type}: "
                f"{error.message}"
            )

    print()
    print("=" * 76)

    if result.errors:
        print(
            "AST analysis completed with warnings."
        )
    else:
        print(
            "AST analysis completed successfully."
        )

    print("=" * 76)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())