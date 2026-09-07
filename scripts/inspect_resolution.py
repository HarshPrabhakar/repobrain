from __future__ import annotations

import argparse

from collections import Counter

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


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 2.5 "
            "cross-file symbol resolution inspector"
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Path to repository",
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
    analyzer = PythonRepositoryAnalyzer()

    analysis = analyzer.analyze(
        scan_result
    )

    # Phase 2.5
    resolver = PythonSymbolResolver()

    resolved, summary = (
        resolver.resolve_repository(
            analysis
        )
    )

    print()
    print("=" * 80)
    print("REPOBRAIN")
    print(
        "Phase 2.5 - Cross-File Symbol Resolution"
    )
    print("=" * 80)

    print()
    print(
        f"Repository ID        : "
        f"{resolved.repository_id}"
    )

    print(
        f"Files analyzed       : "
        f"{resolved.files_analyzed}"
    )

    print(
        f"Symbols              : "
        f"{len(resolved.symbols)}"
    )

    print(
        f"Relationships        : "
        f"{summary.total_relationships}"
    )

    print()
    print("Resolution")
    print("-" * 80)

    print(
        f"Already linked       : "
        f"{summary.already_linked}"
    )

    print(
        f"Newly resolved       : "
        f"{summary.newly_resolved}"
    )

    print(
        f"Still unlinked       : "
        f"{summary.unresolved_local_or_external}"
    )

    linked = [
        relationship
        for relationship
        in resolved.relationships
        if relationship.target_symbol_id
        is not None
    ]

    resolution_counts = Counter(
        relationship.relationship_type.value
        for relationship in linked
    )

    print()
    print("Linked relationships by type")
    print("-" * 80)

    for (
        relationship_type,
        count,
    ) in sorted(
        resolution_counts.items()
    ):
        print(
            f"{relationship_type:<24}"
            f"{count:>8}"
        )

    print()
    print("Resolved relationships")
    print("-" * 80)

    for relationship in resolved.relationships:

        if (
            relationship.target_symbol_id
            is None
        ):
            continue

        reason = (
            relationship.metadata.get(
                "resolution_reason",
                "phase-2",
            )
        )

        print(
            f"{relationship.relationship_type.value:<14} "
            f"{relationship.source_qualified_name} "
            f"-> "
            f"{relationship.target_qualified_name} "
            f"[{reason}]"
        )

    if resolved.errors:

        print()
        print("Analysis errors")
        print("-" * 80)

        for error in resolved.errors:

            print(
                f"{error.relative_path}: "
                f"{error.error_type}: "
                f"{error.message}"
            )

    print()
    print("=" * 80)

    print(
        "Symbol resolution completed successfully."
    )

    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())