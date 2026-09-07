from __future__ import annotations

import argparse
import sys
from pathlib import Path

from repobrain.ingestion import RepositoryScanner


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 1 - "
            "scan and analyze repository structure."
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Path to the local repository.",
    )

    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print indexed files.",
    )

    return parser.parse_args()


def print_scan_result(
    result,
    show_files: bool = False,
) -> None:
    metadata = result.metadata

    print()
    print("=" * 72)
    print("REPOBRAIN")
    print("Phase 1 - Repository Scanner")
    print("=" * 72)

    print()
    print(f"Repository : {metadata.name}")
    print(f"ID         : {metadata.repository_id}")
    print(f"Root       : {metadata.root_path}")

    print()
    print("Repository summary")
    print("-" * 72)

    print(
        f"Files discovered : "
        f"{metadata.total_files_discovered}"
    )

    print(
        f"Files indexed    : "
        f"{metadata.indexed_files}"
    )

    print(
        f"Files ignored    : "
        f"{metadata.ignored_files}"
    )

    print(
        f"Primary language : "
        f"{metadata.primary_language or 'Unknown'}"
    )

    print()
    print("Languages")
    print("-" * 72)

    if metadata.languages:
        for language, count in metadata.languages.items():
            print(
                f"{language:<24}"
                f"{count:>8}"
            )

    else:
        print("No indexable files found.")

    print()
    print("Classification")
    print("-" * 72)

    print(
        f"Test files       : "
        f"{metadata.test_files}"
    )

    print(
        f"Config files     : "
        f"{metadata.config_files}"
    )

    print(
        f"Generated files  : "
        f"{metadata.generated_files}"
    )

    if show_files:
        print()
        print("Indexed files")
        print("-" * 72)

        for file in sorted(
            result.files,
            key=lambda item: item.relative_path,
        ):
            print(
                f"{file.language:<16} "
                f"{file.relative_path}"
            )

    if result.errors:
        print()
        print("Scan warnings")
        print("-" * 72)

        for error in result.errors:
            print(
                f"[{error.error_type}] "
                f"{error.path}: "
                f"{error.message}"
            )

    print()
    print("=" * 72)

    if result.errors:
        print(
            "Scan completed with warnings."
        )
    else:
        print(
            "Scan completed successfully."
        )

    print("=" * 72)
    print()


def main() -> int:
    args = parse_arguments()

    scanner = RepositoryScanner()

    try:
        result = scanner.scan(
            args.repository
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
    ) as exc:
        print(
            f"RepoBrain scan failed: {exc}",
            file=sys.stderr,
        )

        return 1

    print_scan_result(
        result=result,
        show_files=args.show_files,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())