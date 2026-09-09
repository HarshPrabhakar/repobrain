from __future__ import annotations

import argparse

from pathlib import Path

from repobrain.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
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

from repobrain.retrieval import (
    RepositoryChunkBuilder,
    SemanticSearchEngine,
)


BENCHMARK_QUERIES = [
    (
        "Repository boundary security",
        (
            "where does RepoBrain prevent unsafe paths "
            "from leaving the repository?"
        ),
    ),
    (
        "File fingerprint",
        (
            "how does RepoBrain create a fingerprint "
            "for every file?"
        ),
    ),
    (
        "Ignored folders",
        (
            "how does the scanner avoid wasting time "
            "walking folders we don't care about?"
        ),
    ),
    (
        "Project purpose",
        (
            "what is this project trying to build?"
        ),
    ),
]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare RepoBrain semantic "
            "embedding models"
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def build_chunks(
    repository: Path,
):

    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        repository
    )

    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = analyzer.analyze(
        scan_result
    )

    resolver = (
        PythonSymbolResolver()
    )

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    builder = (
        RepositoryChunkBuilder()
    )

    return builder.build(
        scan_result=scan_result,
        analysis=resolved,
    )


def print_results(
    model_name: str,
    chunks,
    top_k: int,
) -> None:

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=model_name,
        )
    )

    engine = SemanticSearchEngine(
        chunks=chunks,
        embedding_provider=provider,
    )

    print()
    print("=" * 100)

    print(
        f"MODEL: {model_name}"
    )

    print(
        f"DEVICE: {provider.device}"
    )

    print(
        f"DIMENSION: {provider.dimension}"
    )

    print("=" * 100)

    for title, query in BENCHMARK_QUERIES:

        print()
        print(
            f"[{title}]"
        )

        print(
            f"Query: {query}"
        )

        results = engine.search(
            query,
            top_k=top_k,
        )

        for number, result in enumerate(
            results,
            start=1,
        ):

            target = (
                result.qualified_name
                or result.relative_path
            )

            print(
                f"{number:>2}. "
                f"{result.score:.4f} "
                f"{target}"
            )


def main() -> int:

    args = parse_args()

    chunks = build_chunks(
        args.repository
    )

    print()
    print(
        f"Chunks: {len(chunks)}"
    )

    print_results(
        FALLBACK_EMBEDDING_MODEL,
        chunks,
        args.top_k,
    )

    print_results(
        DEFAULT_EMBEDDING_MODEL,
        chunks,
        args.top_k,
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )