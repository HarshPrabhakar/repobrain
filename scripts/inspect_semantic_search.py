from __future__ import annotations

import argparse

from pathlib import Path

from repobrain.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
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


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 3C "
            "semantic repository search"
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
        help="Semantic search query",
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

    parser.add_argument(
        "--model",
        type=str,
        default=(
            DEFAULT_EMBEDDING_MODEL
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "cpu, cuda, or omit for automatic selection"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding batch size",
    ),

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Disable Phase 3C.3 semantic "
            "intent-aware reranking"
        ),
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
    default=512,
    help=(
        "Maximum token length for each "
        "semantic-search chunk"
    ),
)

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    print()
    print("=" * 94)
    print("REPOBRAIN")
    print(
        "Phase 3C - Semantic Search"
    )
    print("=" * 94)

    # ---------------------------------------------------------
    # Phase 1
    # ---------------------------------------------------------

    print()
    print("[1/6] Scanning repository...")

    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        args.repository
    )

    # ---------------------------------------------------------
    # Phase 2
    # ---------------------------------------------------------

    print(
        "[2/6] Extracting Python AST intelligence..."
    )

    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = analyzer.analyze(
        scan_result
    )

    # ---------------------------------------------------------
    # Phase 2.5
    # ---------------------------------------------------------

    print(
        "[3/6] Resolving repository symbols..."
    )

    resolver = PythonSymbolResolver()

    resolved_analysis, resolution_summary = (
        resolver.resolve_repository(
            analysis
        )
)

    # ---------------------------------------------------------
    # Phase 3B chunk infrastructure
    # ---------------------------------------------------------

    print(
        "[4/6] Building repository chunks..."
    )

    chunk_builder = RepositoryChunkBuilder()

    chunks = chunk_builder.build(
        scan_result=scan_result,
        analysis=resolved_analysis,
    )

    # ---------------------------------------------------------
    # Embedding provider
    # ---------------------------------------------------------

    print(
        "[5/6] Loading embedding model..."
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=(
                args.max_seq_length
            ),
        )
    )

    print(
        f"     Batch size : "
        f"{provider.batch_size}"
    )

    print(
        f"     Max seq len : "
        f"{provider.max_seq_length}"
    )

    print(
        f"      Model     : "
        f"{provider.model_name}"
    )

    print(
        f"      Device    : "
        f"{provider.device}"
    )

    print(
        f"      Dimension : "
        f"{provider.dimension}"
    )

    # ---------------------------------------------------------
    # Semantic index
    # ---------------------------------------------------------

    print(
        "[6/6] Creating FAISS semantic index..."
    )

    semantic_engine = SemanticSearchEngine(
        chunks=chunks,
        embedding_provider=provider,
        symbols=resolved_analysis.symbols,
        relationships=resolved_analysis.relationships,
    )

    results = semantic_engine.search(
        args.query,
        top_k=args.top_k,
        language=args.language,
        rerank=(
            not args.no_rerank
        ),
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()

    print(
        f"Query       : {args.query}"
    )

    print(
        f"Files       : "
        f"{scan_result.metadata.indexed_files}"
    )

    print(
        f"Chunks      : {len(chunks)}"
    )

    print(
        f"Vector dim  : "
        f"{semantic_engine.dimension}"
    )

    print(
        f"Results     : "
        f"{len(results)}"
    )

    print()

    print("Semantic Search Results")

    print("-" * 94)

    if not results:

        print(
            "No semantic matches found."
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
                f"    Semantic : "
                f"{result.score:.4f}"
            )

            if result.ranking_score is not None:

                print(
                    f"    Ranking  : "
                    f"{result.ranking_score:.4f}"
                )

            if result.query_intent:

                print(
                    f"    Intent   : "
                    f"{result.query_intent}"
                )

            if result.source_kind:

                print(
                    f"    Source   : "
                    f"{result.source_kind}"
                )

            if result.ranking_reasons:

                print(
                    "    Reasons  :"
                )

                for reason in (
                    result.ranking_reasons
                ):

                    print(
                        f"               - {reason}"
                    )

            print(
                f"    Excerpt : "
                f"{result.excerpt}"
            )

    print()

    print("=" * 94)

    print(
        "Semantic search completed successfully."
    )

    print("=" * 94)

    print()

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )