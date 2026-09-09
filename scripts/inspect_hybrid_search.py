from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.embeddings import (
    SentenceTransformerEmbeddingProvider,
)

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

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
    GraphEvidenceExpander,
    HybridRetrievalEngine,
    SemanticSearchEngine,
    SymbolSearchEngine,
)

from repobrain.retrieval.chunks import (
    RepositoryChunkBuilder,
)

from repobrain.retrieval.lexical import (
    BM25Index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 5 hybrid retrieval inspector"
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
        help="Natural-language repository query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of final hybrid results",
    )

    parser.add_argument(
        "--candidate-size",
        type=int,
        default=30,
        help="Candidate count for each base retriever",
    )

    parser.add_argument(
        "--graph-seeds",
        type=int,
        default=8,
        help="Number of preliminary symbols used as graph seeds",
    )

    parser.add_argument(
        "--graph-depth",
        type=int,
        default=1,
        help="Maximum graph expansion depth",
    )

    return parser.parse_args()


def separator() -> None:
    print("-" * 105)


def main() -> int:
    args = parse_args()

    print()
    print("=" * 105)
    print("REPOBRAIN")
    print("Phase 5 - Hybrid Retrieval & Evidence Fusion")
    print("=" * 105)
    print()

    # =========================================================
    # Phase 1
    # =========================================================

    print("[1/9] Scanning repository...")

    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        args.repository
    )

    print(
        f"      Files discovered: "
        f"{len(scan_result.files)}"
    )

    # =========================================================
    # Phase 2
    # =========================================================

    print("[2/9] Extracting Python AST intelligence...")

    analyzer = PythonRepositoryAnalyzer()

    analysis = analyzer.analyze(
        scan_result
    )

    print(
        f"      Symbols extracted: "
        f"{len(analysis.symbols)}"
    )

    print(
        f"      Relationships: "
        f"{len(analysis.relationships)}"
    )

    # =========================================================
    # Phase 2.5
    # =========================================================

    print("[3/9] Resolving repository relationships...")

    resolver = PythonSymbolResolver()

    resolved_analysis, resolution_summary = (
        resolver.resolve_repository(
            analysis
        )
    )

    print(
        f"      Resolved symbols: "
        f"{len(resolved_analysis.symbols)}"
    )

    # =========================================================
    # Phase 3 chunks
    # =========================================================

    print("[4/9] Building retrieval chunks...")

    chunk_builder = RepositoryChunkBuilder()

    chunks = chunk_builder.build(
        scan_result=scan_result,
        analysis=resolved_analysis,
    )

    print(
        f"      Chunks: "
        f"{len(chunks)}"
    )

    # =========================================================
    # Symbol retrieval
    # =========================================================

    print("[5/9] Building symbol index...")

    symbol_index = SymbolIndex(
        resolved_analysis.symbols
    )

    symbol_engine = SymbolSearchEngine(
        symbol_index
    )

    print(
        f"      Indexed symbols: "
        f"{len(symbol_index)}"
    )

    # =========================================================
    # BM25
    # =========================================================

    print("[6/9] Building BM25 lexical index...")

    lexical_engine = BM25Index(
        chunks
    )

    print(
        f"      BM25 documents: "
        f"{len(chunks)}"
    )

    # =========================================================
    # Semantic retrieval
    # =========================================================

    print("[7/9] Loading semantic embedding model...")

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    semantic_engine = SemanticSearchEngine(
        chunks,
        embedding_provider,
        symbols=(
            resolved_analysis.symbols
        ),
        relationships=(
            resolved_analysis.relationships
        ),
    )

    print(
        "      Semantic index: READY"
    )

    # =========================================================
    # Graph retrieval
    # =========================================================

    print("[8/9] Building repository knowledge graph...")

    graph = RepositoryKnowledgeGraph(
        symbols=(
            resolved_analysis.symbols
        ),
        relationships=(
            resolved_analysis.relationships
        ),
    )

    graph_stats = graph.stats()

    print(
        f"      Graph nodes: "
        f"{graph_stats.nodes}"
    )

    print(
        f"      Graph edges: "
        f"{graph_stats.edges}"
    )

    graph_expander = GraphEvidenceExpander(
        graph
    )

    # =========================================================
    # Hybrid retrieval
    # =========================================================

    print("[9/9] Running hybrid retrieval...")

    hybrid_engine = HybridRetrievalEngine(
        symbol_engine=(
            symbol_engine
        ),
        lexical_engine=(
            lexical_engine
        ),
        semantic_engine=(
            semantic_engine
        ),
        graph_expander=(
            graph_expander
        ),
    )

    results = hybrid_engine.search(
        args.query,
        top_k=args.top_k,
        candidate_size=(
            args.candidate_size
        ),
        graph_seed_count=(
            args.graph_seeds
        ),
        graph_depth=(
            args.graph_depth
        ),
    )

    # =========================================================
    # Output
    # =========================================================

    print()
    print("=" * 105)
    print("QUERY")
    print("=" * 105)
    print(args.query)

    print()
    print("=" * 105)
    print("HYBRID RESULTS")
    print("=" * 105)

    if not results:
        print("No results.")
        return 0

    for position, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"[{position}] "
            f"{result.qualified_name or result.relative_path or result.result_key}"
        )

        separator()

        print(
            f"Fused score : "
            f"{result.fused_score:.8f}"
        )

        print(
            "Channels    : "
            + ", ".join(
                channel.value
                for channel in result.channels
            )
        )

        if result.symbol_id:
            print(
                f"Symbol ID   : "
                f"{result.symbol_id}"
            )

        if result.chunk_id:
            print(
                f"Chunk ID    : "
                f"{result.chunk_id}"
            )

        if result.relative_path:
            print(
                f"Path        : "
                f"{result.relative_path}"
            )

        if result.chunk_type is not None:
            print(
                f"Chunk type  : "
                f"{result.chunk_type.value}"
            )

        if result.start_line is not None:
            print(
                f"Lines       : "
                f"{result.start_line}-"
                f"{result.end_line}"
            )

        print()
        print("Evidence")
        separator()

        for evidence in result.evidence:
            print(
                f"{evidence.channel.value:<10} "
                f"rank={evidence.rank:<3} "
                f"rrf={evidence.contribution:.8f}"
            )

            if evidence.raw_score is not None:
                print(
                    f"{'':<10} "
                    f"raw_score="
                    f"{evidence.raw_score:.8f}"
                )

            if (
                evidence.relationship_type
                is not None
            ):
                print(
                    f"{'':<10} "
                    f"relationship="
                    f"{evidence.relationship_type.value}"
                )

            if (
                evidence.hop_distance
                is not None
            ):
                print(
                    f"{'':<10} "
                    f"hop_distance="
                    f"{evidence.hop_distance}"
                )

            if (
                evidence.seed_symbol_id
                is not None
            ):
                seed_node = graph.get_node(
                    evidence.seed_symbol_id
                )

                seed_name = (
                    seed_node.qualified_name
                    if seed_node is not None
                    else evidence.seed_symbol_id
                )

                print(
                    f"{'':<10} "
                    f"seed="
                    f"{seed_name}"
                )

            if evidence.details:
                print(
                    f"{'':<10} "
                    f"details="
                    f"{evidence.details}"
                )

        if result.excerpt:
            print()
            print("Excerpt")
            separator()

            excerpt = (
                result.excerpt
                .strip()
            )

            if len(excerpt) > 600:
                excerpt = (
                    excerpt[:600]
                    + "..."
                )

            print(excerpt)

    print()
    print("=" * 105)
    print(
        f"Returned {len(results)} "
        f"hybrid results."
    )
    print("=" * 105)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )