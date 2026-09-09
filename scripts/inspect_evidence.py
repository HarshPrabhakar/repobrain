from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.embeddings import (
    SentenceTransformerEmbeddingProvider,
)

from repobrain.evidence import (
    EvidenceAssembler,
    EvidenceBudget,
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


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RepoBrain Phase 6 Evidence Assembly Inspector"
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
        help="Repository question",
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=20,
        help=(
            "Number of hybrid results passed "
            "to the evidence assembler"
        ),
    )

    parser.add_argument(
        "--candidate-size",
        type=int,
        default=30,
        help=(
            "Candidate count retrieved from "
            "each base retrieval channel"
        ),
    )

    parser.add_argument(
        "--graph-seeds",
        type=int,
        default=8,
        help=(
            "Number of preliminary hybrid symbols "
            "used to seed graph expansion"
        ),
    )

    parser.add_argument(
        "--graph-depth",
        type=int,
        default=1,
        help="Maximum hybrid graph expansion depth",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=8,
        help="Maximum evidence items",
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=24_000,
        help="Maximum evidence source characters",
    )

    parser.add_argument(
        "--max-graph-relations",
        type=int,
        default=5,
        help=(
            "Maximum graph relationships attached "
            "to each evidence item"
        ),
    )

    return parser.parse_args()


# ============================================================================
# Formatting
# ============================================================================


def separator(
    character: str = "-",
    length: int = 105,
) -> None:
    print(
        character * length
    )


def heading(
    title: str,
) -> None:
    print()
    separator("=")
    print(title)
    separator("=")


# ============================================================================
# Main pipeline
# ============================================================================


def main() -> int:
    args = parse_args()

    if args.retrieval_top_k <= 0:
        raise ValueError(
            "--retrieval-top-k must be > 0"
        )

    if args.candidate_size <= 0:
        raise ValueError(
            "--candidate-size must be > 0"
        )

    if args.graph_seeds < 0:
        raise ValueError(
            "--graph-seeds must be >= 0"
        )

    if args.graph_depth < 0:
        raise ValueError(
            "--graph-depth must be >= 0"
        )

    if args.max_graph_relations < 0:
        raise ValueError(
            "--max-graph-relations must be >= 0"
        )

    # ========================================================================
    # Header
    # ========================================================================

    print()
    separator("=")

    print(
        "REPOBRAIN"
    )

    print(
        "Phase 6 - Evidence Assembly & Context Building"
    )

    separator("=")

    print()

    # ========================================================================
    # Phase 1 - Repository scan
    # ========================================================================

    print(
        "[1/10] Scanning repository..."
    )

    scanner = RepositoryScanner()

    scan_result = scanner.scan(
        args.repository
    )

    print(
        f"       Files discovered : "
        f"{len(scan_result.files)}"
    )

    # ========================================================================
    # Phase 2 - AST
    # ========================================================================

    print(
        "[2/10] Extracting Python AST intelligence..."
    )

    analyzer = PythonRepositoryAnalyzer()

    analysis = analyzer.analyze(
        scan_result
    )

    print(
        f"       Symbols           : "
        f"{len(analysis.symbols)}"
    )

    print(
        f"       Relationships     : "
        f"{len(analysis.relationships)}"
    )

    # ========================================================================
    # Phase 2.5 - Resolution
    # ========================================================================

    print(
        "[3/10] Resolving repository symbols..."
    )

    resolver = PythonSymbolResolver()

    (
        resolved_analysis,
        resolution_summary,
    ) = resolver.resolve_repository(
        analysis
    )

    print(
        f"       Resolved symbols  : "
        f"{len(resolved_analysis.symbols)}"
    )

    # ========================================================================
    # Phase 3 - Chunks
    # ========================================================================

    print(
        "[4/10] Building repository chunks..."
    )

    chunk_builder = RepositoryChunkBuilder()

    chunks = chunk_builder.build(
        scan_result=scan_result,
        analysis=resolved_analysis,
    )

    print(
        f"       Chunks            : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # Symbol retrieval
    # ========================================================================

    print(
        "[5/10] Building symbol search..."
    )

    symbol_index = SymbolIndex(
        resolved_analysis.symbols
    )

    symbol_engine = SymbolSearchEngine(
        symbol_index
    )

    print(
        f"       Indexed symbols   : "
        f"{len(symbol_index)}"
    )

    # ========================================================================
    # BM25
    # ========================================================================

    print(
        "[6/10] Building BM25 index..."
    )

    lexical_engine = BM25Index(
        chunks
    )

    print(
        f"       BM25 documents    : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # Semantic
    # ========================================================================

    print(
        "[7/10] Loading semantic embedding model..."
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    print(
        f"       Model             : "
        f"{embedding_provider.model_name}"
    )

    print(
        f"       Device            : "
        f"{embedding_provider.device}"
    )

    print(
        f"       Dimension         : "
        f"{embedding_provider.dimension}"
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
        "       Semantic index    : READY"
    )

    # ========================================================================
    # Phase 4 - Graph
    # ========================================================================

    print(
        "[8/10] Building repository knowledge graph..."
    )

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
        f"       Graph nodes       : "
        f"{graph_stats.nodes}"
    )

    print(
        f"       Graph edges       : "
        f"{graph_stats.edges}"
    )

    graph_expander = GraphEvidenceExpander(
        graph
    )

    # ========================================================================
    # Phase 5 - Hybrid retrieval
    # ========================================================================

    print(
        "[9/10] Running hybrid retrieval..."
    )

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

    hybrid_results = hybrid_engine.search(
        args.query,
        top_k=(
            args.retrieval_top_k
        ),
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

    print(
        f"       Hybrid results    : "
        f"{len(hybrid_results)}"
    )

    # ========================================================================
    # Phase 6 - Evidence assembly
    # ========================================================================

    print(
        "[10/10] Assembling bounded evidence..."
    )

    budget = EvidenceBudget(
        max_items=(
            args.max_items
        ),
        max_characters=(
            args.max_characters
        ),
    )

    assembler = EvidenceAssembler(
        chunks=chunks,
        graph=graph,
        budget=budget,
        max_graph_relations=(
            args.max_graph_relations
        ),
    )

    bundle = assembler.assemble(
        query=args.query,
        results=hybrid_results,
    )

    # ========================================================================
    # Query
    # ========================================================================

    heading(
        "QUERY"
    )

    print(
        args.query
    )

    # ========================================================================
    # Bundle summary
    # ========================================================================

    heading(
        "EVIDENCE BUNDLE"
    )

    print(
        f"Evidence items       : "
        f"{len(bundle.items)}"
    )

    print(
        f"Characters           : "
        f"{bundle.total_characters:,}"
    )

    print(
        f"Estimated tokens     : "
        f"{bundle.estimated_tokens:,}"
    )

    print(
        f"Omitted items        : "
        f"{bundle.omitted_items}"
    )

    print(
        f"Maximum items        : "
        f"{bundle.max_items}"
    )

    print(
        f"Character budget     : "
        f"{bundle.max_characters:,}"
    )

    # ========================================================================
    # Evidence items
    # ========================================================================

    if not bundle.items:
        print()
        print(
            "No evidence was assembled."
        )

        return 0

    for position, item in enumerate(
        bundle.items,
        start=1,
    ):
        print()

        separator("=")

        title = (
            item.qualified_name
            or item.relative_path
            or item.result_key
        )

        print(
            f"[{position}] {title}"
        )

        separator("=")

        print(
            f"Evidence ID        : "
            f"{item.evidence_id}"
        )

        print(
            f"Kind               : "
            f"{item.evidence_kind.value}"
        )

        print(
            f"Fused score        : "
            f"{item.fused_score:.8f}"
        )

        if item.relative_path:
            print(
                f"Path               : "
                f"{item.relative_path}"
            )

        if item.symbol_id:
            print(
                f"Symbol ID          : "
                f"{item.symbol_id}"
            )

        if item.qualified_name:
            print(
                f"Qualified name     : "
                f"{item.qualified_name}"
            )

        if item.start_line is not None:
            print(
                f"Lines              : "
                f"{item.start_line}"
                f"-"
                f"{item.end_line}"
            )

        print(
            f"Characters         : "
            f"{item.character_count:,}"
        )

        print(
            f"Estimated tokens   : "
            f"{item.estimated_tokens:,}"
        )

        print(
            "Retrieval channels : "
            + (
                ", ".join(
                    channel.value
                    for channel
                    in item.retrieval_channels
                )
                if item.retrieval_channels
                else "(none)"
            )
        )

        # --------------------------------------------------------------------
        # Graph context
        # --------------------------------------------------------------------

        print()
        print(
            "GRAPH CONTEXT"
        )

        separator()

        if not item.graph_context:
            print(
                "(none)"
            )

        else:
            for relation in (
                item.graph_context
            ):
                arrow = (
                    "->"
                    if relation.direction
                    == "outgoing"
                    else "<-"
                )

                line_info = ""

                if (
                    relation.line_number
                    is not None
                ):
                    line_info = (
                        f" [line "
                        f"{relation.line_number}]"
                    )

                print(
                    f"{arrow} "
                    f"{relation.relationship_type.value:<14} "
                    f"{relation.related_qualified_name}"
                    f"{line_info}"
                )

        # --------------------------------------------------------------------
        # Source
        # --------------------------------------------------------------------

        print()
        print(
            "SOURCE"
        )

        separator()

        print(
            item.source_text.rstrip()
        )

    # ========================================================================
    # Footer
    # ========================================================================

    print()

    separator("=")

    print(
        "Evidence assembly completed successfully."
    )

    separator("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )