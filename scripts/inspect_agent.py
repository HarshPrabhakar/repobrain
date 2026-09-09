from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.agent import (
    AgentToolbox,
    RepoBrainAgentOrchestrator,
)

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
            "RepoBrain Phase 7 Agent Orchestrator Inspector"
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
        "--max-steps",
        type=int,
        default=3,
        help="Maximum agent investigation steps",
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=20,
        help=(
            "Number of hybrid results used during "
            "agent evidence retrieval"
        ),
    )

    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=8,
        help="Maximum final evidence items",
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=24_000,
        help="Maximum evidence source characters",
    )

    parser.add_argument(
        "--graph-depth",
        type=int,
        default=1,
        help="Hybrid graph expansion depth",
    )

    parser.add_argument(
        "--graph-seeds",
        type=int,
        default=8,
        help="Number of hybrid graph seed symbols",
    )

    parser.add_argument(
        "--candidate-size",
        type=int,
        default=30,
        help="Candidate count per retrieval channel",
    )

    return parser.parse_args()


# ============================================================================
# Formatting helpers
# ============================================================================


WIDTH = 110


def separator(
    character: str = "-",
) -> None:
    print(
        character * WIDTH
    )


def heading(
    title: str,
) -> None:
    print()
    separator("=")
    print(title)
    separator("=")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()

    if not args.query.strip():
        raise ValueError(
            "query must not be empty."
        )

    if args.max_steps <= 0:
        raise ValueError(
            "--max-steps must be > 0."
        )

    if args.retrieval_top_k <= 0:
        raise ValueError(
            "--retrieval-top-k must be > 0."
        )

    if args.max_evidence_items <= 0:
        raise ValueError(
            "--max-evidence-items must be > 0."
        )

    if args.max_characters <= 0:
        raise ValueError(
            "--max-characters must be > 0."
        )

    if args.graph_depth < 0:
        raise ValueError(
            "--graph-depth must be >= 0."
        )

    if args.graph_seeds < 0:
        raise ValueError(
            "--graph-seeds must be >= 0."
        )

    if args.candidate_size <= 0:
        raise ValueError(
            "--candidate-size must be > 0."
        )

    print()
    separator("=")
    print("REPOBRAIN")
    print("Phase 7 - Deterministic Agent Orchestrator")
    separator("=")
    print()

    # ========================================================================
    # Phase 1 - Scan
    # ========================================================================

    print(
        "[1/11] Scanning repository..."
    )

    scanner = (
        RepositoryScanner()
    )

    scan_result = scanner.scan(
        args.repository
    )

    print(
        f"        Files discovered : "
        f"{len(scan_result.files)}"
    )

    # ========================================================================
    # Phase 2 - AST
    # ========================================================================

    print(
        "[2/11] Extracting Python AST intelligence..."
    )

    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = analyzer.analyze(
        scan_result
    )

    print(
        f"        Symbols          : "
        f"{len(analysis.symbols)}"
    )

    print(
        f"        Relationships    : "
        f"{len(analysis.relationships)}"
    )

    # ========================================================================
    # Phase 2.5 - Resolution
    # ========================================================================

    print(
        "[3/11] Resolving repository symbols..."
    )

    resolver = (
        PythonSymbolResolver()
    )

    (
        resolved_analysis,
        resolution_summary,
    ) = resolver.resolve_repository(
        analysis
    )

    print(
        f"        Resolved symbols : "
        f"{len(resolved_analysis.symbols)}"
    )

    # ========================================================================
    # Phase 3 - Chunks
    # ========================================================================

    print(
        "[4/11] Building repository chunks..."
    )

    chunk_builder = (
        RepositoryChunkBuilder()
    )

    chunks = chunk_builder.build(
        scan_result=scan_result,
        analysis=resolved_analysis,
    )

    print(
        f"        Chunks           : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # Symbol index
    # ========================================================================

    print(
        "[5/11] Building symbol index..."
    )

    symbol_index = (
        SymbolIndex(
            resolved_analysis.symbols
        )
    )

    symbol_engine = (
        SymbolSearchEngine(
            symbol_index
        )
    )

    print(
        f"        Indexed symbols  : "
        f"{len(symbol_index)}"
    )

    # ========================================================================
    # BM25
    # ========================================================================

    print(
        "[6/11] Building BM25 index..."
    )

    lexical_engine = (
        BM25Index(
            chunks
        )
    )

    print(
        f"        BM25 documents   : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # Semantic
    # ========================================================================

    print(
        "[7/11] Loading semantic model..."
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    print(
        f"        Model            : "
        f"{embedding_provider.model_name}"
    )

    print(
        f"        Device           : "
        f"{embedding_provider.device}"
    )

    print(
        f"        Dimension        : "
        f"{embedding_provider.dimension}"
    )

    semantic_engine = (
        SemanticSearchEngine(
            chunks,
            embedding_provider,
            symbols=(
                resolved_analysis.symbols
            ),
            relationships=(
                resolved_analysis.relationships
            ),
        )
    )

    print(
        "        Semantic index   : READY"
    )

    # ========================================================================
    # Graph
    # ========================================================================

    print(
        "[8/11] Building knowledge graph..."
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

    graph_stats = graph.stats()

    print(
        f"        Graph nodes      : "
        f"{graph_stats.nodes}"
    )

    print(
        f"        Graph edges      : "
        f"{graph_stats.edges}"
    )

    graph_expander = (
        GraphEvidenceExpander(
            graph
        )
    )

    # ========================================================================
    # Hybrid retrieval
    # ========================================================================

    print(
        "[9/11] Building hybrid retrieval engine..."
    )

    hybrid_engine = (
        HybridRetrievalEngine(
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
    )

    print(
        "        Hybrid engine    : READY"
    )

    # ========================================================================
    # Evidence assembler
    # ========================================================================

    print(
        "[10/11] Building evidence assembler..."
    )

    budget = (
        EvidenceBudget(
            max_items=(
                args.max_evidence_items
            ),
            max_characters=(
                args.max_characters
            ),
        )
    )

    evidence_assembler = (
        EvidenceAssembler(
            chunks=chunks,
            graph=graph,
            budget=budget,
            max_graph_relations=5,
        )
    )

    print(
        "        Evidence layer   : READY"
    )

    # ========================================================================
    # Agent
    # ========================================================================

    print(
        "[11/11] Running RepoBrain agent..."
    )

    toolbox = (
        AgentToolbox(
            hybrid_engine=(
                hybrid_engine
            ),
            evidence_assembler=(
                evidence_assembler
            ),
            graph=graph,
            symbol_index=(
                symbol_index
            ),
        )
    )

    agent = (
        RepoBrainAgentOrchestrator(
            toolbox=toolbox,
            max_steps=(
                args.max_steps
            ),
            retrieval_top_k=(
                args.retrieval_top_k
            ),
        )
    )

    result = agent.run(
        args.query
    )

    # ========================================================================
    # Output - Query
    # ========================================================================

    heading(
        "QUERY"
    )

    print(
        args.query
    )

    # ========================================================================
    # Output - Agent state
    # ========================================================================

    heading(
        "AGENT STATE"
    )

    state = result.state

    print(
        f"Intent              : "
        f"{state.intent.value}"
    )

    print(
        f"Status              : "
        f"{state.status.value}"
    )

    print(
        f"Steps used          : "
        f"{state.step_count}"
    )

    print(
        f"Maximum steps       : "
        f"{state.max_steps}"
    )

    print(
        f"Remaining steps     : "
        f"{state.remaining_steps}"
    )

    # ========================================================================
    # Resolved symbol
    # ========================================================================

    heading(
        "RESOLVED SYMBOL"
    )

    if (
        result.resolved_symbol_id
        is None
    ):
        print(
            "(none)"
        )

    else:
        print(
            f"Symbol ID           : "
            f"{result.resolved_symbol_id}"
        )

        print(
            f"Qualified name      : "
            f"{result.resolved_qualified_name}"
        )

    # ========================================================================
    # Investigation steps
    # ========================================================================

    heading(
        "INVESTIGATION STEPS"
    )

    if not state.steps:
        print(
            "(none)"
        )

    for step in state.steps:

        print(
            f"[{step.step_number}] "
            f"{step.action.value}"
        )

        separator()

        print(
            f"Query        : "
            f"{step.query}"
        )

        print(
            f"Reason       : "
            f"{step.reason}"
        )

        print(
            f"Observations : "
            f"{step.observations}"
        )

        print()

    # ========================================================================
    # Graph facts
    # ========================================================================

    heading(
        "GRAPH FACTS"
    )

    if not result.graph_facts:
        print(
            "(none)"
        )

    else:
        for index, fact in enumerate(
            result.graph_facts,
            start=1,
        ):

            print(
                f"[{index}] "
                f"{fact.source_qualified_name}"
            )

            print(
                f"    --"
                f"{fact.relationship_type.value}"
                f"--> "
                f"{fact.target_qualified_name}"
            )

    # ========================================================================
    # Evidence
    # ========================================================================

    heading(
        "SOURCE-BACKED EVIDENCE"
    )

    bundle = (
        result.evidence
    )

    if bundle is None:
        print(
            "(no evidence bundle)"
        )

    else:
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

        print()

        for index, item in enumerate(
            bundle.items,
            start=1,
        ):

            separator("=")

            title = (
                item.qualified_name
                or item.relative_path
                or item.result_key
            )

            print(
                f"[{index}] {title}"
            )

            separator("=")

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

            if item.start_line is not None:
                print(
                    f"Lines              : "
                    f"{item.start_line}"
                    f"-"
                    f"{item.end_line}"
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

            # ================================================================
            # Evidence graph context
            # ================================================================

            print()
            print(
                "Graph context"
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

                    if (
                        relation.direction
                        == "outgoing"
                    ):
                        arrow = "->"
                    else:
                        arrow = "<-"

                    line_info = ""

                    if (
                        relation.line_number
                        is not None
                    ):
                        line_info = (
                            f" [line "
                            f"{relation.line_number}"
                            f"]"
                        )

                    print(
                        f"{arrow} "
                        f"{relation.relationship_type.value:<14} "
                        f"{relation.related_qualified_name}"
                        f"{line_info}"
                    )

            # ================================================================
            # Source
            # ================================================================

            print()
            print(
                "Source"
            )

            separator()

            print(
                item.source_text.rstrip()
            )

            print()

    # ========================================================================
    # Final summary
    # ========================================================================

    heading(
        "AGENT SUMMARY"
    )

    print(
        f"Intent classified as : "
        f"{state.intent.value}"
    )

    print(
        f"Investigation status : "
        f"{state.status.value}"
    )

    print(
        f"Actions executed     : "
        f"{state.step_count}"
    )

    print(
        f"Graph facts found    : "
        f"{len(result.graph_facts)}"
    )

    print(
        f"Evidence items       : "
        f"{len(bundle.items) if bundle else 0}"
    )

    print()

    separator("=")

    print(
        "RepoBrain Phase 7 agent inspection completed."
    )

    separator("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )