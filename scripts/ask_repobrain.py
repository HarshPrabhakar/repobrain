from __future__ import annotations

import argparse
from pathlib import Path

from repobrain.agent import (
    AgentToolbox,
    RepoBrainAgentOrchestrator,
)

from repobrain.answering import (
    GroundedAnswerGenerator,
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

from repobrain.llm import (
    OllamaLLMProvider,
)

from repobrain.models.answer import (
    AnswerCitationKind,
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
# Constants
# ============================================================================

WIDTH = 110


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask RepoBrain questions about a local repository "
            "using deterministic repository evidence and a local "
            "Ollama language model."
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help="Path to the repository to analyze.",
    )

    parser.add_argument(
        "query",
        type=str,
        help="Question to ask about the repository.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:14b",
        help=(
            "Ollama model name. "
            "Default: qwen2.5-coder:14b"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help=(
            "Ollama server URL. "
            "Default: http://localhost:11434"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help=(
            "Generation temperature. "
            "Default: 0.2"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help=(
            "Maximum number of deterministic "
            "agent investigation steps."
        ),
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=20,
        help=(
            "Number of hybrid retrieval results "
            "available to the evidence layer."
        ),
    )

    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=8,
        help=(
            "Maximum number of evidence items "
            "assembled for one query."
        ),
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=24_000,
        help=(
            "Maximum source-evidence character "
            "budget for one query."
        ),
    )

    parser.add_argument(
        "--max-graph-relations",
        type=int,
        default=5,
        help=(
            "Maximum graph-context relationships "
            "attached to each evidence item."
        ),
    )

    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help=(
            "Print source evidence after the answer."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Console helpers
# ============================================================================


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
# Validation
# ============================================================================


def validate_args(
    args: argparse.Namespace,
) -> None:
    repository = (
        args.repository
        .expanduser()
        .resolve()
    )

    if not repository.exists():
        raise FileNotFoundError(
            f"Repository does not exist: "
            f"{repository}"
        )

    if not repository.is_dir():
        raise NotADirectoryError(
            f"Repository path is not a directory: "
            f"{repository}"
        )

    if not args.query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    if args.max_steps <= 0:
        raise ValueError(
            "--max-steps must be greater than 0."
        )

    if args.retrieval_top_k <= 0:
        raise ValueError(
            "--retrieval-top-k must be greater than 0."
        )

    if args.max_evidence_items <= 0:
        raise ValueError(
            "--max-evidence-items must be greater than 0."
        )

    if args.max_characters <= 0:
        raise ValueError(
            "--max-characters must be greater than 0."
        )

    if args.max_graph_relations < 0:
        raise ValueError(
            "--max-graph-relations must be >= 0."
        )

    if args.temperature < 0:
        raise ValueError(
            "--temperature must be >= 0."
        )


# ============================================================================
# Main pipeline
# ============================================================================


def main() -> int:
    args = parse_args()

    validate_args(
        args
    )

    repository_path = (
        args.repository
        .expanduser()
        .resolve()
    )

    print()
    separator("=")

    print(
        "REPOBRAIN"
    )

    print(
        "Evidence-Grounded Repository QA"
    )

    separator("=")

    print()

    # ========================================================================
    # 1. Repository scanner
    # ========================================================================

    print(
        "[1/12] Scanning repository..."
    )

    scanner = (
        RepositoryScanner()
    )

    scan_result = (
        scanner.scan(
            repository_path
        )
    )

    print(
        f"        Files discovered : "
        f"{len(scan_result.files)}"
    )

    # ========================================================================
    # 2. Python AST analysis
    # ========================================================================

    print(
        "[2/12] Extracting Python AST..."
    )

    analyzer = (
        PythonRepositoryAnalyzer()
    )

    analysis = (
        analyzer.analyze(
            scan_result
        )
    )

    print(
        f"        Symbols           : "
        f"{len(analysis.symbols)}"
    )

    print(
        f"        Relationships     : "
        f"{len(analysis.relationships)}"
    )

    # ========================================================================
    # 3. Deterministic symbol resolution
    # ========================================================================

    print(
        "[3/12] Resolving repository symbols..."
    )

    resolver = (
        PythonSymbolResolver()
    )

    (
        resolved_analysis,
        resolution_summary,
    ) = (
        resolver.resolve_repository(
            analysis
        )
    )

    print(
        f"        Resolved symbols  : "
        f"{len(resolved_analysis.symbols)}"
    )

    # ========================================================================
    # 4. Repository chunks
    # ========================================================================

    print(
        "[4/12] Building repository chunks..."
    )

    chunk_builder = (
        RepositoryChunkBuilder()
    )

    chunks = (
        chunk_builder.build(
            scan_result=scan_result,
            analysis=resolved_analysis,
        )
    )

    print(
        f"        Chunks            : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # 5. Symbol search
    # ========================================================================

    print(
        "[5/12] Building symbol index..."
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
        f"        Indexed symbols   : "
        f"{len(symbol_index)}"
    )

    # ========================================================================
    # 6. BM25 lexical search
    # ========================================================================

    print(
        "[6/12] Building BM25 index..."
    )

    lexical_engine = (
        BM25Index(
            chunks
        )
    )

    print(
        f"        BM25 documents    : "
        f"{len(chunks)}"
    )

    # ========================================================================
    # 7. Semantic retrieval
    # ========================================================================

    print(
        "[7/12] Loading semantic model..."
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    print(
        f"        Embedding model   : "
        f"{embedding_provider.model_name}"
    )

    print(
        f"        Device            : "
        f"{embedding_provider.device}"
    )

    print(
        f"        Dimension         : "
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
        "        Semantic index    : READY"
    )

    # ========================================================================
    # 8. Knowledge graph
    # ========================================================================

    print(
        "[8/12] Building knowledge graph..."
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

    graph_stats = (
        graph.stats()
    )

    print(
        f"        Graph nodes       : "
        f"{graph_stats.nodes}"
    )

    print(
        f"        Graph edges       : "
        f"{graph_stats.edges}"
    )

    graph_expander = (
        GraphEvidenceExpander(
            graph
        )
    )

    # ========================================================================
    # 9. Hybrid retrieval
    # ========================================================================

    print(
        "[9/12] Building hybrid retrieval..."
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
        "        Hybrid engine     : READY"
    )

    # ========================================================================
    # 10. Evidence assembly
    # ========================================================================

    print(
        "[10/12] Building evidence assembler..."
    )

    evidence_budget = (
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
            budget=evidence_budget,
            max_graph_relations=(
                args.max_graph_relations
            ),
        )
    )

    print(
        "        Evidence layer    : READY"
    )

    # ========================================================================
    # 11. Deterministic agent
    # ========================================================================

    print(
        "[11/12] Running agent investigation..."
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

    agent_result = (
        agent.run(
            args.query
        )
    )

    print(
        f"        Intent            : "
        f"{agent_result.state.intent.value}"
    )

    print(
        f"        Agent status      : "
        f"{agent_result.state.status.value}"
    )

    print(
        f"        Steps             : "
        f"{agent_result.state.step_count}"
    )

    print(
        f"        Graph facts       : "
        f"{len(agent_result.graph_facts)}"
    )

    evidence_count = 0

    if (
        agent_result.evidence
        is not None
    ):
        evidence_count = len(
            agent_result.evidence.items
        )

    print(
        f"        Evidence items    : "
        f"{evidence_count}"
    )

    # ========================================================================
    # 12. Local Ollama answer generation
    # ========================================================================

    print(
        "[12/12] Generating grounded answer with Ollama..."
    )

    llm_provider = (
        OllamaLLMProvider(
            model_name=(
                args.model
            ),
            host=(
                args.ollama_host
            ),
            temperature=(
                args.temperature
            ),
        )
    )

    answer_generator = (
        GroundedAnswerGenerator(
            provider=(
                llm_provider
            )
        )
    )

    answer = (
        answer_generator.generate(
            agent_result
        )
    )

    print(
        f"        LLM model         : "
        f"{answer.model_name}"
    )

    # ========================================================================
    # Question
    # ========================================================================

    heading(
        "QUESTION"
    )

    print(
        args.query
    )

    # ========================================================================
    # Answer
    # ========================================================================

    heading(
        "REPOBRAIN ANSWER"
    )

    print(
        answer.answer_text
    )

    # ========================================================================
    # Validated citations
    # ========================================================================

    heading(
        "VALIDATED CITATIONS"
    )

    if not answer.citations:

        print(
            "(none)"
        )

    else:

        for citation in (
            answer.citations
        ):

            print(
                f"[{citation.citation_id}]"
            )

            print(
                f"  Type   : "
                f"{citation.citation_kind.value}"
            )

            # ================================================================
            # SOURCE citation
            # ================================================================

            if (
                citation.citation_kind
                == AnswerCitationKind.SOURCE
            ):

                if (
                    citation.qualified_name
                ):

                    print(
                        f"  Symbol : "
                        f"{citation.qualified_name}"
                    )

                if (
                    citation.relative_path
                ):

                    print(
                        f"  File   : "
                        f"{citation.relative_path}"
                    )

                if (
                    citation.start_line
                    is not None
                ):

                    print(
                        f"  Lines  : "
                        f"{citation.start_line}"
                        f"-"
                        f"{citation.end_line}"
                    )

            # ================================================================
            # GRAPH citation
            # ================================================================

            elif (
                citation.citation_kind
                == AnswerCitationKind.GRAPH
            ):

                relationship_name = (
                    citation.relationship_type.value
                    if (
                        citation.relationship_type
                        is not None
                    )
                    else "UNKNOWN"
                )

                print(
                    f"  Fact   : "
                    f"{citation.source_qualified_name}"
                )

                print(
                    f"           --"
                    f"{relationship_name}"
                    f"--> "
                    f"{citation.target_qualified_name}"
                )

            print()

    # ========================================================================
    # Invalid citations
    # ========================================================================

    if (
        answer.invalid_citation_ids
    ):

        heading(
            "INVALID CITATIONS"
        )

        for citation_id in (
            answer.invalid_citation_ids
        ):

            print(
                f"- [{citation_id}]"
            )

    # ========================================================================
    # Grounding status
    # ========================================================================

    heading(
        "GROUNDING STATUS"
    )

    print(
        f"Grounded                 : "
        f"{answer.grounded}"
    )

    print(
        f"Evidence available       : "
        f"{answer.evidence_items_available}"
    )

    print(
        f"Graph facts available    : "
        f"{answer.graph_facts_available}"
    )

    print(
        f"Validated citations      : "
        f"{len(answer.citations)}"
    )

    print(
        f"Invalid citations        : "
        f"{len(answer.invalid_citation_ids)}"
    )

    # ========================================================================
    # Optional source evidence display
    # ========================================================================

    if (
        args.show_evidence
        and agent_result.evidence
        is not None
    ):

        heading(
            "SOURCE EVIDENCE"
        )

        bundle = (
            agent_result.evidence
        )

        for index, item in enumerate(
            bundle.items,
            start=1,
        ):

            print(
                f"[RAW-E{index}] "
                f"{item.qualified_name or item.relative_path}"
            )

            separator()

            if (
                item.relative_path
            ):

                print(
                    f"Path     : "
                    f"{item.relative_path}"
                )

            if (
                item.start_line
                is not None
            ):

                print(
                    f"Lines    : "
                    f"{item.start_line}"
                    f"-"
                    f"{item.end_line}"
                )

            print(
                f"Type     : "
                f"{item.evidence_kind.value}"
            )

            if (
                item.retrieval_channels
            ):

                print(
                    "Channels : "
                    + ", ".join(
                        channel.value
                        for channel
                        in item.retrieval_channels
                    )
                )

            if (
                item.graph_context
            ):

                print(
                    "Graph    :"
                )

                for relation in (
                    item.graph_context
                ):

                    print(
                        f"  "
                        f"{relation.direction} "
                        f"{relation.relationship_type.value} "
                        f"{relation.related_qualified_name}"
                    )

            print()
            print(
                item.source_text.rstrip()
            )
            print()

    # ========================================================================
    # Optional deterministic graph facts display
    # ========================================================================

    if (
        args.show_evidence
        and agent_result.graph_facts
    ):

        heading(
            "GRAPH FACTS"
        )

        for index, fact in enumerate(
            agent_result.graph_facts,
            start=1,
        ):

            print(
                f"[RAW-G{index}] "
                f"{fact.source_qualified_name} "
                f"--"
                f"{fact.relationship_type.value}"
                f"--> "
                f"{fact.target_qualified_name}"
            )

    # ========================================================================
    # Footer
    # ========================================================================

    print()
    separator("=")

    if answer.grounded:

        print(
            "RepoBrain query completed: GROUNDED."
        )

    else:

        print(
            "RepoBrain query completed: NOT FULLY GROUNDED."
        )

    separator("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )