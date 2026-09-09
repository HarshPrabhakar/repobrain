from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from repobrain.agent.orchestrator import (
    RepoBrainAgentOrchestrator,
)

from repobrain.agent.tools import (
    AgentToolbox,
)

from repobrain.answering.generator import (
    GroundedAnswerGenerator,
)

from repobrain.config import (
    ScannerConfig,
)

from repobrain.conversation import (
    StatefulGroundedAnsweringOrchestrator,
    StatefulRepoBrainOrchestrator,
    UnresolvedFollowUpReferenceError,
)

from repobrain.evidence.assembler import (
    EvidenceAssembler,
)

from repobrain.evidence.budgeting import (
    EvidenceBudget,
)

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.indexing import (
    PythonSymbolResolver,
    SymbolIndex,
)

from repobrain.indexing.symbol_index import (
    SymbolIndex,
)

from repobrain.ingestion.scanner import (
    RepositoryScanner,
)

from repobrain.llm import (
    OllamaLLMProvider,
)

from repobrain.parsing.python_ast import (
    PythonRepositoryAnalyzer,
)

from repobrain.retrieval import (
    GraphEvidenceExpander,
    HybridRetrievalEngine,
    SemanticSearchEngine,
    SymbolSearchEngine,
)

from repobrain.retrieval.lexical import (
    BM25Index,
)

from repobrain.retrieval.semantic import (
    SemanticSearchEngine,
)

from repobrain.embeddings import (
    SentenceTransformerEmbeddingProvider,
)

from repobrain.retrieval.chunks import (
    RepositoryChunkBuilder,
)

SEPARATOR = (
    "=" * 110
)


def _print_header(
    title: str,
) -> None:

    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def _print_step(
    index: int,
    total: int,
    text: str,
) -> None:

    print(
        f"[{index}/{total}] {text}"
    )


def _format_tuple(
    values: Iterable[str],
) -> str:

    items = list(
        values
    )

    if not items:
        return "(none)"

    return ", ".join(
        items
    )


def build_pipeline(
    repository_root: Path,
    *,
    ollama_base_url: str,
    ollama_model: str,
    embedding_model: str,
) -> StatefulGroundedAnsweringOrchestrator:

    total_steps = 12

    # ==================================================================
    # 1. Repository scanner
    # ==================================================================

    _print_step(
        1,
        total_steps,
        "Scanning repository...",
    )

    scanner = RepositoryScanner(
        ScannerConfig()
    )

    scan_result = scanner.scan(
        repository_root
    )

    print(
        f"        Files discovered : "
        f"{len(scan_result.files)}"
    )

    # ==================================================================
    # 2. Python AST
    # ==================================================================

    _print_step(
        2,
        total_steps,
        "Extracting Python AST...",
    )

    analyzer = PythonRepositoryAnalyzer()

    analysis = analyzer.analyze(
        scan_result
    )

    symbols = list(
        analysis.symbols
    )

    relationships = list(
        analysis.relationships
    )

    print(
        f"        Symbols           : "
        f"{len(symbols)}"
    )

    print(
        f"        Relationships     : "
        f"{len(relationships)}"
    )

    # ==================================================================
    # 3. Symbol resolution
    # ==================================================================

    _print_step(
        3,
        total_steps,
        "Resolving repository symbols...",
    )

    symbol_index = SymbolIndex(
        symbols
    )

    resolver = (
        PythonSymbolResolver(
            symbol_index
        )
    )

    resolved_relationships = (
        resolver.resolve_relationships(
            relationships
        )
    )

    print(
        f"        Resolved symbols  : "
        f"{len(symbols)}"
    )

    # ==================================================================
    # 4. Chunk builder
    # ==================================================================

    _print_step(
        4,
        total_steps,
        "Building repository chunks...",
    )

    chunk_builder = RepositoryChunkBuilder()

    chunks = chunk_builder.build(
        scan_result=scan_result,
        symbols=symbols,
    )

    print(
        f"        Chunks            : "
        f"{len(chunks)}"
    )

    # ==================================================================
    # 5. Symbol search
    # ==================================================================

    _print_step(
        5,
        total_steps,
        "Building symbol index...",
    )

    symbol_search = (
        SymbolSearchEngine(
            symbol_index
        )
    )

    print(
        f"        Indexed symbols   : "
        f"{len(symbols)}"
    )

    # ==================================================================
    # 6. BM25
    # ==================================================================

    _print_step(
        6,
        total_steps,
        "Building BM25 index...",
    )

    lexical_search = (
        BM25Index(
            chunks
        )
    )

    print(
        f"        BM25 documents    : "
        f"{len(chunks)}"
    )

    # ==================================================================
    # 7. Semantic retrieval
    # ==================================================================

    _print_step(
        7,
        total_steps,
        "Loading semantic model...",
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=(
                embedding_model
            ),
            device="cuda",
        )
    )

    semantic_search = (
        SemanticSearchEngine(
            chunks,
            embedding_provider,
            symbols=symbols,
            relationships=(
                resolved_relationships
            ),
        )
    )

    dimension = getattr(
        embedding_provider,
        "dimension",
        None,
    )

    if callable(
        dimension
    ):
        dimension = dimension()

    print(
        f"        Embedding model   : "
        f"{embedding_model}"
    )

    print(
        "        Device            : cuda"
    )

    if dimension is not None:

        print(
            f"        Dimension         : "
            f"{dimension}"
        )

    print(
        "        Semantic index    : READY"
    )

    # ==================================================================
    # 8. Knowledge graph
    # ==================================================================

    _print_step(
        8,
        total_steps,
        "Building knowledge graph...",
    )

    graph_expander = GraphExpander(
        symbols=symbols,
        relationships=(
            resolved_relationships
        ),
    )

    graph_stats = (
        graph.stats()
    )

    node_count = getattr(
        graph_stats,
        "node_count",
        None,
    )

    edge_count = getattr(
        graph_stats,
        "edge_count",
        None,
    )

    if node_count is None:

        node_count = len(
            symbols
        )

    if edge_count is None:

        edge_count = len(
            resolved_relationships
        )

    print(
        f"        Graph nodes       : "
        f"{node_count}"
    )

    print(
        f"        Graph edges       : "
        f"{edge_count}"
    )

    # ==================================================================
    # 9. Hybrid retrieval
    # ==================================================================

    _print_step(
        9,
        total_steps,
        "Building hybrid retrieval...",
    )

    graph_expander = (
        GraphExpander(
            graph
        )
    )

    hybrid_search = (
        HybridRetrievalEngine(
            symbol_engine=(
                symbol_search
            ),
            lexical_engine=(
                lexical_search
            ),
            semantic_engine=(
                semantic_search
            ),
            graph_expander=(
                graph_expander
            ),
        )
    )

    print(
        "        Hybrid engine     : READY"
    )

    # ==================================================================
    # 10. Evidence
    # ==================================================================

    _print_step(
        10,
        total_steps,
        "Building evidence assembler...",
    )

    evidence_assembler = (
        EvidenceAssembler(
            budget=(
                EvidenceBudget(
                    max_items=8,
                    max_characters=24000,
                )
            )
        )
    )

    print(
        "        Evidence layer    : READY"
    )

    # ==================================================================
    # 11. Deterministic agent
    # ==================================================================

    _print_step(
        11,
        total_steps,
        "Building stateful agent...",
    )

    toolbox = AgentToolbox(
        hybrid_engine=(
            hybrid_search
        ),
        graph=(
            graph
        ),
        evidence_assembler=(
            evidence_assembler
        ),
        symbol_index=(
            symbol_index
        ),
    )

    base_agent = (
        RepoBrainAgentOrchestrator(
            toolbox=(
                toolbox
            )
        )
    )

    stateful_agent = (
        StatefulRepoBrainOrchestrator(
            agent_runner=(
                base_agent.run
            )
        )
    )

    print(
        "        Stateful agent    : READY"
    )

    # ==================================================================
    # 12. Grounded answering
    # ==================================================================

    _print_step(
        12,
        total_steps,
        "Loading grounded answer layer...",
    )

    llm_provider = (
        OllamaLLMProvider(
            base_url=(
                ollama_base_url
            ),
            model=(
                ollama_model
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

    stateful_grounded = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                stateful_agent
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    print(
        f"        Ollama model      : "
        f"{ollama_model}"
    )

    print(
        "        Grounded QA       : READY"
    )

    return stateful_grounded


def print_help() -> None:

    _print_header(
        "COMMANDS"
    )

    print(
        "/help"
        "      Show available commands."
    )

    print(
        "/state"
        "     Show current investigation state."
    )

    print(
        "/clear"
        "     Clear investigation context while preserving the session ID."
    )

    print(
        "/new"
        "       Start a completely new investigation session."
    )

    print(
        "/history"
        "   Show recent investigation turns."
    )

    print(
        "/quit"
        "      Exit RepoBrain."
    )

    print(
        "/exit"
        "      Exit RepoBrain."
    )


def print_state(
    orchestrator: StatefulGroundedAnsweringOrchestrator,
) -> None:

    state = (
        orchestrator.state
    )

    _print_header(
        "INVESTIGATION STATE"
    )

    print(
        f"Conversation ID     : "
        f"{state.conversation_id}"
    )

    print(
        f"Turn number         : "
        f"{state.turn_number}"
    )

    print(
        f"Current symbol      : "
        f"{state.current_qualified_name or '(none)'}"
    )

    print(
        f"Previous symbol     : "
        f"{state.previous_qualified_name or '(none)'}"
    )

    print(
        f"Previous intent     : "
        f"{(
            state.previous_intent.value
            if state.previous_intent is not None
            else '(none)'
        )}"
    )

    print(
        f"Last query          : "
        f"{state.last_query or '(none)'}"
    )

    print(
        f"Last resolved query : "
        f"{state.last_resolved_query or '(none)'}"
    )

    print(
        f"Last grounded       : "
        f"{state.last_grounded}"
    )

    print(
        f"Last citations      : "
        f"{_format_tuple(state.last_citation_ids)}"
    )

    print(
        f"Invalid citations   : "
        f"{_format_tuple(state.last_invalid_citation_ids)}"
    )

    print(
        f"History length      : "
        f"{len(state.history)}"
    )


def print_history(
    orchestrator: StatefulGroundedAnsweringOrchestrator,
) -> None:

    state = (
        orchestrator.state
    )

    _print_header(
        "INVESTIGATION HISTORY"
    )

    if not state.history:

        print(
            "(no completed turns)"
        )

        return

    for turn in state.history:

        print(
            f"Turn {turn.turn_number}"
        )

        print(
            f"  Query      : "
            f"{turn.query}"
        )

        print(
            f"  Resolved   : "
            f"{turn.resolved_query or '(none)'}"
        )

        print(
            f"  Intent     : "
            f"{(
                turn.intent.value
                if turn.intent is not None
                else '(none)'
            )}"
        )

        print(
            f"  Focus      : "
            f"{turn.focused_qualified_name or '(none)'}"
        )

        print(
            f"  Grounded   : "
            f"{turn.grounded}"
        )

        print(
            f"  Citations  : "
            f"{_format_tuple(turn.citation_ids)}"
        )

        print()


def print_turn(
    turn: object,
) -> None:

    agent_turn = getattr(
        turn,
        "agent_turn",
    )

    answer = getattr(
        turn,
        "answer",
    )

    state_after = getattr(
        turn,
        "state_after",
    )

    answer_text = (
        getattr(
            answer,
            "text",
            None,
        )
    )

    if answer_text is None:

        answer_text = (
            str(
                answer
            )
        )

    _print_header(
        "REPOBRAIN"
    )

    print(
        answer_text
    )

    print()

    print(
        f"Resolved query       : "
        f"{agent_turn.resolved_query}"
    )

    print(
        f"Intent               : "
        f"{agent_turn.agent_result.state.intent.value}"
    )

    print(
        f"Current focus        : "
        f"{state_after.current_qualified_name or '(none)'}"
    )

    print(
        f"Grounded             : "
        f"{state_after.last_grounded}"
    )

    print(
        f"Validated citations  : "
        f"{_format_tuple(state_after.last_citation_ids)}"
    )

    print(
        f"Invalid citations    : "
        f"{_format_tuple(state_after.last_invalid_citation_ids)}"
    )


def chat_loop(
    orchestrator: StatefulGroundedAnsweringOrchestrator,
) -> None:

    _print_header(
        "REPOBRAIN CHAT"
    )

    print(
        "RepoBrain is ready."
    )

    print(
        "Type /help for commands."
    )

    while True:

        print()

        try:

            query = input(
                "You > "
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            print(
                "RepoBrain > Goodbye."
            )

            break

        normalized = (
            query.strip()
        )

        if not normalized:

            continue

        command = (
            normalized.lower()
        )

        # --------------------------------------------------------------
        # Commands
        # --------------------------------------------------------------

        if command in {
            "/quit",
            "/exit",
        }:

            print(
                "RepoBrain > Goodbye."
            )

            break

        if command == "/help":

            print_help()

            continue

        if command == "/state":

            print_state(
                orchestrator
            )

            continue

        if command == "/history":

            print_history(
                orchestrator
            )

            continue

        if command == "/clear":

            old_id = (
                orchestrator.state
                .conversation_id
            )

            state = (
                orchestrator.clear()
            )

            print(
                "RepoBrain > Investigation context cleared."
            )

            print(
                f"Session ID preserved: "
                f"{state.conversation_id == old_id}"
            )

            continue

        if command == "/new":

            old_id = (
                orchestrator.state
                .conversation_id
            )

            state = (
                orchestrator
                .new_conversation()
            )

            print(
                "RepoBrain > New investigation started."
            )

            print(
                f"New session ID: "
                f"{state.conversation_id}"
            )

            print(
                f"Session changed: "
                f"{state.conversation_id != old_id}"
            )

            continue

        # --------------------------------------------------------------
        # Repository investigation
        # --------------------------------------------------------------

        try:

            turn = (
                orchestrator.run(
                    normalized
                )
            )

        except (
            UnresolvedFollowUpReferenceError
        ) as exc:

            print()

            print(
                "RepoBrain > I cannot resolve that follow-up "
                "from the current investigation state."
            )

            if (
                exc.reference_text
                is not None
            ):

                print(
                    f"Reference: "
                    f"{exc.reference_text!r}"
                )

            print(
                "Please name the function, method, class, "
                "or symbol explicitly."
            )

            continue

        except Exception as exc:

            print()

            print(
                "RepoBrain > The investigation failed."
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        print_turn(
            turn
        )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Interactive stateful RepoBrain repository chat."
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help=(
            "Repository root to investigate."
        ),
    )

    parser.add_argument(
        "--ollama-url",
        default=(
            "http://localhost:11434"
        ),
        help=(
            "Ollama server URL."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "qwen2.5-coder:14b"
        ),
        help=(
            "Ollama model name."
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            "nomic-ai/CodeRankEmbed"
        ),
        help=(
            "SentenceTransformer embedding model."
        ),
    )

    return (
        parser.parse_args()
    )


def main() -> int:

    args = (
        parse_args()
    )

    repository_root = (
        args.repository
        .expanduser()
        .resolve()
    )

    if not repository_root.exists():

        print(
            f"Repository does not exist: "
            f"{repository_root}",
            file=sys.stderr,
        )

        return 2

    if not repository_root.is_dir():

        print(
            f"Repository path is not a directory: "
            f"{repository_root}",
            file=sys.stderr,
        )

        return 2

    _print_header(
        "REPOBRAIN"
    )

    print(
        "Stateful Repository Intelligence"
    )

    print(
        f"Repository : "
        f"{repository_root}"
    )

    try:

        orchestrator = (
            build_pipeline(
                repository_root,
                ollama_base_url=(
                    args.ollama_url
                ),
                ollama_model=(
                    args.model
                ),
                embedding_model=(
                    args.embedding_model
                ),
            )
        )

    except Exception as exc:

        print()

        print(
            "Failed to initialize RepoBrain."
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 1

    chat_loop(
        orchestrator
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )