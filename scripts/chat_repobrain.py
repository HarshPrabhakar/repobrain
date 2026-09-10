from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from repobrain.application import (
    RepositoryRuntimeBuilder,
)
from repobrain.conversation import (
    StatefulGroundedAnsweringOrchestrator,
    UnresolvedFollowUpReferenceError,
)


WIDTH = 110
SEPARATOR = "=" * WIDTH

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"

DEFAULT_MAX_EVIDENCE_ITEMS = 8
DEFAULT_MAX_CHARACTERS = 24_000
DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_MAX_AGENT_STEPS = 4
DEFAULT_MAX_GRAPH_RELATIONS = 20


# =============================================================================
# Generic display helpers
# =============================================================================


def print_header(
    title: str,
) -> None:

    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def enum_value(
    value: object,
) -> str:

    if value is None:
        return "(none)"

    result = getattr(
        value,
        "value",
        None,
    )

    if result is not None:
        return str(result)

    return str(value)


def format_values(
    values: Iterable[object] | None,
) -> str:

    if values is None:
        return "(none)"

    items = [
        str(value)
        for value in values
        if value is not None
        and str(value).strip()
    ]

    if not items:
        return "(none)"

    return ", ".join(
        items
    )


def first_existing_attribute(
    obj: object,
    *names: str,
    default: object = None,
) -> object:

    for name in names:

        if hasattr(
            obj,
            name,
        ):

            return getattr(
                obj,
                name,
            )

    return default


# =============================================================================
# Phase 10.2 runtime-builder progress display
# =============================================================================


def print_build_progress(
    step: int,
    total: int,
    message: str,
    details: dict[str, object],
) -> None:
    """
    CLI-only rendering of RepositoryRuntimeBuilder progress.

    The application builder reports structured progress.
    This function decides how that progress appears in the terminal.
    """

    if not details:

        print(
            f"[{step}/{total}] {message}"
        )

        return

    labels = {
        "files_discovered": "Files discovered",
        "symbols": "Symbols",
        "relationships": "Relationships",
        "resolved_symbols": "Resolved symbols",
        "chunks": "Chunks",
        "indexed_symbols": "Indexed symbols",
        "bm25_documents": "BM25 documents",
        "embedding_model": "Embedding model",
        "device": "Device",
        "dimension": "Dimension",
        "semantic_index": "Semantic index",
        "graph_nodes": "Graph nodes",
        "graph_edges": "Graph edges",
        "hybrid_engine": "Hybrid engine",
        "evidence_layer": "Evidence layer",
        "agent": "Agent",
        "stateful_agent": "Stateful agent",
        "llm_model": "LLM model",
        "grounded_qa": "Grounded QA",
    }

    for key, value in details.items():

        label = labels.get(
            key,
            key.replace(
                "_",
                " ",
            ).title(),
        )

        print(
            f"        {label:<18}: "
            f"{value}"
        )


# =============================================================================
# State display
# =============================================================================


def print_state(
    orchestrator: (
        StatefulGroundedAnsweringOrchestrator
    ),
) -> None:

    state = (
        orchestrator.state
    )

    print_header(
        "INVESTIGATION STATE"
    )

    print(
        f"Conversation ID      : "
        f"{state.conversation_id}"
    )

    print(
        f"Turn number          : "
        f"{state.turn_number}"
    )

    print(
        f"Current symbol       : "
        f"{state.current_qualified_name or '(none)'}"
    )

    print(
        f"Previous symbol      : "
        f"{state.previous_qualified_name or '(none)'}"
    )

    print(
        f"Relationship type    : "
        f"{enum_value(state.relationship_focus_type)}"
    )

    print(
        f"Relationship symbol  : "
        f"{state.relationship_qualified_name or '(none)'}"
    )

    print(
        f"Previous intent      : "
        f"{enum_value(state.previous_intent)}"
    )

    print(
        f"Last query           : "
        f"{state.last_query or '(none)'}"
    )

    print(
        f"Last resolved query  : "
        f"{state.last_resolved_query or '(none)'}"
    )

    print(
        f"Last grounded        : "
        f"{state.last_grounded}"
    )

    print(
        f"Last citations       : "
        f"{format_values(state.last_citation_ids)}"
    )

    print(
        f"Invalid citations    : "
        f"{format_values(state.last_invalid_citation_ids)}"
    )

    print(
        f"History length       : "
        f"{len(state.history)}"
    )


# =============================================================================
# History display
# =============================================================================


def print_history(
    orchestrator: (
        StatefulGroundedAnsweringOrchestrator
    ),
) -> None:

    state = (
        orchestrator.state
    )

    print_header(
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
            f"{enum_value(turn.intent)}"
        )

        print(
            f"  Focus      : "
            f"{turn.focused_qualified_name or '(none)'}"
        )

        print(
            f"  Relation   : "
            f"{enum_value(turn.relationship_focus_type)}"
        )

        print(
            f"  Rel. Focus : "
            f"{turn.relationship_qualified_name or '(none)'}"
        )

        print(
            f"  Grounded   : "
            f"{turn.grounded}"
        )

        print(
            f"  Citations  : "
            f"{format_values(turn.citation_ids)}"
        )

        print(
            f"  Invalid    : "
            f"{format_values(turn.invalid_citation_ids)}"
        )

        print()


# =============================================================================
# Answer display
# =============================================================================


def answer_text(
    answer: object,
) -> str:

    text = first_existing_attribute(
        answer,
        "answer_text",
        "text",
        "answer",
        "content",
        default=None,
    )

    if text is None:
        return str(answer)

    return str(text)


def print_turn(
    turn: object,
) -> None:

    answer = (
        turn.answer
    )

    agent_turn = (
        turn.agent_turn
    )

    state_after = (
        turn.state_after
    )

    agent_result = (
        agent_turn.agent_result
    )

    print_header(
        "REPOBRAIN ANSWER"
    )

    print(
        answer_text(
            answer
        )
    )

    print()

    print(
        f"Original query       : "
        f"{agent_turn.original_query}"
    )

    print(
        f"Resolved query       : "
        f"{agent_turn.resolved_query}"
    )

    result_state = getattr(
        agent_result,
        "state",
        None,
    )

    intent = (
        getattr(
            result_state,
            "intent",
            None,
        )
        if result_state is not None
        else None
    )

    print(
        f"Intent               : "
        f"{enum_value(intent)}"
    )

    print(
        f"Current focus        : "
        f"{state_after.current_qualified_name or '(none)'}"
    )

    print(
        f"Relationship type    : "
        f"{enum_value(state_after.relationship_focus_type)}"
    )

    print(
        f"Relationship focus   : "
        f"{state_after.relationship_qualified_name or '(none)'}"
    )

    print(
        f"Grounded             : "
        f"{state_after.last_grounded}"
    )

    print(
        f"Validated citations  : "
        f"{format_values(state_after.last_citation_ids)}"
    )

    print(
        f"Invalid citations    : "
        f"{format_values(state_after.last_invalid_citation_ids)}"
    )


# =============================================================================
# Help
# =============================================================================


def print_help() -> None:

    print_header(
        "REPOBRAIN CHAT COMMANDS"
    )

    print(
        "/help      Show this help."
    )

    print(
        "/state     Show current investigation state."
    )

    print(
        "/history   Show completed turns."
    )

    print(
        "/clear     Clear investigation context but keep "
        "the current conversation ID."
    )

    print(
        "/new       Start a completely new investigation."
    )

    print(
        "/quit      Exit RepoBrain."
    )

    print(
        "/exit      Exit RepoBrain."
    )


# =============================================================================
# Interactive loop
# =============================================================================


def chat_loop(
    orchestrator: (
        StatefulGroundedAnsweringOrchestrator
    ),
) -> None:

    print_header(
        "REPOBRAIN CHAT"
    )

    print(
        "RepoBrain is ready."
    )

    print(
        "Repository intelligence is owned by a reusable "
        "Phase 10 runtime."
    )

    print(
        "This conversation has independent Phase 9 state."
    )

    print(
        "Type /help for commands."
    )

    while True:

        print()

        try:

            raw_query = input(
                "You > "
            )

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()

            print(
                "RepoBrain > Goodbye."
            )

            return

        query = (
            raw_query.strip()
        )

        if not query:
            continue

        command = (
            query.lower()
        )

        if command in {
            "/quit",
            "/exit",
        }:

            print(
                "RepoBrain > Goodbye."
            )

            return

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
                orchestrator
                .state
                .conversation_id
            )

            orchestrator.clear()

            new_id = (
                orchestrator
                .state
                .conversation_id
            )

            print(
                "RepoBrain > Investigation context cleared."
            )

            print(
                f"Conversation ID preserved: "
                f"{old_id == new_id}"
            )

            continue

        if command == "/new":

            old_id = (
                orchestrator
                .state
                .conversation_id
            )

            orchestrator.new_conversation()

            new_id = (
                orchestrator
                .state
                .conversation_id
            )

            print(
                "RepoBrain > New investigation started."
            )

            print(
                f"Conversation ID: "
                f"{new_id}"
            )

            print(
                f"Conversation changed: "
                f"{old_id != new_id}"
            )

            continue

        try:

            turn = (
                orchestrator.run(
                    query
                )
            )

        except UnresolvedFollowUpReferenceError as exc:

            print()

            print(
                "RepoBrain > I cannot resolve that follow-up "
                "from the current investigation state."
            )

            reference = getattr(
                exc,
                "reference_text",
                None,
            )

            if reference:

                print(
                    f"Reference: "
                    f"{reference!r}"
                )

            print(
                "Please name the function, method, class, "
                "or symbol explicitly."
            )

            continue

        except Exception as exc:

            print()

            print(
                "RepoBrain > Investigation failed."
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        print_turn(
            turn
        )


# =============================================================================
# CLI arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Interactive multi-turn RepoBrain repository intelligence."
        )
    )

    parser.add_argument(
        "repository",
        type=Path,
        help=(
            "Local repository root to investigate."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            DEFAULT_OLLAMA_MODEL
        ),
        help=(
            "Ollama model name."
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default=(
            DEFAULT_OLLAMA_HOST
        ),
        help=(
            "Ollama server URL."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "LLM generation temperature."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=(
            DEFAULT_MAX_AGENT_STEPS
        ),
        help=(
            "Maximum deterministic agent steps."
        ),
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=(
            DEFAULT_RETRIEVAL_TOP_K
        ),
        help=(
            "Maximum initial hybrid retrieval results."
        ),
    )

    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=(
            DEFAULT_MAX_EVIDENCE_ITEMS
        ),
        help=(
            "Maximum evidence items supplied to answering."
        ),
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=(
            DEFAULT_MAX_CHARACTERS
        ),
        help=(
            "Maximum evidence character budget."
        ),
    )

    parser.add_argument(
        "--max-graph-relations",
        type=int,
        default=(
            DEFAULT_MAX_GRAPH_RELATIONS
        ),
        help=(
            "Maximum graph relations used per investigation."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================


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

    print_header(
        "REPOBRAIN"
    )

    print(
        "Stateful Repository Intelligence"
    )

    print(
        f"Repository : "
        f"{repository_root}"
    )

    print(
        f"LLM        : "
        f"{args.model}"
    )

    try:

        builder = (
            RepositoryRuntimeBuilder(
                ollama_host=(
                    args.ollama_host
                ),
                ollama_model=(
                    args.model
                ),
                temperature=(
                    args.temperature
                ),
                max_steps=(
                    args.max_steps
                ),
                retrieval_top_k=(
                    args.retrieval_top_k
                ),
                max_evidence_items=(
                    args.max_evidence_items
                ),
                max_characters=(
                    args.max_characters
                ),
                max_graph_relations=(
                    args.max_graph_relations
                ),
                progress_callback=(
                    print_build_progress
                ),
            )
        )

        runtime = (
            builder.build(
                repository_root
            )
        )

        orchestrator = (
            runtime.create_conversation()
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

    snapshot = (
        runtime.snapshot()
    )

    print()

    print(
        f"Runtime ID : "
        f"{snapshot.runtime_id}"
    )

    print(
        f"Conversations created : "
        f"{snapshot.conversation_count}"
    )

    chat_loop(
        orchestrator
    )

    runtime.close()

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
